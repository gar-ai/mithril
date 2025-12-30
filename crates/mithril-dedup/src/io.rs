//! File I/O for JSONL and Parquet formats.
//!
//! Provides efficient reading and writing of document datasets.

use arrow::array::{Array, ArrayRef, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Document representation for deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier.
    pub id: u64,
    /// Document text content.
    pub text: String,
    /// Original line number in the input file (for tracking).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_number: Option<usize>,
}

/// Rich document with all original columns preserved.
///
/// This is used when reading Parquet files where we want to preserve
/// all columns, not just id and text, for HuggingFace Datasets compatibility.
#[derive(Debug, Clone)]
pub struct RichDocument {
    /// Unique document identifier.
    pub id: u64,
    /// Document text content (from the specified text column).
    pub text: String,
    /// Name of the text column.
    pub text_column: String,
    /// All columns from the original record (as Arrow arrays).
    /// Maps column name to the value at this row.
    pub columns: std::collections::HashMap<String, ColumnValue>,
    /// Original row index in the batch.
    pub row_index: usize,
}

/// A single column value that can be serialized back to Parquet.
#[derive(Debug, Clone)]
pub enum ColumnValue {
    Null,
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    Binary(Vec<u8>),
    /// For unsupported types, store as JSON string
    Json(String),
}

impl Document {
    /// Create a new document.
    #[must_use]
    pub fn new(id: u64, text: String) -> Self {
        Self {
            id,
            text,
            line_number: None,
        }
    }

    /// Create a document with line number tracking.
    #[must_use]
    pub fn with_line_number(id: u64, text: String, line_number: usize) -> Self {
        Self {
            id,
            text,
            line_number: Some(line_number),
        }
    }
}

/// Errors that can occur during I/O operations.
#[derive(Error, Debug)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Field '{field}' not found or not a string at line {line}")]
    MissingField { field: String, line: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Column '{column}' not found in Parquet file")]
    ColumnNotFound { column: String },

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for I/O operations.
pub type Result<T> = std::result::Result<T, IoError>;

/// Read documents from a JSONL file.
///
/// Each line should be a JSON object with the specified text field.
/// The document ID is taken from an "id" field if present, otherwise
/// the line number is used.
///
/// # Arguments
/// * `path` - Path to the JSONL file
/// * `text_field` - Name of the field containing the text
///
/// # Returns
/// Vector of documents
pub fn read_jsonl<P: AsRef<Path>>(path: P, text_field: &str) -> Result<Vec<Document>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = serde_json::from_str(&line).map_err(|e| IoError::Parse {
            line: line_num + 1,
            message: e.to_string(),
        })?;

        let text = json
            .get(text_field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| IoError::MissingField {
                field: text_field.to_string(),
                line: line_num + 1,
            })?;

        // Try to get ID from JSON, otherwise use line number
        let id = json
            .get("id")
            .and_then(|v| v.as_u64())
            .unwrap_or(line_num as u64);

        documents.push(Document::with_line_number(
            id,
            text.to_string(),
            line_num + 1,
        ));
    }

    Ok(documents)
}

/// Read documents from a JSONL file, preserving original JSON.
///
/// Returns both the parsed document and the original JSON line,
/// which is useful for writing back to output while preserving other fields.
///
/// # Arguments
/// * `path` - Path to the JSONL file
/// * `text_field` - Name of the field containing the text
///
/// # Returns
/// Vector of (Document, original_json_line) tuples
pub fn read_jsonl_with_original<P: AsRef<Path>>(
    path: P,
    text_field: &str,
) -> Result<Vec<(Document, String)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = serde_json::from_str(&line).map_err(|e| IoError::Parse {
            line: line_num + 1,
            message: e.to_string(),
        })?;

        let text = json
            .get(text_field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| IoError::MissingField {
                field: text_field.to_string(),
                line: line_num + 1,
            })?;

        let id = json
            .get("id")
            .and_then(|v| v.as_u64())
            .unwrap_or(line_num as u64);

        let doc = Document::with_line_number(id, text.to_string(), line_num + 1);
        documents.push((doc, line));
    }

    Ok(documents)
}

/// Write documents to a JSONL file.
///
/// Each document is written as a JSON object on its own line.
///
/// # Arguments
/// * `path` - Path to the output file
/// * `docs` - Documents to write
pub fn write_jsonl<P: AsRef<Path>>(path: P, docs: &[Document]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for doc in docs {
        let json = serde_json::json!({
            "id": doc.id,
            "text": doc.text
        });
        writeln!(writer, "{}", serde_json::to_string(&json).unwrap())?;
    }

    writer.flush()?;
    Ok(())
}

/// Write original JSON lines to a file (for preserving other fields).
///
/// # Arguments
/// * `path` - Path to the output file
/// * `lines` - Original JSON lines to write
pub fn write_jsonl_lines<P: AsRef<Path>>(path: P, lines: &[String]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for line in lines {
        writeln!(writer, "{line}")?;
    }

    writer.flush()?;
    Ok(())
}

/// Streaming JSONL reader for memory-efficient processing.
pub struct JsonlReader {
    reader: BufReader<File>,
    text_field: String,
    line_num: usize,
}

impl JsonlReader {
    /// Create a new streaming JSONL reader.
    pub fn new<P: AsRef<Path>>(path: P, text_field: &str) -> Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
            text_field: text_field.to_string(),
            line_num: 0,
        })
    }
}

impl Iterator for JsonlReader {
    type Item = Result<(Document, String)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None, // EOF
                Ok(_) => {
                    self.line_num += 1;
                    if line.trim().is_empty() {
                        continue;
                    }

                    let json: serde_json::Value = match serde_json::from_str(&line) {
                        Ok(v) => v,
                        Err(e) => {
                            return Some(Err(IoError::Parse {
                                line: self.line_num,
                                message: e.to_string(),
                            }))
                        }
                    };

                    let text = match json.get(&self.text_field).and_then(|v| v.as_str()) {
                        Some(t) => t.to_string(),
                        None => {
                            return Some(Err(IoError::MissingField {
                                field: self.text_field.clone(),
                                line: self.line_num,
                            }))
                        }
                    };

                    let id = json
                        .get("id")
                        .and_then(|v| v.as_u64())
                        .unwrap_or((self.line_num - 1) as u64);

                    let doc = Document::with_line_number(id, text, self.line_num);
                    return Some(Ok((doc, line.trim_end().to_string())));
                }
                Err(e) => return Some(Err(IoError::Io(e))),
            }
        }
    }
}

// =============================================================================
// Parquet I/O
// =============================================================================

/// Read documents from a Parquet file.
///
/// Reads the specified text column and optional ID column from a Parquet file.
/// Compatible with HuggingFace Datasets format.
///
/// # Arguments
/// * `path` - Path to the Parquet file
/// * `text_column` - Name of the column containing the text
///
/// # Returns
/// Vector of documents
pub fn read_parquet<P: AsRef<Path>>(path: P, text_column: &str) -> Result<Vec<Document>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut documents = Vec::new();
    let mut doc_id: u64 = 0;

    for batch_result in reader {
        let batch = batch_result?;

        // Find the text column
        let text_col_idx =
            batch
                .schema()
                .index_of(text_column)
                .map_err(|_| IoError::ColumnNotFound {
                    column: text_column.to_string(),
                })?;

        let text_array = batch
            .column(text_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                IoError::InvalidConfig(format!("Column '{text_column}' is not a string column"))
            })?;

        // Try to find an ID column
        let id_col_idx = batch.schema().index_of("id").ok();
        let id_array =
            id_col_idx.and_then(|idx| batch.column(idx).as_any().downcast_ref::<UInt64Array>());

        for row in 0..batch.num_rows() {
            if let Some(text) = text_array.value(row).into() {
                let id = id_array
                    .and_then(|arr| {
                        if arr.is_null(row) {
                            None
                        } else {
                            Some(arr.value(row))
                        }
                    })
                    .unwrap_or_else(|| {
                        let current_id = doc_id;
                        doc_id += 1;
                        current_id
                    });

                documents.push(Document::new(id, text.to_string()));
            }
        }
    }

    Ok(documents)
}

/// Write documents to a Parquet file.
///
/// Creates a Parquet file with 'id' and 'text' columns.
/// Compatible with HuggingFace Datasets format.
///
/// # Arguments
/// * `path` - Path to the output file
/// * `docs` - Documents to write
pub fn write_parquet<P: AsRef<Path>>(path: P, docs: &[Document]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("text", DataType::Utf8, false),
    ]));

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Write in batches of 10000 for memory efficiency
    const BATCH_SIZE: usize = 10000;

    for chunk in docs.chunks(BATCH_SIZE) {
        let ids: Vec<u64> = chunk.iter().map(|d| d.id).collect();
        let texts: Vec<&str> = chunk.iter().map(|d| d.text.as_str()).collect();

        let id_array: ArrayRef = Arc::new(UInt64Array::from(ids));
        let text_array: ArrayRef = Arc::new(StringArray::from(texts));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, text_array])?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

// =============================================================================
// Enhanced Parquet I/O with column preservation
// =============================================================================

use arrow::array::{
    BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, UInt16Array, UInt32Array, UInt8Array,
};

impl RichDocument {
    /// Convert to a simple Document (loses extra columns).
    #[must_use]
    pub fn to_document(&self) -> Document {
        Document::new(self.id, self.text.clone())
    }
}

impl ColumnValue {
    /// Extract a value from an Arrow array at the given row.
    fn from_array(array: &dyn Array, row: usize) -> Self {
        use arrow::datatypes::DataType;

        if array.is_null(row) {
            return ColumnValue::Null;
        }

        match array.data_type() {
            DataType::Boolean => {
                if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
                    return ColumnValue::Bool(arr.value(row));
                }
            }
            DataType::Int8 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
                    return ColumnValue::Int8(arr.value(row));
                }
            }
            DataType::Int16 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
                    return ColumnValue::Int16(arr.value(row));
                }
            }
            DataType::Int32 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                    return ColumnValue::Int32(arr.value(row));
                }
            }
            DataType::Int64 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                    return ColumnValue::Int64(arr.value(row));
                }
            }
            DataType::UInt8 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt8Array>() {
                    return ColumnValue::UInt8(arr.value(row));
                }
            }
            DataType::UInt16 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt16Array>() {
                    return ColumnValue::UInt16(arr.value(row));
                }
            }
            DataType::UInt32 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
                    return ColumnValue::UInt32(arr.value(row));
                }
            }
            DataType::UInt64 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
                    return ColumnValue::UInt64(arr.value(row));
                }
            }
            DataType::Float32 => {
                if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
                    return ColumnValue::Float32(arr.value(row));
                }
            }
            DataType::Float64 => {
                if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
                    return ColumnValue::Float64(arr.value(row));
                }
            }
            DataType::Utf8 => {
                if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
                    return ColumnValue::String(arr.value(row).to_string());
                }
            }
            DataType::LargeUtf8 => {
                if let Some(arr) = array
                    .as_any()
                    .downcast_ref::<arrow::array::LargeStringArray>()
                {
                    return ColumnValue::String(arr.value(row).to_string());
                }
            }
            DataType::Binary => {
                if let Some(arr) = array.as_any().downcast_ref::<BinaryArray>() {
                    return ColumnValue::Binary(arr.value(row).to_vec());
                }
            }
            _ => {
                // For unsupported types, try to convert to JSON
                return ColumnValue::Json(format!("<unsupported: {:?}>", array.data_type()));
            }
        }
        ColumnValue::Null
    }
}

/// Read documents from a Parquet file, preserving all columns.
///
/// This function reads all columns from the Parquet file, not just text and id.
/// Use this when you need to write back the data with all original columns preserved,
/// for example when filtering a HuggingFace dataset.
///
/// # Arguments
/// * `path` - Path to the Parquet file
/// * `text_column` - Name of the column containing the text
///
/// # Returns
/// Tuple of (documents, original schema) for use with `write_parquet_full`
pub fn read_parquet_full<P: AsRef<Path>>(
    path: P,
    text_column: &str,
) -> Result<(Vec<RichDocument>, Arc<Schema>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let mut documents = Vec::new();
    let mut doc_id: u64 = 0;

    for batch_result in reader {
        let batch = batch_result?;

        // Find the text column index
        let text_col_idx =
            batch
                .schema()
                .index_of(text_column)
                .map_err(|_| IoError::ColumnNotFound {
                    column: text_column.to_string(),
                })?;

        let text_array = batch
            .column(text_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                IoError::InvalidConfig(format!("Column '{text_column}' is not a string column"))
            })?;

        // Try to find an ID column
        let id_col_idx = batch.schema().index_of("id").ok();
        let id_array =
            id_col_idx.and_then(|idx| batch.column(idx).as_any().downcast_ref::<UInt64Array>());

        // Get column names
        let column_names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();

        for row in 0..batch.num_rows() {
            if text_array.is_null(row) {
                continue;
            }

            let text = text_array.value(row).to_string();

            let id = id_array
                .and_then(|arr| {
                    if arr.is_null(row) {
                        None
                    } else {
                        Some(arr.value(row))
                    }
                })
                .unwrap_or_else(|| {
                    let current_id = doc_id;
                    doc_id += 1;
                    current_id
                });

            // Extract all column values
            let mut columns = std::collections::HashMap::new();
            for (col_idx, col_name) in column_names.iter().enumerate() {
                let array = batch.column(col_idx);
                let value = ColumnValue::from_array(array.as_ref(), row);
                columns.insert(col_name.clone(), value);
            }

            documents.push(RichDocument {
                id,
                text,
                text_column: text_column.to_string(),
                columns,
                row_index: row,
            });
        }
    }

    Ok((documents, schema))
}

/// Write rich documents to a Parquet file, preserving all columns.
///
/// This function writes all columns from the original documents,
/// maintaining HuggingFace Datasets compatibility.
///
/// # Arguments
/// * `path` - Path to the output file
/// * `docs` - Documents to write
/// * `schema` - Original schema from `read_parquet_full`
pub fn write_parquet_full<P: AsRef<Path>>(
    path: P,
    docs: &[RichDocument],
    schema: &Schema,
) -> Result<()> {
    if docs.is_empty() {
        // Write empty file with schema
        let file = File::create(path)?;
        let props = WriterProperties::builder().build();
        let writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
        writer.close()?;
        return Ok(());
    }

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let schema_ref = Arc::new(schema.clone());
    let mut writer = ArrowWriter::try_new(file, schema_ref.clone(), Some(props))?;

    // Write in batches
    const BATCH_SIZE: usize = 10000;

    for chunk in docs.chunks(BATCH_SIZE) {
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for field in schema.fields() {
            let col_name = field.name();
            let data_type = field.data_type();

            let array: ArrayRef = match data_type {
                DataType::Boolean => {
                    let values: Vec<Option<bool>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Bool(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(BooleanArray::from(values))
                }
                DataType::Int8 => {
                    let values: Vec<Option<i8>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Int8(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(Int8Array::from(values))
                }
                DataType::Int16 => {
                    let values: Vec<Option<i16>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Int16(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(Int16Array::from(values))
                }
                DataType::Int32 => {
                    let values: Vec<Option<i32>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Int32(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(Int32Array::from(values))
                }
                DataType::Int64 => {
                    let values: Vec<Option<i64>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Int64(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(Int64Array::from(values))
                }
                DataType::UInt8 => {
                    let values: Vec<Option<u8>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::UInt8(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(UInt8Array::from(values))
                }
                DataType::UInt16 => {
                    let values: Vec<Option<u16>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::UInt16(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(UInt16Array::from(values))
                }
                DataType::UInt32 => {
                    let values: Vec<Option<u32>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::UInt32(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(UInt32Array::from(values))
                }
                DataType::UInt64 => {
                    let values: Vec<Option<u64>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::UInt64(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(UInt64Array::from(values))
                }
                DataType::Float32 => {
                    let values: Vec<Option<f32>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Float32(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(Float32Array::from(values))
                }
                DataType::Float64 => {
                    let values: Vec<Option<f64>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Float64(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    Arc::new(Float64Array::from(values))
                }
                DataType::Utf8 => {
                    let values: Vec<Option<String>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::String(v)) => Some(v.clone()),
                            _ => None,
                        })
                        .collect();
                    Arc::new(StringArray::from(
                        values.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
                    ))
                }
                DataType::LargeUtf8 => {
                    let values: Vec<Option<String>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::String(v)) => Some(v.clone()),
                            _ => None,
                        })
                        .collect();
                    Arc::new(arrow::array::LargeStringArray::from(
                        values.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
                    ))
                }
                DataType::Binary => {
                    let values: Vec<Option<Vec<u8>>> = chunk
                        .iter()
                        .map(|doc| match doc.columns.get(col_name) {
                            Some(ColumnValue::Binary(v)) => Some(v.clone()),
                            _ => None,
                        })
                        .collect();
                    Arc::new(BinaryArray::from(
                        values
                            .iter()
                            .map(|v| v.as_deref())
                            .collect::<Vec<Option<&[u8]>>>(),
                    ))
                }
                _ => {
                    // For unsupported types, write as null
                    Arc::new(arrow::array::NullArray::new(chunk.len()))
                }
            };

            arrays.push(array);
        }

        let batch = RecordBatch::try_new(schema_ref.clone(), arrays)?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Detect input format from file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Jsonl,
    Parquet,
}

impl InputFormat {
    /// Detect format from file path extension.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        let ext = path.as_ref().extension()?.to_str()?;
        match ext.to_lowercase().as_str() {
            "jsonl" | "json" | "ndjson" => Some(InputFormat::Jsonl),
            "parquet" | "pq" => Some(InputFormat::Parquet),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_jsonl(content: &str) -> NamedTempFile {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), content).unwrap();
        file
    }

    #[test]
    fn test_read_jsonl_basic() {
        let content = r#"{"id": 1, "text": "Hello world"}
{"id": 2, "text": "Second document"}
{"id": 3, "text": "Third document"}"#;

        let file = create_test_jsonl(content);
        let docs = read_jsonl(file.path(), "text").unwrap();

        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].id, 1);
        assert_eq!(docs[0].text, "Hello world");
        assert_eq!(docs[1].id, 2);
        assert_eq!(docs[2].id, 3);
    }

    #[test]
    fn test_read_jsonl_custom_field() {
        let content = r#"{"id": 1, "content": "Hello world"}
{"id": 2, "content": "Second document"}"#;

        let file = create_test_jsonl(content);
        let docs = read_jsonl(file.path(), "content").unwrap();

        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].text, "Hello world");
    }

    #[test]
    fn test_read_jsonl_no_id() {
        let content = r#"{"text": "First doc"}
{"text": "Second doc"}"#;

        let file = create_test_jsonl(content);
        let docs = read_jsonl(file.path(), "text").unwrap();

        assert_eq!(docs.len(), 2);
        // Should use line number as ID
        assert_eq!(docs[0].id, 0);
        assert_eq!(docs[1].id, 1);
    }

    #[test]
    fn test_read_jsonl_missing_field() {
        let content = r#"{"id": 1, "other_field": "Hello"}"#;

        let file = create_test_jsonl(content);
        let result = read_jsonl(file.path(), "text");

        assert!(result.is_err());
        match result.unwrap_err() {
            IoError::MissingField { field, line } => {
                assert_eq!(field, "text");
                assert_eq!(line, 1);
            }
            _ => panic!("Expected MissingField error"),
        }
    }

    #[test]
    fn test_read_jsonl_parse_error() {
        let content = r#"{"id": 1, "text": "valid"}
not valid json
{"id": 3, "text": "also valid"}"#;

        let file = create_test_jsonl(content);
        let result = read_jsonl(file.path(), "text");

        assert!(result.is_err());
        match result.unwrap_err() {
            IoError::Parse { line, .. } => {
                assert_eq!(line, 2);
            }
            _ => panic!("Expected Parse error"),
        }
    }

    #[test]
    fn test_read_jsonl_empty_lines() {
        let content = r#"{"id": 1, "text": "First"}

{"id": 2, "text": "Second"}
"#;

        let file = create_test_jsonl(content);
        let docs = read_jsonl(file.path(), "text").unwrap();

        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_write_jsonl() {
        let docs = vec![
            Document::new(1, "Hello world".to_string()),
            Document::new(2, "Second document".to_string()),
        ];

        let file = NamedTempFile::new().unwrap();
        write_jsonl(file.path(), &docs).unwrap();

        // Read back and verify
        let read_docs = read_jsonl(file.path(), "text").unwrap();
        assert_eq!(read_docs.len(), 2);
        assert_eq!(read_docs[0].id, 1);
        assert_eq!(read_docs[0].text, "Hello world");
    }

    #[test]
    fn test_read_jsonl_with_original() {
        let content = r#"{"id": 1, "text": "Hello", "extra": "data"}
{"id": 2, "text": "World", "other": 123}"#;

        let file = create_test_jsonl(content);
        let docs = read_jsonl_with_original(file.path(), "text").unwrap();

        assert_eq!(docs.len(), 2);

        // Original lines should be preserved
        assert!(docs[0].1.contains("extra"));
        assert!(docs[1].1.contains("other"));
    }

    #[test]
    fn test_write_jsonl_lines() {
        let lines = vec![
            r#"{"id": 1, "text": "Hello", "extra": "data"}"#.to_string(),
            r#"{"id": 2, "text": "World", "other": 123}"#.to_string(),
        ];

        let file = NamedTempFile::new().unwrap();
        write_jsonl_lines(file.path(), &lines).unwrap();

        // Read back as raw lines
        let content = std::fs::read_to_string(file.path()).unwrap();
        let read_lines: Vec<&str> = content.lines().collect();

        assert_eq!(read_lines.len(), 2);
        assert!(read_lines[0].contains("extra"));
        assert!(read_lines[1].contains("other"));
    }

    #[test]
    fn test_streaming_reader() {
        let content = r#"{"id": 1, "text": "First"}
{"id": 2, "text": "Second"}
{"id": 3, "text": "Third"}"#;

        let file = create_test_jsonl(content);
        let reader = JsonlReader::new(file.path(), "text").unwrap();

        let docs: Vec<_> = reader.collect();
        assert_eq!(docs.len(), 3);

        for result in docs {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_document_line_number() {
        let content = r#"{"id": 1, "text": "First"}
{"id": 2, "text": "Second"}"#;

        let file = create_test_jsonl(content);
        let docs = read_jsonl(file.path(), "text").unwrap();

        assert_eq!(docs[0].line_number, Some(1));
        assert_eq!(docs[1].line_number, Some(2));
    }

    // Parquet tests

    #[test]
    fn test_parquet_roundtrip() {
        let docs = vec![
            Document::new(1, "Hello world".to_string()),
            Document::new(2, "Second document".to_string()),
            Document::new(3, "Third document".to_string()),
        ];

        let file = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        write_parquet(file.path(), &docs).unwrap();
        let read_docs = read_parquet(file.path(), "text").unwrap();

        assert_eq!(read_docs.len(), 3);
        assert_eq!(read_docs[0].id, 1);
        assert_eq!(read_docs[0].text, "Hello world");
        assert_eq!(read_docs[1].id, 2);
        assert_eq!(read_docs[1].text, "Second document");
        assert_eq!(read_docs[2].id, 3);
    }

    #[test]
    fn test_parquet_empty() {
        let docs: Vec<Document> = vec![];

        let file = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        write_parquet(file.path(), &docs).unwrap();
        let read_docs = read_parquet(file.path(), "text").unwrap();

        assert!(read_docs.is_empty());
    }

    #[test]
    fn test_parquet_column_not_found() {
        let docs = vec![Document::new(1, "Hello".to_string())];

        let file = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        write_parquet(file.path(), &docs).unwrap();
        let result = read_parquet(file.path(), "nonexistent");

        assert!(result.is_err());
        match result.unwrap_err() {
            IoError::ColumnNotFound { column } => {
                assert_eq!(column, "nonexistent");
            }
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_input_format_detection() {
        assert_eq!(
            InputFormat::from_path("data.jsonl"),
            Some(InputFormat::Jsonl)
        );
        assert_eq!(
            InputFormat::from_path("data.parquet"),
            Some(InputFormat::Parquet)
        );
        assert_eq!(
            InputFormat::from_path("data.pq"),
            Some(InputFormat::Parquet)
        );
        assert_eq!(
            InputFormat::from_path("data.json"),
            Some(InputFormat::Jsonl)
        );
        assert_eq!(
            InputFormat::from_path("data.ndjson"),
            Some(InputFormat::Jsonl)
        );
        assert_eq!(InputFormat::from_path("data.txt"), None);
        assert_eq!(InputFormat::from_path("data"), None);
    }

    // Tests for enhanced Parquet I/O with column preservation

    #[test]
    fn test_parquet_full_roundtrip() {
        use arrow::datatypes::Field;

        // Create a multi-column parquet file
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("label", DataType::Int32, true),
            Field::new("score", DataType::Float64, true),
        ]));

        let file = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        // Write test data directly
        {
            let f = File::create(file.path()).unwrap();
            let props = WriterProperties::builder().build();
            let mut writer = ArrowWriter::try_new(f, schema.clone(), Some(props)).unwrap();

            let id_array: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 3]));
            let text_array: ArrayRef = Arc::new(StringArray::from(vec!["Hello", "World", "Test"]));
            let label_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 0]));
            let score_array: ArrayRef = Arc::new(Float64Array::from(vec![0.9, 0.8, 0.95]));

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![id_array, text_array, label_array, score_array],
            )
            .unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        // Read with full column preservation
        let (docs, read_schema) = read_parquet_full(file.path(), "text").unwrap();

        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].id, 1);
        assert_eq!(docs[0].text, "Hello");
        assert_eq!(docs[1].id, 2);
        assert_eq!(docs[1].text, "World");

        // Check extra columns are preserved
        assert!(matches!(
            docs[0].columns.get("label"),
            Some(ColumnValue::Int32(0))
        ));
        assert!(matches!(
            docs[0].columns.get("score"),
            Some(ColumnValue::Float64(v)) if (*v - 0.9).abs() < 0.001
        ));

        // Filter to keep only first and third documents
        let filtered: Vec<_> = docs.into_iter().filter(|d| d.id != 2).collect();
        assert_eq!(filtered.len(), 2);

        // Write back with all columns
        let output = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();
        write_parquet_full(output.path(), &filtered, &read_schema).unwrap();

        // Verify roundtrip
        let (final_docs, _) = read_parquet_full(output.path(), "text").unwrap();
        assert_eq!(final_docs.len(), 2);
        assert_eq!(final_docs[0].text, "Hello");
        assert_eq!(final_docs[1].text, "Test");

        // Verify extra columns survived
        assert!(matches!(
            final_docs[1].columns.get("label"),
            Some(ColumnValue::Int32(0))
        ));
    }

    #[test]
    fn test_rich_document_to_document() {
        let mut columns = std::collections::HashMap::new();
        columns.insert("text".to_string(), ColumnValue::String("Test".to_string()));
        columns.insert("extra".to_string(), ColumnValue::Int32(42));

        let rich = RichDocument {
            id: 123,
            text: "Test text".to_string(),
            text_column: "text".to_string(),
            columns,
            row_index: 0,
        };

        let simple = rich.to_document();
        assert_eq!(simple.id, 123);
        assert_eq!(simple.text, "Test text");
    }

    #[test]
    fn test_parquet_full_empty() {
        use arrow::datatypes::Field;

        let schema = Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("text", DataType::Utf8, false),
        ]);

        let file = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        let docs: Vec<RichDocument> = vec![];
        write_parquet_full(file.path(), &docs, &schema).unwrap();

        let (read_docs, _) = read_parquet_full(file.path(), "text").unwrap();
        assert!(read_docs.is_empty());
    }
}
