//! Distributed deduplication with coordinator and worker architecture.
//!
//! This module provides a scalable deduplication system that can process
//! datasets across multiple workers with a central coordinator maintaining
//! the global index.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐
//! │   Worker 1  │     │   Worker 2  │
//! │ Local Index │     │ Local Index │
//! └──────┬──────┘     └──────┬──────┘
//!        │                   │
//!        └─────────┬─────────┘
//!                  │
//!           ┌──────┴──────┐
//!           │ Coordinator │
//!           │ Global Index│
//!           └─────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use mithril_dedup::distributed::{Coordinator, Worker, CoordinatorConfig, WorkerConfig};
//!
//! // Start coordinator
//! let coord_config = CoordinatorConfig::default();
//! let coordinator = Coordinator::new(coord_config);
//!
//! // Create workers
//! let worker_config = WorkerConfig::default();
//! let worker = Worker::new(worker_config);
//!
//! // Process documents
//! let results = worker.process_documents(&documents);
//!
//! // Submit to coordinator
//! coordinator.merge_worker_index(worker.index_data()?)?;
//! ```

mod coordinator;
mod messages;
mod worker;

pub use coordinator::{Coordinator, CoordinatorConfig, CoordinatorStats, WorkerInfo};
pub use messages::{
    DeduplicationResult, MergeRequest, MergeResponse, QueryRequest, QueryResponse, RegisterRequest,
    RegisterResponse, SignatureBatch, WorkerStats,
};
pub use worker::{Worker, WorkerConfig, WorkerResult};
