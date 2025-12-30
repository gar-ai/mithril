//! Union-Find data structure for clustering duplicate documents.
//!
//! This module provides an efficient disjoint-set data structure with
//! path compression and union-by-rank optimizations.

use std::collections::HashMap;

/// Union-Find (Disjoint Set Union) data structure.
///
/// Used to group documents into duplicate clusters efficiently.
/// Supports near-constant time operations via path compression and union-by-rank.
pub struct UnionFind {
    /// Parent pointers. parent[i] = j means i's parent is j.
    parent: Vec<usize>,
    /// Rank (approximate tree depth) for union-by-rank.
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new Union-Find structure with n elements.
    ///
    /// Initially, each element is in its own singleton set.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the root (representative) of the set containing x.
    ///
    /// Uses path compression: all nodes on the path to root
    /// are updated to point directly to the root.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing x and y.
    ///
    /// Uses union-by-rank: the shorter tree is attached
    /// under the root of the taller tree.
    ///
    /// Returns true if x and y were in different sets (and are now merged).
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);

        if rx == ry {
            return false; // Already in same set
        }

        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }

        true
    }

    /// Check if x and y are in the same set.
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get all clusters as a map from root -> members.
    ///
    /// Each cluster is represented by its root element.
    #[must_use]
    pub fn clusters(&mut self) -> HashMap<usize, Vec<usize>> {
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let root = self.find(i);
            clusters.entry(root).or_default().push(i);
        }
        clusters
    }

    /// Get clusters with more than one element (actual duplicate groups).
    #[must_use]
    pub fn duplicate_clusters(&mut self) -> HashMap<usize, Vec<usize>> {
        self.clusters()
            .into_iter()
            .filter(|(_, members)| members.len() > 1)
            .collect()
    }

    /// Get the number of distinct sets.
    pub fn num_sets(&mut self) -> usize {
        let mut roots = std::collections::HashSet::new();
        for i in 0..self.parent.len() {
            roots.insert(self.find(i));
        }
        roots.len()
    }

    /// Get the size of the set containing x.
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        let mut count = 0;
        for i in 0..self.parent.len() {
            if self.find(i) == root {
                count += 1;
            }
        }
        count
    }

    /// Get the total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Check if the structure is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_union_find() {
        let uf = UnionFind::new(5);
        assert_eq!(uf.len(), 5);
        assert!(!uf.is_empty());
    }

    #[test]
    fn test_find_initial() {
        let mut uf = UnionFind::new(5);
        // Initially each element is its own root
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }
    }

    #[test]
    fn test_union_basic() {
        let mut uf = UnionFind::new(5);

        assert!(uf.union(0, 1)); // First union returns true
        assert!(uf.connected(0, 1));

        assert!(!uf.union(0, 1)); // Same set returns false
    }

    #[test]
    fn test_union_chain() {
        let mut uf = UnionFind::new(5);

        uf.union(0, 1);
        uf.union(1, 2);

        // 0, 1, 2 should all be connected
        assert!(uf.connected(0, 1));
        assert!(uf.connected(1, 2));
        assert!(uf.connected(0, 2));

        // 3, 4 should not be connected to the group
        assert!(!uf.connected(0, 3));
        assert!(!uf.connected(0, 4));
    }

    #[test]
    fn test_num_sets() {
        let mut uf = UnionFind::new(5);

        assert_eq!(uf.num_sets(), 5); // All singletons

        uf.union(0, 1);
        assert_eq!(uf.num_sets(), 4);

        uf.union(2, 3);
        assert_eq!(uf.num_sets(), 3);

        uf.union(0, 2);
        assert_eq!(uf.num_sets(), 2);
    }

    #[test]
    fn test_clusters() {
        let mut uf = UnionFind::new(6);

        uf.union(0, 1);
        uf.union(0, 2);
        uf.union(3, 4);

        let clusters = uf.clusters();

        // Should have 3 clusters: {0,1,2}, {3,4}, {5}
        assert_eq!(clusters.len(), 3);

        // Find the cluster containing 0
        let root_0 = uf.find(0);
        let cluster_0 = clusters.get(&root_0).unwrap();
        assert_eq!(cluster_0.len(), 3);
        assert!(cluster_0.contains(&0));
        assert!(cluster_0.contains(&1));
        assert!(cluster_0.contains(&2));
    }

    #[test]
    fn test_duplicate_clusters() {
        let mut uf = UnionFind::new(6);

        uf.union(0, 1);
        uf.union(0, 2);
        uf.union(3, 4);
        // 5 remains singleton

        let dup_clusters = uf.duplicate_clusters();

        // Should only have 2 clusters (the singletons are excluded)
        assert_eq!(dup_clusters.len(), 2);

        for (_, members) in &dup_clusters {
            assert!(members.len() > 1);
        }
    }

    #[test]
    fn test_set_size() {
        let mut uf = UnionFind::new(5);

        assert_eq!(uf.set_size(0), 1);

        uf.union(0, 1);
        assert_eq!(uf.set_size(0), 2);
        assert_eq!(uf.set_size(1), 2);

        uf.union(0, 2);
        assert_eq!(uf.set_size(0), 3);
        assert_eq!(uf.set_size(1), 3);
        assert_eq!(uf.set_size(2), 3);
    }

    #[test]
    fn test_path_compression() {
        let mut uf = UnionFind::new(10);

        // Create a long chain: 0 -> 1 -> 2 -> ... -> 9
        for i in 0..9 {
            uf.union(i, i + 1);
        }

        // Find on any element should compress the path
        let root = uf.find(0);

        // After compression, all elements should point directly to root
        for i in 0..10 {
            assert_eq!(uf.parent[i], root);
        }
    }

    #[test]
    fn test_empty_union_find() {
        let uf = UnionFind::new(0);
        assert!(uf.is_empty());
        assert_eq!(uf.len(), 0);
    }

    #[test]
    fn test_large_union_find() {
        let n = 10000;
        let mut uf = UnionFind::new(n);

        // Union all even numbers together
        for i in (0..n).step_by(2) {
            if i + 2 < n {
                uf.union(i, i + 2);
            }
        }

        // Union all odd numbers together
        for i in (1..n).step_by(2) {
            if i + 2 < n {
                uf.union(i, i + 2);
            }
        }

        // Should have exactly 2 sets
        assert_eq!(uf.num_sets(), 2);

        // All even numbers connected
        assert!(uf.connected(0, 2));
        assert!(uf.connected(0, 100));

        // All odd numbers connected
        assert!(uf.connected(1, 3));
        assert!(uf.connected(1, 101));

        // Even and odd not connected
        assert!(!uf.connected(0, 1));
    }
}
