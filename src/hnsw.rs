use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::vectors::{VectorStore, DIMS};
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;

/// HNSW index parameters for 2560d cosine search over the full catalog.
/// connectivity (M): 16 — edges per node, affects recall vs memory
/// expansion_add: 128 — ef_construction — quality of graph during build
/// expansion_search: 64 — ef_search default, can be overridden per query
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 128;
const HNSW_EF_SEARCH: usize = 64;

/// Wrapper around usearch::Index with persistence and ready-state tracking.
pub struct HnswIndex {
    inner: Index,
}

// usearch::Index is designed for concurrent use — safe to share across threads.
unsafe impl Send for HnswIndex {}
unsafe impl Sync for HnswIndex {}

impl HnswIndex {
    fn make_options() -> IndexOptions {
        IndexOptions {
            dimensions: DIMS,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F16, // store internally as f16 (512MB)
            connectivity: HNSW_M,
            expansion_add: HNSW_EF_CONSTRUCTION,
            expansion_search: HNSW_EF_SEARCH,
            ..Default::default()
        }
    }

    /// Load a pre-built index from disk.
    /// usearch 2.x: load() is an INSTANCE method — create empty Index first,
    /// then call load() on it to populate from file.
    pub fn load(path: &str) -> Result<Self> {
        let inner = Index::new(&Self::make_options())?;
        inner.load(path)?;
        let n = inner.size();
        info!("HNSW index loaded from {path} ({n} vectors)");
        Ok(Self { inner })
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    /// Build a new index over every vector in the store, in parallel.
    /// Zero placeholder rows (failed embeddings) are skipped. Call on a blocking thread.
    pub fn build_from_store(store: &VectorStore) -> Result<Self> {
        let inner = Index::new(&Self::make_options())?;
        let threads = rayon::current_num_threads();
        inner.reserve_capacity_and_threads(store.n_vecs, threads)?;

        let done = AtomicUsize::new(0);
        let step = 50_000;
        (0..store.n_vecs).into_par_iter().try_for_each(|i| -> Result<()> {
            if !store.is_zero_row(i) {
                inner.add(store.ids[i] as u64, &store.get_f32(i))?;
            }
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % step == 0 {
                info!("HNSW build progress: {d}/{}", store.n_vecs);
            }
            Ok(())
        })?;

        info!("HNSW index built: {} vectors", inner.size());
        Ok(Self { inner })
    }

    /// Persist the index atomically (temp file then rename) so a crash mid-save
    /// never leaves a truncated index for the next boot.
    pub fn save(&self, path: &str) -> Result<()> {
        let tmp = format!("{path}.tmp");
        self.inner.save(&tmp)?;
        std::fs::rename(&tmp, path)?;
        info!("HNSW index saved to {path}");
        Ok(())
    }

    /// Search for top-k approximate nearest neighbours.
    /// Returns (catalog_id, distance) pairs — distance is 1 - cosine_sim for MetricKind::Cos.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let matches = self.inner.search(query, k)?;
        Ok(matches
            .keys
            .into_iter()
            .zip(matches.distances.into_iter())
            .map(|(key, dist)| (key as u32, 1.0 - dist)) // convert distance → similarity
            .collect())
    }
}

/// Engine readiness flag. Set to true once HNSW index is built/loaded.
/// While false, routes fall back to brute-force rayon search.
pub static ENGINE_READY: AtomicBool = AtomicBool::new(false);

pub fn is_ready() -> bool {
    ENGINE_READY.load(Ordering::Relaxed)
}

/// Spawn background HNSW build task. While building, the engine serves in
/// degraded brute-force mode. On completion, atomically swaps the index and
/// sets ENGINE_READY = true.
pub fn spawn_hnsw_build(
    hnsw_slot: Arc<tokio::sync::RwLock<Option<HnswIndex>>>,
    store: Arc<VectorStore>,
    index_path: String,
) {
    tokio::spawn(async move {
        warn!("HNSW index missing or stale — building over {} vectors in background. Serving brute force.", store.n_vecs);
        match tokio::task::spawn_blocking(move || HnswIndex::build_from_store(&store)).await {
            Ok(Ok(idx)) => {
                if let Err(e) = idx.save(&index_path) {
                    warn!("Failed to persist HNSW index: {e}");
                }
                let mut slot = hnsw_slot.write().await;
                *slot = Some(idx);
                ENGINE_READY.store(true, Ordering::Relaxed);
                info!("HNSW index ready — switching from brute-force to ANN mode.");
            }
            Ok(Err(e)) => warn!("HNSW build failed: {e}"),
            Err(e)     => warn!("HNSW build task panicked: {e}"),
        }
    });
}
