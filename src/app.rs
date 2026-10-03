use crate::catalog::{make_shared_catalog, spawn_catalog_refresh, SharedCatalog};
use crate::centroid::CentroidVault;
use crate::cluster::ClusterSeeds;
use crate::hnsw::{spawn_hnsw_build, HnswIndex, ENGINE_READY};
use crate::mood::MoodVault;
use crate::taxonomy::TaxonomyStore;
use crate::vectors::VectorStore;
use anyhow::Result;
use moka::future::Cache;
use std::sync::{atomic::Ordering, Arc};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::info;

/// Central application state shared across all axum route handlers.
pub struct AppState {
    pub catalog:    SharedCatalog,
    pub vectors:    Arc<VectorStore>,
    pub hnsw:       Arc<RwLock<Option<HnswIndex>>>,
    pub seeds:      Arc<ClusterSeeds>,
    pub vault:      Arc<CentroidVault>,
    pub taxonomy:   Arc<TaxonomyStore>,
    pub mood_vault: Arc<MoodVault>,
    pub feed_cache: Cache<String, String>,
    /// Corpus mean vector, for centred cosine scoring of taxonomy rows
    pub mean:       Arc<Vec<f32>>,
    pub sessions:   crate::sessions::SessionStore,
    /// Kept for FTS search route
    pub config:     Arc<Config>,
}

/// Load all assets and build AppState. HNSW may still build in background.
pub async fn build_app_state(config: &Config) -> Result<Arc<AppState>> {
    // 1. Catalog (SQLite → ArcSwap<HashMap>)
    let catalog = make_shared_catalog(&config.db_path)?;
    spawn_catalog_refresh(catalog.clone(), config.db_path.clone());

    // 2. Vectors — IDs derived from catalog.db (no separate IDs file)
    let vectors = VectorStore::load(&config.vectors_path, &config.db_path)?;

    // 3. Taxonomy embeddings (optional — engine degrades gracefully without them)
    let taxonomy = TaxonomyStore::load(&config.taxonomy_bin_path, &config.taxonomy_json_path)
        .unwrap_or_else(|e| {
            tracing::warn!("TaxonomyStore not loaded ({e}) — using mood+genre fallback titles");
            TaxonomyStore::from_raw(vec![])
        });

    // 3b. Corpus mean and centred taxonomy vectors
    let mean = Arc::new(vectors.compute_mean());
    let mut taxonomy = taxonomy;
    if let Some(t) = Arc::get_mut(&mut taxonomy) {
        t.center_with(&mean);
    }

    // 4. Cluster seeds
    let seeds = ClusterSeeds::load(&config.seeds_path, &config.seeds_legacy_path, &catalog.load())?;

    // 5. CentroidVault (genre centroids from top-quality catalog items)
    let catalog_guard = catalog.load();
    let vault = CentroidVault::compute(&catalog_guard, &vectors)?;
    drop(catalog_guard);

    // 6. MoodVault — from mood_embeddings.bin (Colab notebook Part A); empty when absent
    let mood_vault = Arc::new(MoodVault::new(
        crate::mood::load_mood_embeddings(&config.mood_bin_path).unwrap_or_else(|e| {
            tracing::warn!("MoodVault not loaded ({e}) — row titles use era and genre only");
            vec![]
        }),
    ));

    // 7. HNSW index
    let hnsw_slot: Arc<RwLock<Option<HnswIndex>>> = Arc::new(RwLock::new(None));
    if std::path::Path::new(&config.index_path).exists() {
        match HnswIndex::load(&config.index_path) {
            Ok(idx) if idx.len() * 100 >= vectors.n_vecs * 95 => {
                *hnsw_slot.write().await = Some(idx);
                ENGINE_READY.store(true, Ordering::Relaxed);
                info!("HNSW index loaded from disk");
            }
            Ok(idx) => tracing::warn!(
                "HNSW index is stale ({} of {} vectors) — rebuilding in background",
                idx.len(), vectors.n_vecs
            ),
            Err(e) => tracing::warn!("HNSW load failed ({e}) — rebuilding in background"),
        }
    }

    if !ENGINE_READY.load(Ordering::Relaxed) {
        spawn_hnsw_build(hnsw_slot.clone(), vectors.clone(), config.index_path.clone());
    }

    // 8. Feed cache (W-TinyLFU, 30s TTL)
    let feed_cache: Cache<String, String> = Cache::builder()
        .max_capacity(1024)
        .time_to_live(Duration::from_secs(30))
        .build();

    info!(
        "AppState ready — engine mode: {}",
        if ENGINE_READY.load(Ordering::Relaxed) { "HNSW" } else { "brute-force" }
    );

    let config = Arc::new(config.clone());
    let taxonomy_bin = config.taxonomy_bin_path.clone();
    let state = Arc::new(AppState {
        catalog,
        vectors,
        hnsw: hnsw_slot,
        seeds,
        vault,
        taxonomy,
        mood_vault,
        feed_cache,
        mean,
        sessions: crate::sessions::SessionStore::new(),
        config,
    });
    if !state.taxonomy.entries.is_empty() {
        spawn_support(state.clone(), taxonomy_bin);
    }
    Ok(state)
}

/// Background task: once the HNSW index is ready, load or compute how many quality
/// titles each taxonomy entry can fill, so pages only consider entries that work.
fn spawn_support(state: Arc<AppState>, taxonomy_bin: String) {
    tokio::spawn(async move {
        while !ENGINE_READY.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        let guard = state.hnsw.clone().read_owned().await;
        let st = state.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let Some(idx) = guard.as_ref() else { return };
            if crate::support::load_cache(&st.taxonomy, &taxonomy_bin, idx.len()) {
                return;
            }
            let catalog = st.catalog.load();
            crate::support::compute(&st.taxonomy, &catalog, idx, &taxonomy_bin);
        })
        .await;
    });
}

/// Server configuration — from env vars.
#[derive(Debug, Clone)]
pub struct Config {
    pub db_path:            String,
    pub vectors_path:       String,
    pub taxonomy_bin_path:  String,
    pub taxonomy_json_path: String,
    pub seeds_path:         String,
    pub seeds_legacy_path:  String,
    pub mood_bin_path:      String,
    pub index_path:         String,
    pub port:               u16,
}

impl Config {
    pub fn from_env() -> Self {
        fn env(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }
        let base = env("CATALOG_DIR", "/Users/ibrahim/pstream-catalog");
        Self {
            db_path:            env("CATALOG_DB",    &format!("{base}/catalog.db")),
            vectors_path:       env("VECTORS_BIN",   &format!("{base}/vectors.bin")),
            taxonomy_bin_path:  env("TAXONOMY_BIN",  &format!("{base}/taxonomy_embeddings.bin")),
            taxonomy_json_path: env("TAXONOMY_JSON", &format!("{base}/taxonomy_embeddings.json")),
            seeds_path:         env("SEEDS_BIN",     &format!("{base}/cluster_seeds_v2.bin")),
            seeds_legacy_path:  env("SEEDS_BIN_LEGACY", &format!("{base}/cluster_seeds.bin")),
            mood_bin_path:      env("MOOD_BIN",      &format!("{base}/mood_embeddings.bin")),
            // v2: index keys are type aware item keys; the old hnsw_index.bin holds raw ids and is left untouched
            index_path:         env("HNSW_INDEX",    &format!("{base}/hnsw_index_v2.bin")),
            port:               env("PORT", "8088").parse().unwrap_or(8088),
        }
    }
}
