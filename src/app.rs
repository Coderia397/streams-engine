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
    /// Index over the title profile vectors (the quality titles), which row definitions are matched against; none until built
    pub profile_hnsw: Arc<RwLock<Option<HnswIndex>>>,
    pub seeds:      Arc<ClusterSeeds>,
    pub vault:      Arc<CentroidVault>,
    pub taxonomy:   Arc<TaxonomyStore>,
    pub mood_vault: Arc<MoodVault>,
    /// Model written title profiles and their vectors (empty when the files are missing)
    pub profiles:   Arc<crate::profiles::ProfileStore>,
    /// New, upcoming and trending titles with real release dates (fresh.json, re-read when it changes)
    pub fresh:      crate::fresh::SharedFresh,
    /// Directors and top billed cast to their titles, for searching by person
    pub people:     Arc<crate::people::PeopleIndex>,
    pub feed_cache: Cache<String, String>,
    /// Corpus mean vector, for centred cosine scoring of taxonomy rows
    pub mean:       Arc<Vec<f32>>,
    pub sessions:   crate::sessions::SessionStore,
    /// Supabase token verification; none when SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY is missing (every request anonymous)
    pub identity:   Option<Arc<crate::identity::Identity>>,
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
    // Row definitions are matched against the title profile vectors when those exist, so they are centred on the mean of the profile vectors
    // (a different text from the catalog's, with a different mean); otherwise on the catalog mean.
    let profile_dir0 = std::path::Path::new(&config.db_path).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    let profiles0 = crate::profiles::ProfileStore::load(&profile_dir0);
    if let Some(t) = Arc::get_mut(&mut taxonomy) {
        if profiles0.mean.len() == crate::profiles::DIMS { t.center_with(&profiles0.mean); } else { t.center_with(&mean); }
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
    let stamp = vectors_stamp(&config.vectors_path);
    let stamp_file = format!("{}.stamp", config.index_path);
    // HNSW_TRUST_INDEX=1: a downloaded copy has new file times, so the stamp (size and time of the vectors file) cannot match; the index is trusted as it is
    let trust = std::env::var("HNSW_TRUST_INDEX").map(|v| v == "1").unwrap_or(false);
    let stamp_ok = trust || std::fs::read_to_string(&stamp_file).map(|s| s.trim() == stamp).unwrap_or(false);
    if std::path::Path::new(&config.index_path).exists() && !stamp_ok {
        tracing::warn!("HNSW index has no matching stamp for {} — it was built from other vectors, rebuilding", config.vectors_path);
    }
    if std::path::Path::new(&config.index_path).exists() && stamp_ok {
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
        spawn_hnsw_build(hnsw_slot.clone(), vectors.clone(), config.index_path.clone(), stamp.clone());
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

    let profile_dir = std::path::Path::new(&config.db_path).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    let profiles = Arc::new(profiles0);
    info!("Title profiles: {} ({} with vectors)", profiles.profiles.len(), profiles.vectors.len() / crate::profiles::DIMS);

    // The profile vectors cover the quality titles only, so a row definition finds its titles among them instead of among 650,000 mostly obscure ones.
    let profile_hnsw: Arc<RwLock<Option<HnswIndex>>> = Arc::new(RwLock::new(None));
    if !profiles.keys.is_empty() && profiles.vectors.len() == profiles.keys.len() * crate::profiles::DIMS {
        spawn_profile_index(profile_hnsw.clone(), profiles.clone(), format!("{profile_dir}/profile_vectors_v1.bin.hnsw"));
    }

    let fresh_store = crate::fresh::FreshStore::load(&profile_dir);
    info!("Fresh titles: {} ({})", fresh_store.items.len(), fresh_store.generated);
    let fresh: crate::fresh::SharedFresh = Arc::new(arc_swap::ArcSwap::from_pointee(fresh_store));
    crate::fresh::spawn_refresh(fresh.clone(), profile_dir.clone());

    let people = Arc::new(crate::people::PeopleIndex::load(&config.db_path));
    info!("People index: {} names", people.len());

    let config = Arc::new(config.clone());
    let taxonomy_bin = config.taxonomy_bin_path.clone();
    let state = Arc::new(AppState {
        catalog,
        vectors,
        hnsw: hnsw_slot,
        profile_hnsw,
        seeds,
        vault,
        taxonomy,
        mood_vault,
        profiles,
        fresh,
        people,
        feed_cache,
        mean,
        sessions: crate::sessions::SessionStore::new(),
        identity: crate::identity::Identity::from_env(),
        config,
    });
    if !state.taxonomy.entries.is_empty() {
        spawn_support(state.clone(), taxonomy_bin);
    }
    Ok(state)
}

/// Background task: loads the index over the profile vectors from disk, or builds it (about a minute) and saves it for the next start.
fn spawn_profile_index(slot: Arc<RwLock<Option<HnswIndex>>>, profiles: Arc<crate::profiles::ProfileStore>, path: String) {
    tokio::spawn(async move {
        let built = tokio::task::spawn_blocking(move || -> anyhow::Result<HnswIndex> {
            if std::path::Path::new(&path).exists() {
                if let Ok(idx) = HnswIndex::load(&path) {
                    if idx.len() == profiles.keys.len() {
                        return Ok(idx);
                    }
                }
            }
            let idx = HnswIndex::build_from_f16(&profiles.keys, &profiles.vectors)?;
            let _ = idx.save(&path);
            Ok(idx)
        })
        .await;
        match built {
            Ok(Ok(idx)) => *slot.write().await = Some(idx),
            Ok(Err(e)) => tracing::warn!("profile index not built: {e}"),
            Err(e) => tracing::warn!("profile index task failed: {e}"),
        }
    });
}

/// Background task: once the HNSW index is ready, load or compute how many quality
/// titles each taxonomy entry can fill, so pages only consider entries that work.
fn spawn_support(state: Arc<AppState>, taxonomy_bin: String) {
    tokio::spawn(async move {
        let wants_profile_index = !state.profiles.keys.is_empty();
        loop {
            let ready = if wants_profile_index { state.profile_hnsw.read().await.is_some() } else { ENGINE_READY.load(Ordering::Relaxed) };
            if ready { break; }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        let guard = if wants_profile_index { state.profile_hnsw.clone().read_owned().await } else { state.hnsw.clone().read_owned().await };
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
        let base = env("CATALOG_DIR", "/Users/ibrahim/streams-catalog");
        // The rebuilt catalog and vectors are picked up automatically when present.
        let pick = |v2: &str, v1: &str| {
            if std::path::Path::new(&format!("{base}/{v2}")).exists() { format!("{base}/{v2}") } else { format!("{base}/{v1}") }
        };
        let vectors_path = env("VECTORS_BIN", &pick("vectors_v2.bin", "vectors.bin"));
        // The index and the cluster seeds are only valid for the vectors file they were built from,
        // so their default names derive from it. A new vectors file never reuses an old index.
        Self {
            db_path:            env("CATALOG_DB",    &pick("catalog_v2.db", "catalog.db")),
            taxonomy_bin_path:  env("TAXONOMY_BIN",  &format!("{base}/taxonomy_embeddings.bin")),
            taxonomy_json_path: env("TAXONOMY_JSON", &format!("{base}/taxonomy_embeddings.json")),
            seeds_path:         env("SEEDS_BIN",     &format!("{vectors_path}.seeds")),
            // seeds written before the type aware keys; valid only for the original vectors.bin
            seeds_legacy_path:  env("SEEDS_BIN_LEGACY", &if vectors_path.ends_with("/vectors.bin") { format!("{base}/cluster_seeds.bin") } else { String::new() }),
            mood_bin_path:      env("MOOD_BIN",      &format!("{base}/mood_embeddings.bin")),
            index_path:         env("HNSW_INDEX",    &format!("{vectors_path}.hnsw")),
            port:               env("PORT", "8088").parse().unwrap_or(8088),
            vectors_path,
        }
    }
}

/// Identity of a vectors file (size and modification time). Stored next to the HNSW index so an index
/// built from other vectors is never loaded.
pub fn vectors_stamp(path: &str) -> String {
    match std::fs::metadata(path) {
        Ok(m) => {
            let secs = m.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs()).unwrap_or(0);
            format!("{}:{}", m.len(), secs)
        }
        Err(_) => String::new(),
    }
}
