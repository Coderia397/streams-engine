use crate::catalog::{make_shared_catalog, spawn_catalog_refresh, SharedCatalog};
use crate::centroid::CentroidVault;
use crate::cluster::ClusterSeeds;
use crate::hnsw::{spawn_hnsw_build, HnswIndex, ENGINE_READY};
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
    /// World, atom and theme group of every title (single database)
    pub groups:     Arc<crate::groups::GroupMap>,
    /// Names of the groups at every level of the hierarchy
    pub group_names: Arc<crate::groups::GroupNames>,
    /// Vibe group members, moods and seasonal sub groups
    pub vibe:       Arc<crate::vibe::VibeIndex>,
    pub vault:      Arc<CentroidVault>,
    pub taxonomy:   Arc<TaxonomyStore>,
    /// The row vocabulary of the Kids pages (`taxonomy_kids.bin` and `.json` next to the main one); empty when the files are not there
    pub taxonomy_kids: Arc<TaxonomyStore>,
    /// Model written title profiles and their vectors (empty when the files are missing)
    pub profiles:   Arc<crate::profiles::ProfileStore>,
    /// Names of the topic rows (`topics_meta.json`), empty when the file is missing
    pub topics:     Arc<crate::topics::TopicStore>,
    /// New, upcoming and trending titles with real release dates (fresh.json, re-read when it changes)
    pub fresh:      crate::fresh::SharedFresh,
    /// Directors and top billed cast to their titles, for searching by person
    pub people:     Arc<crate::people::PeopleIndex>,
    pub feed_cache: Cache<String, String>,
    /// The full rows of a page (every title of every row) under the same key as `feed_cache`: the feed sends the first titles of each row, `/api/feed/row` the rest
    pub row_cache: Cache<String, Arc<Vec<serde_json::Value>>>,
    /// Corpus mean vector, for centred cosine scoring of taxonomy rows
    pub mean:       Arc<Vec<f32>>,
    pub sessions:   crate::sessions::SessionStore,
    /// Requests the pages make rows from when `STREAMS_REQUEST_ROWS=1` (see requests/)
    pub requests:   Arc<crate::requests::RequestStore>,
    /// Supabase token verification; none when SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY is missing (every request anonymous)
    pub identity:   Option<Arc<crate::identity::Identity>>,
    /// Kept for FTS search route
    pub config:     Arc<Config>,
}

/// The result of a startup loader thread (a panic there is a bug and stops the start).
fn joined<T>(r: std::thread::Result<T>) -> T {
    r.expect("startup loader panicked")
}

/// Opens the HNSW index file when it exists and matches the vectors file (stamp, or HNSW_TRUST_INDEX=1 for a downloaded copy whose file times cannot match). None otherwise.
fn open_index_file(config: &Config, stamp: &str) -> Option<HnswIndex> {
    if !std::path::Path::new(&config.index_path).exists() {
        return None;
    }
    let stamp_file = format!("{}.stamp", config.index_path);
    let trust = std::env::var("HNSW_TRUST_INDEX").map(|v| v == "1").unwrap_or(false);
    let stamp_ok = trust || std::fs::read_to_string(&stamp_file).map(|s| s.trim() == stamp).unwrap_or(false);
    if !stamp_ok {
        tracing::warn!("HNSW index has no matching stamp for {} — it was built from other vectors, rebuilding", config.vectors_path);
        return None;
    }
    HnswIndex::load(&config.index_path)
        .map_err(|e| tracing::warn!("HNSW load failed ({e}) — rebuilding in background"))
        .ok()
}

/// Load all assets and build AppState. HNSW may still build in background.
/// The loads that do not depend on each other run in parallel threads (first the files and tables, then what needs the catalog or the vectors).
pub async fn build_app_state(config: &Config) -> Result<Arc<AppState>> {
    let db = config.db_path.as_str();
    crate::events::load(db);
    crate::page::rhythm::load(db);
    let profile_dir = std::path::Path::new(db).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    let stamp = vectors_stamp(&config.vectors_path);

    // Stage 1: independent loads. The catalog is the SQLite table in an ArcSwap, the vectors map the f16 file; the rest read small files and tables.
    let (catalog, vectors, taxonomy, taxonomy_kids, profiles, people, groups_map, group_names, fresh_store, index) = std::thread::scope(|s| {
        let h_catalog = s.spawn(|| make_shared_catalog(db));
        let h_vectors = s.spawn(|| VectorStore::load(&config.vectors_path));
        let h_taxonomy = s.spawn(|| TaxonomyStore::load(&config.taxonomy_bin_path, &config.taxonomy_json_path));
        let h_taxonomy_kids = s.spawn(|| {
            let dir = std::path::Path::new(&config.taxonomy_bin_path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
            TaxonomyStore::load(&dir.join("taxonomy_kids.bin").to_string_lossy(), &dir.join("taxonomy_kids.json").to_string_lossy())
        });
        let h_profiles = s.spawn(|| crate::profiles::ProfileStore::load(&profile_dir));
        let h_people = s.spawn(|| crate::people::PeopleIndex::load(db));
        let h_groups = s.spawn(|| crate::groups::load(db));
        let h_names = s.spawn(|| crate::groups::load_names(db));
        let h_fresh = s.spawn(|| crate::fresh::FreshStore::load(&profile_dir));
        let h_index = s.spawn(|| open_index_file(config, &stamp));
        (joined(h_catalog.join()), joined(h_vectors.join()), joined(h_taxonomy.join()), joined(h_taxonomy_kids.join()), joined(h_profiles.join()), joined(h_people.join()), joined(h_groups.join()), joined(h_names.join()), joined(h_fresh.join()), joined(h_index.join()))
    });
    let catalog = catalog?;
    let vectors = vectors?;
    spawn_catalog_refresh(catalog.clone(), config.db_path.clone());
    let mut taxonomy = taxonomy.unwrap_or_else(|e| {
        tracing::warn!("TaxonomyStore not loaded ({e}) — using mood+genre fallback titles");
        TaxonomyStore::from_raw(vec![])
    });

    let mut taxonomy_kids = taxonomy_kids.unwrap_or_else(|_| TaxonomyStore::from_raw(vec![]));
    // Stage 2: what needs the catalog or the vectors. The corpus mean centres the taxonomy vectors (on the title profile mean when those vectors exist, a different text with a different mean).
    let catalog_now = catalog.load();
    let (mean, seeds, vault) = std::thread::scope(|s| {
        let h_mean = s.spawn(|| Arc::new(vectors.compute_mean()));
        let h_seeds = s.spawn(|| ClusterSeeds::load(&config.seeds_path));
        let h_vault = s.spawn(|| CentroidVault::compute(&catalog_now, &vectors));
        (joined(h_mean.join()), joined(h_seeds.join()), joined(h_vault.join()))
    });
    drop(catalog_now);
    let (seeds, vault) = (seeds?, vault?);
    // an index with fewer than 95 percent of the vectors is stale
    let index = index.filter(|idx| {
        let ok = idx.len() * 100 >= vectors.n_vecs * 95;
        if !ok {
            tracing::warn!("HNSW index is stale ({} of {} vectors) — rebuilding in background", idx.len(), vectors.n_vecs);
        }
        ok
    });
    if let Some(t) = Arc::get_mut(&mut taxonomy) {
        t.center_with(&mean);
    }
    if let Some(t) = Arc::get_mut(&mut taxonomy_kids) {
        t.center_with(&mean);
    }
    tracing::info!("Kids row vocabulary: {} entries", taxonomy_kids.entries.len());
    // the rows the engine may make from requests (note 96); empty without requests_v1.json
    let requests = Arc::new(crate::requests::RequestStore::load(&config.db_path, &catalog.load(), &vectors, &mean));

    // HNSW index: loaded from disk, or built in the background (brute force serves until it is ready)
    let hnsw_slot: Arc<RwLock<Option<HnswIndex>>> = Arc::new(RwLock::new(None));
    if let Some(idx) = index {
        *hnsw_slot.write().await = Some(idx);
        ENGINE_READY.store(true, Ordering::Relaxed);
        info!("HNSW index loaded from disk");
    } else {
        spawn_hnsw_build(hnsw_slot.clone(), vectors.clone(), config.index_path.clone(), stamp.clone());
    }

    // 8. Feed cache (W-TinyLFU, 30s TTL)
    let feed_cache: Cache<String, String> = Cache::builder()
        .max_capacity(1024)
        .time_to_live(Duration::from_secs(30))
        .build();
    // the full rows live longer than the page text, so paging inside a row keeps working while a viewer scrolls
    let row_cache: Cache<String, Arc<Vec<serde_json::Value>>> = Cache::builder()
        .max_capacity(512)
        .time_to_live(Duration::from_secs(600))
        .build();

    info!(
        "AppState ready — engine mode: {}",
        if ENGINE_READY.load(Ordering::Relaxed) { "HNSW" } else { "brute-force" }
    );

    let profiles = Arc::new(profiles);
    let topics = Arc::new(crate::topics::TopicStore::load(&profile_dir));
    info!("Topic rows: {} topics", topics.topics.len());
    info!("Title profiles: {}", profiles.profiles.len());

    info!("Fresh titles: {} ({})", fresh_store.items.len(), fresh_store.generated);
    let fresh: crate::fresh::SharedFresh = Arc::new(arc_swap::ArcSwap::from_pointee(fresh_store));
    crate::fresh::spawn_refresh(fresh.clone(), profile_dir.clone());

    let people = Arc::new(people);
    info!("People index: {} names", people.len());

    let config = Arc::new(config.clone());
    let taxonomy_bin = config.taxonomy_bin_path.clone();
    let state = Arc::new(AppState {
        catalog,
        vectors,
        hnsw: hnsw_slot,
        seeds,
        groups: Arc::new(groups_map.clone()),
        group_names: Arc::new(group_names),
        vibe: Arc::new(crate::vibe::load(&config.db_path, crate::groups::vibe_members(&groups_map))),
        vault,
        taxonomy,
        taxonomy_kids,
        profiles,
        topics,
        fresh,
        people,
        feed_cache,
        row_cache,
        mean,
        sessions: crate::sessions::SessionStore::new(),
        requests,
        identity: crate::identity::Identity::from_env(),
        config,
    });
    // the catalog and the vectors must describe the same titles; say so when the export of the pipeline was not run after a catalog change
    let coverage = crate::status::Coverage::of(&state.catalog.load(), &state.vectors);
    info!("Data: {} catalog titles, {} vectors, {} titles without a vector, {} vectors outside the catalog", coverage.titles, coverage.vectors, coverage.titles_without_vector(), coverage.vectors_outside_catalog());
    if let Some(advice) = coverage.advice() {
        tracing::warn!("{advice}");
    }
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
    pub index_path:         String,
    pub port:               u16,
}

impl Config {
    /// Paths come from the environment; the defaults sit in the catalog folder (`CATALOG_DIR`, default `$HOME/streams-catalog`).
    pub fn from_env() -> Self {
        fn env(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let base = env("CATALOG_DIR", &format!("{home}/streams-catalog"));
        let vectors_path = env("VECTORS_BIN", &format!("{base}/vectors_v3.bin"));
        // The index and the cluster seeds are only valid for the vectors file they were built from,
        // so their default names derive from it. A new vectors file never reuses an old index.
        Self {
            db_path:            env("CATALOG_DB",    &format!("{base}/streams_v3.db")),
            taxonomy_bin_path:  env("TAXONOMY_BIN",  &format!("{base}/taxonomy_embeddings.bin")),
            taxonomy_json_path: env("TAXONOMY_JSON", &format!("{base}/taxonomy_embeddings.json")),
            seeds_path:         env("SEEDS_BIN",     &format!("{vectors_path}.seeds")),
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
