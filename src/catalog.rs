use anyhow::Result;
use arc_swap::ArcSwap;
use rusqlite::Connection;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// A single item in the streaming catalog.
#[derive(Debug, Clone)]
pub struct CatalogItem {
    /// Internal key, unique across movies and TV (see `item_key`). Use `tmdb_id` for anything shown to clients.
    pub id: u32,
    /// TMDB id. Movies and TV series share the same id space, so it is only unique together with `media_type`.
    pub tmdb_id: u32,
    pub media_type: String,
    pub title: String,
    pub release_year: Option<i32>,
    /// TMDB genre names, e.g. ["Action", "Science Fiction"]
    pub genres: Vec<String>,
    pub overview: String,
    pub tagline: Option<String>,
    pub director: Option<String>,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
    pub vote_average: f32,
    pub popularity: f32,
    pub is_anime: bool,
    /// Present only in a catalog built from the TMDB pull (catalog_v2); defaults otherwise.
    pub vote_count: u32,
    pub adult: bool,
    pub original_language: String,
    /// TMDB tv type: Scripted, Reality, Talk Show, News, Documentary, Miniseries, Video
    pub tv_type: Option<String>,
}

/// TMDB numbers movies and TV series separately, so the same id exists in both. The engine keys every
/// title by `item_key`: movies keep their TMDB id, TV series get the top bit set.
pub const TV_BIT: u32 = 1 << 31;

#[inline]
pub fn item_key(media_type: &str, tmdb_id: u32) -> u32 {
    if media_type == "tv" { tmdb_id | TV_BIT } else { tmdb_id }
}

/// Resolve a client supplied (id, media_type) to an internal key that exists in the catalog.
/// Without a media type the movie is tried first, then the series.
pub fn resolve_key(catalog: &CatalogMap, tmdb_id: u32, media_type: Option<&str>) -> Option<u32> {
    match media_type {
        Some(mt) => Some(item_key(mt, tmdb_id)).filter(|k| catalog.contains_key(k)),
        None => [tmdb_id, tmdb_id | TV_BIT].into_iter().find(|k| catalog.contains_key(k)),
    }
}

impl CatalogItem {
    /// Number of Unicode scalar values in the overview (anchor gate check).
    pub fn overview_chars(&self) -> usize {
        self.overview.chars().count()
    }

    /// Primary genre (first in list, empty string if none).
    pub fn primary_genre(&self) -> &str {
        self.genres.first().map(|s| s.as_str()).unwrap_or("")
    }
}

pub type CatalogMap = HashMap<u32, CatalogItem>;
pub type SharedCatalog = Arc<ArcSwap<CatalogMap>>;

/// Load the entire catalog from SQLite into a HashMap keyed by `item_key` (type aware).
/// This is the ONLY place SQLite is accessed — never on the request hot path.
pub fn load_catalog(db_path: &str) -> Result<CatalogMap> {
    let conn = Connection::open(db_path)?;

    // catalog_v2 has `tmdb_id` (the primary key `id` there is the internal key) plus optional extra columns.
    let cols: std::collections::HashSet<String> = conn
        .prepare("PRAGMA table_info(catalog)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .collect();
    let id_col = if cols.contains("tmdb_id") { "tmdb_id" } else { "id" };
    let opt = |name: &str, default: &str| if cols.contains(name) { name.to_string() } else { format!("{default} AS {name}") };

    let sql = format!(
        "SELECT {id_col}, media_type, title, release_year, genres,
                overview, poster_path, backdrop_path,
                vote_average, popularity, is_anime, tagline, director,
                {vc}, {ad}, {ol}, {tt}
         FROM catalog",
        vc = opt("vote_count", "0"),
        ad = opt("adult", "0"),
        ol = opt("original_language", "''"),
        tt = opt("tv_type", "NULL"),
    );
    let mut stmt = conn.prepare(&sql)?;

    let mut map = HashMap::with_capacity(900_000);

    let rows = stmt.query_map([], |row| {
        // genres stored as comma-separated string: "Action, Drama, Sci-Fi"
        let genres_str: String = row.get(4).unwrap_or_default();
        let genres: Vec<String> = genres_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // release_year stored as TEXT — parse to i32
        let release_year: Option<i32> = row
            .get::<_, Option<String>>(3)
            .unwrap_or(None)
            .and_then(|s| s.parse().ok());

        // is_anime is INTEGER 0/1
        let is_anime: bool = row.get::<_, i64>(10).unwrap_or(0) != 0;

        let media_type: String = row.get(1).unwrap_or_default();
        let tmdb_id = row.get::<_, i64>(0)? as u32;

        Ok(CatalogItem {
            id:            item_key(&media_type, tmdb_id),
            tmdb_id,
            media_type,
            title:         row.get(2).unwrap_or_default(),
            release_year,
            genres,
            overview:      row.get(5).unwrap_or_default(),
            poster_path:   row.get(6)?,
            backdrop_path: row.get(7)?,
            vote_average:  row.get::<_, f64>(8).unwrap_or(0.0) as f32,
            popularity:    row.get::<_, f64>(9).unwrap_or(0.0) as f32,
            is_anime,
            tagline:       row.get(11)?,
            director:      row.get(12)?,
            vote_count:    row.get::<_, i64>(13).unwrap_or(0).max(0) as u32,
            adult:         row.get::<_, i64>(14).unwrap_or(0) != 0,
            original_language: row.get::<_, Option<String>>(15).unwrap_or(None).unwrap_or_default(),
            tv_type:       row.get::<_, Option<String>>(16).unwrap_or(None).filter(|s| !s.is_empty()),
        })
    })?;

    for row in rows {
        let item = row?;
        map.insert(item.id, item);
    }

    info!("Catalog loaded: {} items", map.len());
    Ok(map)
}

/// Create a shared, atomically-swappable catalog pointer.
pub fn make_shared_catalog(db_path: &str) -> Result<SharedCatalog> {
    let map = load_catalog(db_path)?;
    Ok(Arc::new(ArcSwap::from_pointee(map)))
}

/// Background task: re-loads catalog from SQLite every 5 minutes and atomically
/// swaps the shared pointer. All readers see a consistent snapshot with zero locking.
pub fn spawn_catalog_refresh(catalog: SharedCatalog, db_path: String) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300));
        interval.tick().await; // skip immediate first tick
        loop {
            interval.tick().await;
            match tokio::task::spawn_blocking({
                let p = db_path.clone();
                move || load_catalog(&p)
            })
            .await
            {
                Ok(Ok(new_map)) => {
                    catalog.store(Arc::new(new_map));
                    info!("Catalog refreshed");
                }
                Ok(Err(e)) => warn!("Catalog refresh error: {e}"),
                Err(e)     => warn!("Catalog refresh join error: {e}"),
            }
        }
    });
}
