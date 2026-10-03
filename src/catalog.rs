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
    pub id: u32,
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

/// Load the entire catalog from SQLite into a HashMap keyed by TMDB ID.
/// This is the ONLY place SQLite is accessed — never on the request hot path.
pub fn load_catalog(db_path: &str) -> Result<CatalogMap> {
    let conn = Connection::open(db_path)?;

    let mut stmt = conn.prepare(
        "SELECT id, media_type, title, release_year, genres,
                overview, poster_path, backdrop_path,
                vote_average, popularity, is_anime, tagline, director
         FROM catalog",
    )?;

    let mut map = HashMap::with_capacity(750_000);

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

        Ok(CatalogItem {
            id:            row.get::<_, i64>(0)? as u32,
            media_type:    row.get(1).unwrap_or_default(),
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
