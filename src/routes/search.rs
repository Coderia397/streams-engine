use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use rusqlite::Connection;
use crate::app::AppState;

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}
fn default_limit() -> usize { 20 }

/// GET /api/search?q=<query>&limit=<n>
/// Uses the catalog_fts FTS5 index for fast full-text search.
/// Falls back to in-memory scan if the DB is unreachable.
pub async fn search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    let q = params.q.trim().to_string();
    let limit = params.limit.clamp(1, 100);

    if q.is_empty() {
        return Json(json!({"results": [], "query": q}));
    }

    // Try FTS5 first
    let db_path = state.config.db_path.clone();
    let q_clone = q.clone();
    let fts_result = tokio::task::spawn_blocking(move || {
        fts_search(&db_path, &q_clone, limit)
    }).await;

    let results = match fts_result {
        Ok(Ok(rows)) => rows,
        _ => {
            // Fallback: in-memory substring search over loaded catalog
            let catalog = state.catalog.load();
            let ql = q.to_lowercase();
            let mut items: Vec<_> = catalog
                .values()
                .filter(|item| item.title.to_lowercase().contains(&ql))
                .collect();
            items.sort_unstable_by(|a, b| b.popularity.partial_cmp(&a.popularity).unwrap());
            items.truncate(limit);
            items.iter().map(|item| json!({
                "id": item.id, "title": item.title, "media_type": item.media_type,
                "poster_path": item.poster_path, "vote_average": item.vote_average,
                "genres": item.genres, "release_year": item.release_year,
            })).collect()
        }
    };

    Json(json!({"results": results, "query": q}))
}

fn fts_search(db_path: &str, q: &str, limit: usize) -> anyhow::Result<Vec<serde_json::Value>> {
    let conn = Connection::open(db_path)?;
    // FTS5 MATCH query — escape quotes for safety
    let fts_q = q.replace('"', "\"\"");
    let mut stmt = conn.prepare(
        "SELECT c.id, c.title, c.media_type, c.poster_path, c.vote_average, c.genres, c.release_year, c.popularity
         FROM catalog_fts f
         JOIN catalog c ON c.id = f.id
         WHERE catalog_fts MATCH ?1
         ORDER BY rank, c.popularity DESC
         LIMIT ?2"
    )?;
    let rows: Vec<serde_json::Value> = stmt.query_map(
        rusqlite::params![format!("\"{}\"", fts_q), limit as i64],
        |row| {
            let genres_str: String = row.get::<_,String>(5).unwrap_or_default();
            let genres: Vec<String> = genres_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let release_year: Option<i32> = row.get::<_,Option<String>>(6)
                .unwrap_or(None).and_then(|s| s.parse().ok());
            Ok(json!({
                "id": row.get::<_,i64>(0)?,
                "title": row.get::<_,String>(1)?,
                "media_type": row.get::<_,String>(2)?,
                "poster_path": row.get::<_,Option<String>>(3)?,
                "vote_average": row.get::<_,f64>(4)?,
                "genres": genres,
                "release_year": release_year,
            }))
        }
    )?.filter_map(|r| r.ok()).collect();
    Ok(rows)
}
