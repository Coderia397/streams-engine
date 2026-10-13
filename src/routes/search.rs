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
use crate::identity::{maturity_of, MaybeCaller, Maturity};

#[derive(Deserialize)]
pub struct SearchQuery {
    /// Age rating limit of a local profile (U, PG, 12, 15 or 18). Used only when no verified profile is sent; a verified profile's own setting wins.
    pub rating: Option<String>,
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
    MaybeCaller(caller): MaybeCaller,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| params.rating.as_deref().and_then(crate::identity::Maturity::from_rating));
    let q = params.q.trim().to_string();
    let wanted = params.limit.clamp(1, 100);
    // a restricted profile loses some matches to the filter, so fetch more and trim after
    let limit = if maturity.is_some_and(|m| m != Maturity::Adult) { (wanted * 4).min(200) } else { wanted };

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
                "id": item.tmdb_id, "title": item.title, "media_type": item.media_type,
                "poster_path": item.poster_path, "vote_average": item.vote_average,
                "genres": item.genres, "release_year": item.release_year, "rating": crate::catalog::cert_label(item.cert), "rating_gb": crate::catalog::gb_label(item.cert_gb), "rating_us": crate::catalog::us_label(item.cert_us),
            })).collect()
        }
    };

    let results: Vec<serde_json::Value> = match maturity {
        Some(m) if m != Maturity::Adult => {
            let catalog = state.catalog.load();
            results
                .into_iter()
                .filter(|r| {
                    let id = r.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);
                    let mt = r.get("media_type").and_then(|v| v.as_str());
                    id.and_then(|id| crate::catalog::resolve_key(&catalog, id, mt))
                        .and_then(|k| catalog.get(&k))
                        .map_or(false, |item| m.allows(item))
                })
                .take(wanted)
                .collect()
        }
        _ => results,
    };

    Json(json!({"results": results, "query": q}))
}

fn fts_search(db_path: &str, q: &str, limit: usize) -> anyhow::Result<Vec<serde_json::Value>> {
    let conn = Connection::open(db_path)?;
    // FTS5 MATCH query — escape quotes for safety
    let fts_q = q.replace('"', "\"\"");
    // catalog_v2 keeps the internal key in `id` (joined to the FTS table) and the TMDB id in `tmdb_id`;
    // the legacy catalog has only `id`, which is the TMDB id.
    let has_tmdb_id = conn
        .prepare("PRAGMA table_info(catalog)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|c| c == "tmdb_id");
    let id_expr = if has_tmdb_id { "c.tmdb_id" } else { "c.id" };

    let map_row = |row: &rusqlite::Row| -> rusqlite::Result<(i64, serde_json::Value)> {
        let genres_str: String = row.get::<_, Option<String>>(5).unwrap_or(None).unwrap_or_default();
        let genres: Vec<String> = genres_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let release_year: Option<i32> = row.get::<_, Option<String>>(6)
            .unwrap_or(None).and_then(|s| s.parse().ok());
        Ok((row.get::<_, i64>(8)?, json!({
            "id": row.get::<_, i64>(0)?,
            "title": row.get::<_, String>(1)?,
            "media_type": row.get::<_, String>(2)?,
            "poster_path": row.get::<_, Option<String>>(3)?,
            "vote_average": row.get::<_, f64>(4)?,
            "genres": genres,
            "release_year": release_year,
        })))
    };

    // 1. titles containing the query: the 3000 best text matches (keeps very common words fast), then exact
    //    title first, titles starting with the query next, then by popularity
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut out: Vec<serde_json::Value> = Vec::new();
    {
        let mut stmt = conn.prepare(&format!(
            "SELECT {id_expr}, c.title, c.media_type, c.poster_path, c.vote_average, c.genres, c.release_year, c.popularity, c.id
             FROM (SELECT id FROM catalog_fts WHERE catalog_fts MATCH ?1 ORDER BY rank LIMIT 3000) f
             JOIN catalog c ON c.id = f.id
             ORDER BY (lower(c.title) = lower(?2)) DESC, (lower(c.title) LIKE lower(?2) || '%') DESC, c.popularity DESC
             LIMIT ?3"
        ))?;
        let rows = stmt.query_map(
            rusqlite::params![format!("title : \"{}\"", fts_q), q, limit as i64],
            |row| map_row(row),
        )?;
        for r in rows.filter_map(|r| r.ok()) {
            if seen.insert(r.0) { out.push(r.1); }
        }
    }

    // 2. fill the rest with matches anywhere (overview, genres, director), best text match first
    if out.len() < limit {
        let mut stmt = conn.prepare(&format!(
            "SELECT {id_expr}, c.title, c.media_type, c.poster_path, c.vote_average, c.genres, c.release_year, c.popularity, c.id
             FROM catalog_fts f
             JOIN catalog c ON c.id = f.id
             WHERE catalog_fts MATCH ?1
             ORDER BY rank, c.popularity DESC
             LIMIT ?2"
        ))?;
        let rows = stmt.query_map(
            rusqlite::params![format!("\"{}\"", fts_q), limit as i64],
            |row| map_row(row),
        )?;
        for r in rows.filter_map(|r| r.ok()) {
            if out.len() >= limit { break; }
            if seen.insert(r.0) { out.push(r.1); }
        }
    }
    Ok(out)
}
