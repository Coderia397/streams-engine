//! GET /api/check/page-pool?surface=films&genre=Horror&rating=15: the titles a page may show in its rows (the page's filters, quality gate and vote floor), as
//! `{"pool": [["movie", 550], ...]}`. Used by ai-engine/scripts/verify_rows.py to rebuild every row from the source with the engine's own page rules instead of a copy of them.
use crate::app::AppState;
use crate::identity::{maturity_of, Maturity, MaybeCaller};
use crate::page::{PageAsk, PageContext};
use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct PoolQuery {
    pub surface: Option<String>,
    pub genre: Option<String>,
    pub language: Option<String>,
    pub rating: Option<String>,
}

pub async fn page_pool(State(state): State<Arc<AppState>>, MaybeCaller(caller): MaybeCaller, Query(q): Query<PoolQuery>) -> impl IntoResponse {
    let surface = match q.surface.as_deref() {
        Some("movie") | Some("films") | Some("film") | Some("movies") => "movie",
        Some("tv") | Some("series") | Some("shows") => "tv",
        _ => "both",
    };
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating));
    let genre = q.genre.as_deref().map(str::trim).filter(|g| !g.is_empty());
    let language = q.language.as_deref().map(|l| l.trim().to_lowercase()).filter(|l| !l.is_empty());
    let spec = genre.map(|g| crate::genre_spec::resolve(g, surface));
    let catalog = state.catalog.load();
    let fresh = state.fresh.load();
    let signals = crate::signals::Signals::default();
    let ask = PageAsk {
        surface,
        genre,
        genre_spec: spec.as_ref(),
        language: language.as_deref(),
        maturity,
        late_night: false,
        taste: vec![0.0; crate::vectors::DIMS],
        target_rows: 0,
        session: "",
        signals: &signals,
    };
    let ctx = PageContext::new(&state, &catalog, &fresh, ask);
    let mut pool: Vec<(&str, u32, u32)> = catalog.values().filter(|i| ctx.page_pool_member(i)).map(|i| (i.media_type.as_str(), i.tmdb_id, i.id)).collect();
    pool.sort_by_key(|p| p.2);
    Json(json!({ "surface": surface, "genre": genre, "count": pool.len(), "pool": pool.iter().map(|(m, id, _)| json!([m, id])).collect::<Vec<_>>() }))
}
