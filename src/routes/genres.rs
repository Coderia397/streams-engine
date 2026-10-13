//! GET /api/genre-support?surface=movie|tv|both&names=Action,Comedies&rating=PG: for each genre name, how many rows its page can fill for this
//! profile and whether the page is worth offering (at least `MIN_GENRE_PAGE_ROWS`). A limited profile is only offered the pages that have enough
//! titles; the numbers are the same ones the feed uses for its row cap.
use crate::app::AppState;
use crate::identity::{maturity_of, MaybeCaller, Maturity};
use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct GenreSupportQuery {
    pub surface: Option<String>,
    /// Comma separated genre names as the client shows them (at most 80)
    pub names: String,
    pub rating: Option<String>,
}

pub async fn genre_support(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(q): Query<GenreSupportQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating));
    let surface = match q.surface.as_deref() {
        Some("tv") | Some("series") => "tv",
        Some("movie") | Some("films") => "movie",
        _ => "both",
    };
    let catalog = state.catalog.load();
    let mut out = serde_json::Map::new();
    for name in q.names.split(',').map(str::trim).filter(|n| !n.is_empty()).take(80) {
        let spec = crate::genre_spec::resolve(name, surface);
        let cap = crate::routes::feed::genre_row_cap(&catalog, &state.profiles, surface, &spec, maturity);
        out.insert(name.to_string(), json!({ "row_cap": cap, "supported": cap >= crate::routes::feed::MIN_GENRE_PAGE_ROWS }));
    }
    Json(json!({ "genres": out }))
}
