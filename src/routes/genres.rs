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

#[derive(Deserialize)]
pub struct GenreListQuery {
    pub surface: Option<String>,
    pub rating: Option<String>,
}

/// GET /api/genre-list?surface=movie|tv&rating=PG: the genre pages of a surface in picker order with the id of their address (`/browse/genre/<id>?bc=<page id>`), and for this profile
/// how many rows each can fill and whether it is offered, so a client does not keep a list of its own.
pub async fn genre_list(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(q): Query<GenreListQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating)).or(Some(Maturity::Adult));
    let surface = if matches!(q.surface.as_deref(), Some("tv") | Some("series")) { "tv" } else { "movie" };
    let catalog = state.catalog.load();
    let genres: Vec<serde_json::Value> = crate::genre_pages::pages(surface)
        .iter()
        .map(|(id, name)| {
            let spec = crate::genre_spec::resolve(name, surface);
            let cap = crate::routes::feed::genre_row_cap(&catalog, &state.profiles, surface, &spec, maturity);
            json!({ "id": id, "name": name, "row_cap": cap, "supported": cap >= crate::routes::feed::MIN_GENRE_PAGE_ROWS })
        })
        .collect();
    let page = if surface == "tv" { crate::genre_pages::SERIES_PAGE_ID } else { crate::genre_pages::FILMS_PAGE_ID };
    Json(json!({ "surface": surface, "page_id": page, "genres": genres }))
}

#[derive(Deserialize)]
pub struct LanguageSupportQuery {
    pub surface: Option<String>,
    /// Comma separated original language codes (at most 60)
    pub codes: String,
    pub rating: Option<String>,
}

/// GET /api/language-support?codes=ko,ja&rating=PG: rows a language page can fill for this profile, and whether the page is worth offering.
pub async fn language_support(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(q): Query<LanguageSupportQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating));
    let surface = match q.surface.as_deref() {
        Some("tv") | Some("series") => "tv",
        Some("movie") | Some("films") => "movie",
        _ => "both",
    };
    let catalog = state.catalog.load();
    let mut out = serde_json::Map::new();
    for code in q.codes.split(',').map(str::trim).filter(|c| !c.is_empty() && c.len() <= 3).take(60) {
        let cap = crate::routes::feed::language_row_cap(&catalog, &state.profiles, surface, &code.to_lowercase(), maturity);
        out.insert(code.to_string(), json!({ "row_cap": cap, "supported": cap >= crate::routes::feed::MIN_GENRE_PAGE_ROWS }));
    }
    Json(json!({ "languages": out }))
}
