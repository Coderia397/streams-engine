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

/// GET /api/season-chips: the events in season that have a page of their own (a genre page id), for the chips of the mobile top bar (Halloween in the weeks around it). An event
/// without a page id is left out; the list is empty out of season. The page of an event mixes films and series, like the home page.
pub async fn season_chips(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let today = crate::taxonomy::today_md();
    let chips: Vec<serde_json::Value> = crate::personal::events_in_season(&state.taxonomy, &state.vibe, today)
        .into_iter()
        .filter_map(|ev| {
            let label = crate::personal::prettify(ev);
            let id = crate::genre_pages::MOVIE_PAGES.iter().chain(crate::genre_pages::TV_PAGES.iter()).find(|(_, n)| n.eq_ignore_ascii_case(&label)).map(|(i, _)| *i)?;
            Some(json!({ "id": ev, "label_key": format!("event.{ev}"), "label": label, "genre_id": id }))
        })
        .collect();
    Json(json!({ "chips": chips }))
}

/// GET /api/features: switches the clients read to show or hide parts of the product (`{ "games": false }`). The file `features.json` next to the database is read on every
/// request, so a switch changes at once without a restart; a missing file or switch means off.
pub async fn features(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let path = std::path::Path::new(&state.config.db_path).parent().map(|d| d.join("features.json"));
    let flags: serde_json::Value = path.and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}));
    let mut out = serde_json::Map::new();
    // the flags the page knows of, each off unless the file turns it on
    out.insert("games".to_string(), json!(flags.get("games").and_then(|v| v.as_bool()).unwrap_or(false)));
    // everything but the visitor pages needs an account, unless the file turns that off for a while
    out.insert("accounts_required".to_string(), json!(flags.get("accounts_required").and_then(|v| v.as_bool()).unwrap_or(true)));
    Json(serde_json::Value::Object(out))
}

/// GET /api/games: the games to show on the New & Hot page and the Games page, `{ "games": [ { "id", "title", "description", "image", "url", "genre" } ] }`, from `games.json` next to the
/// database (read on every request). Empty while there is no file or Games is switched off in `features.json`.
pub async fn games(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let dir = std::path::Path::new(&state.config.db_path).parent().map(|d| d.to_path_buf());
    let read = |name: &str| dir.as_ref().and_then(|d| std::fs::read_to_string(d.join(name)).ok()).and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
    let on = read("features.json").and_then(|f| f.get("games").and_then(|g| g.as_bool())).unwrap_or(false);
    let list = if on { read("games.json").filter(|v| v.is_array()).unwrap_or_else(|| json!([])) } else { json!([]) };
    Json(json!({ "games": list }))
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
