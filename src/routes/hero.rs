use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::app::AppState;
use crate::catalog::CatalogItem;

/// Candidates kept per surface/genre before the session-seeded draw.
const POOL_SIZE: usize = 50;
const MIN_VOTE_AVERAGE: f32 = 6.5;
const MIN_OVERVIEW_CHARS: usize = 60;
const MIN_POPULARITY: f32 = 15.0;
/// Formats that make poor hero banners.
const EXCLUDED_GENRES: [&str; 5] = ["News", "Talk", "Soap", "Reality", "Documentary"];

#[derive(Deserialize)]
pub struct HeroQuery {
    /// home | films | movies | series | tv
    #[serde(default = "default_surface")]
    pub surface: String,
    pub genre: Option<String>,
    pub session_id: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}
fn default_surface() -> String { "home".to_string() }
fn default_limit() -> usize { 8 }

fn surface_media_type(surface: &str) -> Option<&'static str> {
    match surface {
        "films" | "movies" | "movie" => Some("movie"),
        "series" | "tv" | "shows" => Some("tv"),
        _ => None,
    }
}

fn quality_score(item: &CatalogItem) -> f32 {
    item.vote_average * item.popularity.ln_1p()
}

/// Stable uniform value in (0, 1) derived from the session and item id.
fn session_uniform(session: &str, id: u32) -> f64 {
    let mut h = DefaultHasher::new();
    session.hash(&mut h);
    id.hash(&mut h);
    ((h.finish() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
}

fn hero_item_json(item: &CatalogItem) -> Value {
    json!({
        "id":            item.tmdb_id,
        "media_type":    item.media_type,
        "title":         item.title,
        "release_year":  item.release_year.map(|y| y.to_string()),
        "director":      item.director,
        "tagline":       item.tagline,
        "genres":        item.genres.join(", "),
        "overview":      item.overview,
        "poster_path":   item.poster_path,
        "backdrop_path": item.backdrop_path,
        "vote_average":  item.vote_average,
    })
}

/// GET /api/hero-feed?surface=&genre=&session_id=&limit=
/// Ranked hero carousel. The top quality pool for the surface/genre is sampled
/// with a session-seeded weighted draw so each session gets a stable but
/// different rotation.
pub async fn hero_feed(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HeroQuery>,
) -> impl IntoResponse {
    let surface = q.surface.to_lowercase();
    let genre = q.genre.as_deref().map(str::trim).filter(|g| !g.is_empty()).map(str::to_lowercase);
    // The client sends a genre id or name; both resolve through genre_spec for this surface
    let hero_surface = match surface_media_type(&surface) { Some(m) => m, None => "both" };
    let genre_spec = genre.as_deref().map(|g| crate::genre_spec::resolve(g, hero_surface));
    let nonscripted_ok = genre_spec.as_ref().map_or(false, |s| s.includes_nonscripted());
    let session = q.session_id.clone().unwrap_or_default();
    let limit = q.limit.clamp(1, 20);

    let session_state = if session.is_empty() { None } else { state.sessions.get(&session) };
    let version = session_state.as_ref().map_or(0, |s| s.lock().unwrap().version);
    let cache_key = format!("hero:{surface}:{}:{session}:{limit}:v{version}", genre.as_deref().unwrap_or(""));
    if let Some(hit) = state.feed_cache.get(&cache_key).await {
        if let Ok(v) = serde_json::from_str::<Value>(&hit) {
            return Json(v);
        }
    }

    let catalog = state.catalog.load();
    let media_type = surface_media_type(&surface);

    let seen = session_state.as_ref().map(|s| s.lock().unwrap().seen_ids().clone()).unwrap_or_default();

    let mut pool: Vec<&CatalogItem> = catalog
        .values()
        .filter(|i| {
            !seen.contains(&i.id)
                && i.backdrop_path.as_deref().is_some_and(|b| !b.is_empty())
                && i.poster_path.as_deref().is_some_and(|p| !p.is_empty())
                && i.vote_average >= MIN_VOTE_AVERAGE
                && i.popularity >= MIN_POPULARITY
                && (nonscripted_ok || !i.genres.iter().any(|g| EXCLUDED_GENRES.iter().any(|x| g.eq_ignore_ascii_case(x))))
                && i.overview_chars() >= MIN_OVERVIEW_CHARS
                && media_type.map_or(true, |m| i.media_type == m)
                && genre_spec.as_ref().map_or(true, |spec| spec.matches(i))
        })
        .collect();

    let by_score_desc = |a: &&CatalogItem, b: &&CatalogItem| {
        quality_score(b).partial_cmp(&quality_score(a)).unwrap_or(std::cmp::Ordering::Equal)
    };
    if pool.len() > POOL_SIZE {
        pool.select_nth_unstable_by(POOL_SIZE, by_score_desc);
        pool.truncate(POOL_SIZE);
    }
    pool.sort_by(by_score_desc);

    // Efraimidis-Spirakis weighted draw: key = u^(1/w), highest keys win.
    // Without a session the ranking stays purely by quality.
    let chosen: Vec<&CatalogItem> = if session.is_empty() {
        pool.into_iter().take(limit).collect()
    } else {
        let mut keyed: Vec<(f64, &CatalogItem)> = pool
            .into_iter()
            .map(|i| {
                let w = (quality_score(i) as f64).max(0.01);
                (session_uniform(&session, i.id).powf(1.0 / w), i)
            })
            .collect();
        keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        keyed.into_iter().take(limit).map(|(_, i)| i).collect()
    };

    let items: Vec<Value> = chosen.iter().map(|i| hero_item_json(i)).collect();
    let body = json!({
        "surface": surface,
        "genre":   genre,
        "count":   items.len(),
        "items":   items,
    });

    state.feed_cache.insert(cache_key, body.to_string()).await;
    Json(body)
}
