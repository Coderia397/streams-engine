//! GET /api/rating?id=<tmdb id>&media_type=movie|tv: the British age rating of one title, or NR (not rated).
use crate::app::AppState;
use crate::catalog::{cert_label, gb_label, item_key, us_label};
use crate::identity::{maturity_of, MaybeCaller, Maturity};
use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct RatingQuery {
    pub id: u32,
    pub media_type: Option<String>,
}

pub async fn rating(State(state): State<Arc<AppState>>, Query(q): Query<RatingQuery>) -> impl IntoResponse {
    let catalog = state.catalog.load();
    let media = q.media_type.as_deref().unwrap_or("movie");
    let media = if media == "tv" { "tv" } else { "movie" };
    let item = catalog.get(&item_key(media, q.id));
    Json(json!({
        "id": q.id, "media_type": media,
        "rating": item.map_or("NR", |i| cert_label(i.cert)),
        "rating_gb": item.and_then(|i| gb_label(i.cert_gb)),
        "rating_us": item.and_then(|i| us_label(i.cert_us)),
    }))
}

#[derive(Deserialize)]
pub struct RatingsQuery {
    /// Comma separated keys such as `movie:550,tv:1399` (at most 100)
    pub ids: String,
    /// Age rating limit of a local profile (U, PG, 12, 15 or 18); a verified profile's own setting wins
    pub rating: Option<String>,
}

/// GET /api/ratings?ids=movie:550,tv:1399&rating=PG: for each title, its rating and whether the profile may see it. Titles the catalog
/// does not hold are allowed for 18 (no limit) and refused for every lower limit (an unknown title cannot be shown to a limited profile).
pub async fn ratings(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(q): Query<RatingsQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating));
    let catalog = state.catalog.load();
    let mut allowed = Vec::new();
    let mut labels = serde_json::Map::new();
    for key in q.ids.split(',').map(str::trim).filter(|k| !k.is_empty()).take(100) {
        let Some((media, id)) = key.split_once(':') else { continue };
        let Ok(id) = id.parse::<u32>() else { continue };
        let media = if media == "tv" { "tv" } else { "movie" };
        let item = catalog.get(&item_key(media, id));
        labels.insert(key.to_string(), json!({
            "rating": item.map_or("NR", |i| cert_label(i.cert)),
            "rating_gb": item.and_then(|i| gb_label(i.cert_gb)),
            "rating_us": item.and_then(|i| us_label(i.cert_us)),
        }));
        let ok = match (maturity, item) {
            (None, _) | (Some(Maturity::Adult), _) => true,
            (Some(m), Some(i)) => m.allows(i),
            (Some(_), None) => false,
        };
        if ok {
            allowed.push(key.to_string());
        }
    }
    Json(json!({ "allowed": allowed, "ratings": labels }))
}
