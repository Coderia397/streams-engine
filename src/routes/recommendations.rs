use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use crate::app::AppState;

#[derive(Deserialize)]
pub struct RecsQuery {
    /// TMDB item ID to base recommendations on
    pub id: u32,
    #[serde(default = "default_limit")]
    pub limit: usize,
}
fn default_limit() -> usize { 20 }

/// GET /api/recommendations?id=<tmdb_id>&limit=<n>
/// Returns items most similar to the given item via vector search.
pub async fn recommendations(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecsQuery>,
) -> impl IntoResponse {
    let catalog = state.catalog.load();

    let Some(anchor_item) = catalog.get(&params.id) else {
        return Json(json!({"error": "item not found", "id": params.id}));
    };

    let Some(anchor_idx) = state.vectors.index_of(params.id) else {
        return Json(json!({"error": "no vector for item", "id": params.id}));
    };

    let anchor_vec = state.vectors.get_f32(anchor_idx);
    let limit = params.limit.clamp(1, 100);

    // Vector similarity search
    let similar = state.vectors.brute_force_topk(&anchor_vec, limit + 1);

    let results: Vec<_> = similar
        .iter()
        .filter(|(id, _)| *id != params.id) // exclude the anchor itself
        .take(limit)
        .filter_map(|(id, sim)| {
            let item = catalog.get(id)?;
            Some(json!({
                "id":           item.id,
                "title":        item.title,
                "media_type":   item.media_type,
                "poster_path":  item.poster_path,
                "vote_average": item.vote_average,
                "genres":       item.genres,
                "release_year": item.release_year,
                "similarity":   sim,
            }))
        })
        .collect();

    Json(json!({
        "anchor": { "id": anchor_item.id, "title": anchor_item.title },
        "results": results,
    }))
}
