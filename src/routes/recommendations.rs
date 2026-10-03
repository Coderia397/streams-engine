use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use crate::app::AppState;
use crate::hnsw::ENGINE_READY;
use crate::vectors::l2_normalize;

#[derive(Deserialize)]
pub struct RecsQuery {
    /// TMDB item ID to base recommendations on
    pub id: u32,
    /// "movie" or "tv". TMDB ids repeat across the two, so send it; without it the movie is tried first.
    pub media_type: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}
fn default_limit() -> usize { 20 }

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

/// Unit vector of an item, centred on the corpus mean. Raw pairs in this space share a high baseline
/// cosine, so match percentages come from the centred value.
fn centred(state: &AppState, idx: usize) -> Vec<f32> {
    let mut v: Vec<f32> = state.vectors.get_f32(idx).iter().zip(state.mean.iter()).map(|(x, m)| x - m).collect();
    l2_normalize(&mut v);
    v
}

/// GET /api/recommendations?id=<tmdb_id>&media_type=<movie|tv>&limit=<n>
/// Items most similar to the given item by vector search. The response carries `recommendations`
/// (the shape the frontend reads: match_percentage, cosine_similarity, vibe_pills) and `results`
/// (older shape, same items).
pub async fn recommendations(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecsQuery>,
) -> impl IntoResponse {
    let catalog = state.catalog.load();

    let Some(anchor_key) = crate::catalog::resolve_key(&catalog, params.id, params.media_type.as_deref()) else {
        return Json(json!({"error": "item not found", "id": params.id}));
    };
    let anchor_item = &catalog[&anchor_key];

    let Some(anchor_idx) = state.vectors.index_of(anchor_key) else {
        return Json(json!({"error": "no vector for item", "id": params.id}));
    };

    let anchor_vec = state.vectors.get_f32(anchor_idx);
    let limit = params.limit.clamp(1, 100);

    // HNSW when the index is ready, brute force otherwise
    let mut similar: Vec<(u32, f32)> = Vec::new();
    if ENGINE_READY.load(Ordering::Relaxed) {
        let guard = state.hnsw.read().await;
        if let Some(idx) = guard.as_ref() {
            if let Ok(r) = idx.search(&anchor_vec, limit + 1) { similar = r; }
        }
    }
    if similar.is_empty() {
        similar = state.vectors.brute_force_topk(&anchor_vec, limit + 1);
    }

    let anchor_c = centred(&state, anchor_idx);
    let anchor_genres: Vec<&String> = anchor_item.genres.iter().collect();

    let results: Vec<_> = similar
        .iter()
        .filter(|(id, _)| *id != anchor_key) // exclude the anchor itself
        .take(limit)
        .filter_map(|(id, sim)| {
            let item = catalog.get(id)?;
            let c = state.vectors.index_of(*id).map(|i| dot(&anchor_c, &centred(&state, i))).unwrap_or(0.0);
            let shared: Vec<&String> = item.genres.iter().filter(|g| anchor_genres.contains(g)).take(2).collect();
            let pills: Vec<&String> = if shared.is_empty() { item.genres.iter().take(2).collect() } else { shared };
            Some(json!({
                "id":                item.tmdb_id,
                "title":             item.title,
                "media_type":        item.media_type,
                "release_year":      item.release_year.map(|y| y.to_string()),
                "genres":            item.genres.join(", "),
                "overview":          item.overview,
                "tagline":           item.tagline,
                "director":          item.director,
                "poster_path":       item.poster_path,
                "backdrop_path":     item.backdrop_path,
                "vote_average":      item.vote_average,
                "similarity":        sim,
                "cosine_similarity": sim,
                "match_percentage":  (c.clamp(0.0, 0.99) * 100.0).round() as u32,
                "vibe_pills":        pills,
            }))
        })
        .collect();

    Json(json!({
        "anchor": { "id": anchor_item.tmdb_id, "media_type": anchor_item.media_type, "title": anchor_item.title },
        "target_id": anchor_item.tmdb_id,
        "target_title": anchor_item.title,
        "recommendations": results.clone(),
        "results": results,
    }))
}
