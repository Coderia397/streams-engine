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
use crate::identity::{maturity_of, MaybeCaller, Maturity};
use crate::vectors::l2_normalize;

#[derive(Deserialize)]
pub struct RecsQuery {
    /// Age rating limit of a local profile (U, PG, 12, 15 or 18). Used only when no verified profile is sent; a verified profile's own setting wins.
    pub rating: Option<String>,
    /// TMDB item ID to base recommendations on
    pub id: u32,
    /// "movie" or "tv". TMDB ids repeat across the two, so send it; without it the movie is tried first.
    pub media_type: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Tuning only (ai-engine/scripts/eval_recs.py): weight of popularity, of rating, and of the profile similarity in the mean of the two similarities
    pub w_pop: Option<f32>,
    pub w_rate: Option<f32>,
    pub w_profile: Option<f32>,
    /// Tuning: boost for a title in the anchor's world level 2 group, atom and theme group
    pub w_g2: Option<f32>,
    pub w_atom: Option<f32>,
    pub w_theme: Option<f32>,
}
fn default_limit() -> usize { 20 }

/// Weights of popularity and rating in the order of recommendations (measured with ai-engine/scripts/eval_recs.py).
const REC_POPULARITY_WEIGHT: f32 = 0.3;
const REC_RATING_WEIGHT: f32 = 0.06;
/// Share of the profile similarity in the mean of the two similarities. Zero: measured with eval_recs.py on 19 anchors, the text similarity alone
/// gave 4.3 good titles in the first 10 and the profile similarity alone 1.4 (a mean of the two 2.8 to 3.8), so the profile space is only used on request.
const REC_PROFILE_WEIGHT: f32 = 0.0;
const REC_G2_WEIGHT: f32 = 0.0;
const REC_ATOM_WEIGHT: f32 = 0.0;
const REC_THEME_WEIGHT: f32 = 0.0;

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
    MaybeCaller(caller): MaybeCaller,
    Query(params): Query<RecsQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| params.rating.as_deref().and_then(crate::identity::Maturity::from_rating));
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
    // many neighbours are dropped (age limit, repeats of one franchise, shorts), so look well beyond the page
    let restricted = maturity.is_some_and(|m| m != Maturity::Adult);
    let fetch = (limit * 8).max(80) * if restricted { 2 } else { 1 } + 1;

    let (w_g2, w_atom, w_theme) = (params.w_g2.unwrap_or(REC_G2_WEIGHT), params.w_atom.unwrap_or(REC_ATOM_WEIGHT), params.w_theme.unwrap_or(REC_THEME_WEIGHT));
    let anchor_groups = state.groups.get(&anchor_key).copied();
    let (w_pop, w_rate, w_profile) = (params.w_pop.unwrap_or(REC_POPULARITY_WEIGHT), params.w_rate.unwrap_or(REC_RATING_WEIGHT), params.w_profile.unwrap_or(REC_PROFILE_WEIGHT).clamp(0.0, 1.0));

    // neighbours by what the catalog text says (HNSW when ready, brute force otherwise) and by what the title is about (profile vectors)
    let mut similar: Vec<(u32, f32)> = Vec::new();
    if ENGINE_READY.load(Ordering::Relaxed) {
        let guard = state.hnsw.read().await;
        if let Some(idx) = guard.as_ref() {
            if let Ok(r) = idx.search(&anchor_vec, fetch) { similar = r; }
        }
    }
    if similar.is_empty() {
        similar = state.vectors.brute_force_topk(&anchor_vec, fetch);
    }
    let profile_near = if w_profile > 0.0 { state.profiles.nearest(anchor_key, fetch.min(160)) } else { Vec::new() };
    let mut keys: Vec<u32> = similar.iter().map(|(k, _)| *k).chain(profile_near.iter().map(|(k, _)| *k)).filter(|k| *k != anchor_key).collect();
    let mut seen_keys = std::collections::HashSet::new();
    keys.retain(|k| seen_keys.insert(*k));

    let anchor_c = centred(&state, anchor_idx);
    let anchor_genres: Vec<&String> = anchor_item.genres.iter().collect();
    let anchor_title = crate::people::normalise(&anchor_item.title);

    // score: the mean of the two similarities when the title has a profile, plus a small push for titles people watch and rate well
    let mut scored: Vec<(f32, f32, &crate::catalog::CatalogItem)> = keys
        .iter()
        .filter_map(|k| {
            let item = catalog.get(k)?;
            if !maturity.is_none_or(|m| crate::page::maturity_allows_with(m, &state.profiles, item)) {
                return None;
            }
            // a short (a clip, a making of) is no match for a feature, and another version of the same title is not a new suggestion
            if item.media_type == "movie" && item.runtime > 0 && item.runtime < 40 && anchor_item.runtime >= 60 {
                return None;
            }
            if item.media_type == anchor_item.media_type && crate::people::normalise(&item.title) == anchor_title {
                return None;
            }
            let from_text = state.vectors.index_of(*k).map(|i| dot(&anchor_c, &centred(&state, i)));
            let from_profile = if w_profile > 0.0 { state.profiles.centred_cos(anchor_key, *k) } else { None };
            let sim = match (from_text, from_profile) {
                (Some(a), Some(b)) => (1.0 - w_profile) * a + w_profile * b,
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => return None,
            };
            let quality = (item.popularity.ln_1p() / 100f32.ln_1p()).clamp(0.0, 1.0);
            let rating = ((item.vote_average - 5.0) / 4.0).clamp(0.0, 1.0);
            let group_bonus = match (anchor_groups, state.groups.get(k)) {
                (Some(a), Some(b)) => (if a.w2 >= 0 && a.w2 == b.w2 { w_g2 } else { 0.0 }) + (if a.atom >= 0 && a.atom == b.atom { w_atom } else { 0.0 }) + (if a.theme >= 0 && a.theme == b.theme { w_theme } else { 0.0 }),
                _ => 0.0,
            };
            Some((sim + w_pop * quality + w_rate * rating + group_bonus, sim, item))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.2.id.cmp(&b.2.id)));

    // one title once (the best of its versions) and at most three titles from one franchise (the first word they share)
    let mut seen_titles: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut franchise: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let results: Vec<_> = scored
        .into_iter()
        .filter(|(_, _, item)| {
            let norm = crate::people::normalise(&item.title);
            if !seen_titles.insert((item.media_type.clone(), norm.clone())) {
                return false;
            }
            let first = norm.split(' ').next().unwrap_or("").to_string();
            if first.chars().count() >= 4 && first != "the" {
                let n = franchise.entry(first).or_insert(0);
                *n += 1;
                return *n <= 3;
            }
            true
        })
        .take(limit)
        .map(|(_, sim, item)| {
            let shared: Vec<&String> = item.genres.iter().filter(|g| anchor_genres.contains(g)).take(2).collect();
            let pills: Vec<&String> = if shared.is_empty() { item.genres.iter().take(2).collect() } else { shared };
            json!({
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
                "rating":            crate::catalog::cert_label(item.cert),
                "rating_gb":         crate::catalog::gb_label(item.cert_gb),
                "rating_us":         crate::catalog::us_label(item.cert_us),
                "similarity":        sim,
                "cosine_similarity": sim,
                "match_percentage":  (sim.clamp(0.0, 0.99) * 100.0).round() as u32,
                "vibe_pills":        pills,
            })
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
