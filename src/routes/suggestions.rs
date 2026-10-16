//! `GET /api/search-suggestions`: what the search page offers before anything is typed. Titles the visitor has not started, well known and well rated, leaning towards what the session has
//! shown interest in (its learned genre taste), different per session and every six hours, films and series alternating, at most three of one genre. Without a taste it is the best known titles.

use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::app::AppState;
use crate::catalog::CatalogItem;
use crate::identity::{maturity_of, MaybeCaller, Maturity};

#[derive(Deserialize)]
pub struct SuggestionsQuery {
    pub session_id: Option<String>,
    pub rating: Option<String>,
    pub limit: Option<usize>,
}

pub async fn suggestions(State(state): State<Arc<AppState>>, MaybeCaller(caller): MaybeCaller, Query(q): Query<SuggestionsQuery>) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating));
    let limit = q.limit.unwrap_or(12).clamp(1, 30);
    let session = q.session_id.clone().unwrap_or_else(|| "-".into());
    let slot = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600)) as usize;

    let (taste, seen) = match q.session_id.as_deref().and_then(|sid| state.sessions.get(sid)) {
        Some(s) => { let s = s.lock().unwrap(); (s.taste(), s.seen_ids().clone()) }
        None => (Vec::new(), Default::default()),
    };
    let taste_key = taste.iter().map(|(g, w)| format!("{g}{:.1}", w)).collect::<Vec<_>>().join(",");
    let cache_key = format!("suggest:{}:{}:{}:{}:{}:{}", session, slot, limit, maturity.map_or("-", |m| m.as_str()), taste_key, seen.len());
    if let Some(hit) = state.feed_cache.get(&cache_key).await {
        if let Ok(v) = serde_json::from_str::<Value>(&hit) { return Json(v); }
    }

    let catalog = state.catalog.load();
    let year = crate::fresh::ymd(crate::fresh::today_days())[..4].parse::<i32>().unwrap_or(2026);
    let affinity = |i: &CatalogItem| -> f32 { i.genres.iter().map(|g| taste.iter().find(|(t, _)| t == g).map_or(0.0, |(_, w)| *w)).sum() };

    let mut scored: Vec<(f32, &CatalogItem)> = catalog
        .values()
        .filter(|i| i.popularity >= 8.0 && crate::page::chart_ok(i) && i.release_year.is_none_or(|y| y <= year) && !seen.contains(&i.id))
        .filter(|i| maturity.is_none_or(|m| crate::page::maturity_allows_with(m, &state.profiles, i)))
        .map(|i| (
            (1.0 + i.popularity).ln() + 0.25 * i.vote_average + 4.0 * affinity(i) + 1.6 * crate::page::entry_noise(&session, i.id as usize + slot * 7),
            i,
        ))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.id.cmp(&b.1.id)));

    // at most three of a genre, films and series alternating
    let mut per_genre: HashMap<&str, usize> = HashMap::new();
    let (mut films, mut series): (Vec<&CatalogItem>, Vec<&CatalogItem>) = (Vec::new(), Vec::new());
    for (_, i) in scored.into_iter().take(600) {
        let n = per_genre.entry(i.genres.first().map_or("", |g| g.as_str())).or_insert(0);
        if *n >= 3 { continue; }
        *n += 1;
        if i.media_type == "tv" { series.push(i) } else { films.push(i) }
    }
    let mut picked: Vec<&CatalogItem> = Vec::new();
    let (mut f, mut s) = (films.into_iter(), series.into_iter());
    while picked.len() < limit {
        match (f.next(), s.next()) {
            (None, None) => break,
            (a, b) => { picked.extend(a); picked.extend(b); }
        }
    }
    picked.truncate(limit);

    let body = json!({
        "title_key": "search.recommended",
        "personalised": !taste.is_empty(),
        "results": picked.into_iter().map(|i| json!({
            "id": i.tmdb_id, "title": i.title, "media_type": i.media_type, "poster_path": i.poster_path, "backdrop_path": i.backdrop_path,
            "vote_average": i.vote_average, "genres": i.genres, "release_year": i.release_year, "overview": i.overview,
            "rating": crate::catalog::cert_label(i.cert), "rating_gb": crate::catalog::gb_label(i.cert_gb), "rating_us": crate::catalog::us_label(i.cert_us),
        })).collect::<Vec<_>>(),
    });
    state.feed_cache.insert(cache_key, body.to_string()).await;
    Json(body)
}
