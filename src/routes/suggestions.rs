//! `GET /api/search-suggestions`: what the search page offers before anything is typed. A mix taken in turn from several groups: titles that fit the session's learned taste, what is most
//! popular today, this year's releases, the critically acclaimed, the event in season (Halloween while it is on) and one judged topic. Each group has its own measure, a little noise per
//! session and six hour slot, titles the visitor has started are left out, and at most three titles of one genre are shown. Each result says which group it came from (`source`).

use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use crate::det::{HashMap, HashSet};
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
    let today = crate::fresh::today_days();
    let year = crate::fresh::ymd(today)[..4].parse::<i32>().unwrap_or(2026);
    let affinity = |i: &CatalogItem| -> f32 { i.genres.iter().map(|g| taste.iter().find(|(t, _)| t == g).map_or(0.0, |(_, w)| *w)).sum() };
    let noise = |i: &CatalogItem, salt: usize| crate::page::entry_noise(&session, i.id as usize + slot * 7 + salt);
    // what may be offered at all: complete, well rated, out already, not started, within the age limit
    let ok = |i: &CatalogItem| crate::page::chart_ok(i) && i.release_year.is_none_or(|y| y <= year) && !seen.contains(&i.id) && maturity.is_none_or(|m| crate::page::maturity_allows_with(m, &state.profiles, i));

    let pool: Vec<&CatalogItem> = catalog.values().filter(|i| i.popularity >= 6.0 && ok(i)).collect();
    fn by<'a>(mut v: Vec<(f32, &'a CatalogItem)>) -> Vec<&'a CatalogItem> {
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.id.cmp(&b.1.id)));
        v.into_iter().take(60).map(|(_, i)| i).collect()
    }

    // several groups, each ranked by its own measure with a little noise per session and slot; the page takes from each in turn
    let mut sources: Vec<(&'static str, Vec<&CatalogItem>)> = Vec::new();
    if !taste.is_empty() {
        sources.push(("for_you", by(pool.iter().copied().filter(|i| affinity(i) > 0.0).map(|i| ((1.0 + i.popularity).ln() + 0.25 * i.vote_average + 5.0 * affinity(i) + 1.4 * noise(i, 1), i)).collect())));
    }
    sources.push(("top10", by(pool.iter().copied().map(|i| (i.popularity + 25.0 * noise(i, 2), i)).collect())));
    sources.push(("new", by(pool.iter().copied().filter(|i| i.release_year.is_some_and(|y| y >= year)).map(|i| ((1.0 + i.popularity).ln() + 1.4 * noise(i, 3), i)).collect())));
    sources.push(("acclaimed", by(pool.iter().copied().filter(|i| i.vote_average >= 7.6 && i.vote_count >= 500).map(|i| (i.vote_average + 0.3 * (i.vote_count as f32).ln() + 1.4 * noise(i, 4), i)).collect())));
    // the event in season (Halloween while it is on): its best fitting titles
    let events = crate::personal::events_in_season(&state.taxonomy, &state.vibe, crate::taxonomy::today_md());
    let season: Vec<&CatalogItem> = events.iter().filter_map(|ev| state.vibe.seasons.get(*ev)).flat_map(|subs| subs.iter().flat_map(|sub| sub.items.iter())).filter_map(|k| catalog.get(k)).filter(|i| ok(i)).take(40).collect();
    if !season.is_empty() {
        sources.push(("season", season));
    }
    // one topic of the judged topics, chosen by the session and the slot
    let mut slugs: Vec<&String> = catalog.topic_keys.iter().filter(|(_, v)| v.len() >= 30).map(|(k, _)| k).collect();
    slugs.sort();
    if !slugs.is_empty() {
        let slug = slugs[(crate::page::entry_noise(&session, 41 + slot) * slugs.len() as f32) as usize % slugs.len()];
        let items: Vec<&CatalogItem> = catalog.topic_keys[slug].iter().filter_map(|k| catalog.get(k)).filter(|i| i.popularity >= 3.0 && ok(i)).take(40).collect();
        if !items.is_empty() {
            sources.push(("topic", items));
        }
    }

    // the order of the groups changes with the session and the slot (the first group gives the first title); then one title from each group in turn, never twice the same title, at most
    // three of a genre
    sources.sort_by(|a, b| crate::page::entry_noise(&session, a.0.len() * 13 + slot).partial_cmp(&crate::page::entry_noise(&session, b.0.len() * 13 + slot)).unwrap_or(std::cmp::Ordering::Equal));
    let mut cursors = vec![0usize; sources.len()];
    let mut used: HashSet<u32> = HashSet::default();
    let mut per_genre: HashMap<&str, usize> = HashMap::default();
    let mut picked: Vec<(&'static str, &CatalogItem)> = Vec::new();
    'fill: loop {
        let mut progressed = false;
        for (n, (name, list)) in sources.iter().enumerate() {
            while cursors[n] < list.len() {
                let i = list[cursors[n]];
                cursors[n] += 1;
                let g = per_genre.entry(i.genres.first().map_or("", |g| g.as_str())).or_insert(0);
                if used.contains(&i.id) || *g >= 3 { continue; }
                *g += 1;
                used.insert(i.id);
                picked.push((name, i));
                progressed = true;
                break;
            }
            if picked.len() >= limit { break 'fill; }
        }
        if !progressed { break; }
    }

    let body = json!({
        "title_key": "search.recommended",
        "personalised": !taste.is_empty(),
        "results": picked.into_iter().map(|(source, i)| json!({
            "source": source, "id": i.tmdb_id, "title": i.title, "media_type": i.media_type, "poster_path": i.poster_path, "backdrop_path": i.backdrop_path,
            "vote_average": i.vote_average, "genres": i.genres, "release_year": i.release_year, "overview": i.overview,
            "rating": crate::catalog::cert_label(i.cert), "rating_gb": crate::catalog::gb_label(i.cert_gb), "rating_us": crate::catalog::us_label(i.cert_us),
        })).collect::<Vec<_>>(),
    });
    state.feed_cache.insert(cache_key, body.to_string()).await;
    Json(body)
}
