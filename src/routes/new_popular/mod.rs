//! GET /api/new-popular: the New & Popular page. Without `tab` it returns the desktop rows (a few rows of charts, trending, new, coming
//! soon and so on); with `tab=watching|top10_movies|top10_series|just_landed|coming_soon` it returns one row for the mobile New & Hot
//! tab. Rows come from the catalog (popularity refreshed daily) and `fresh.json` (real release dates, titles too new for the catalog).
//! Every row respects the profile's age limit; a title with no known certificate and no catalog entry is shown only to unrestricted profiles.
use crate::app::AppState;
use crate::catalog::{resolve_key, CatalogItem, CatalogMap};
use crate::fresh::{parse_days, today_days, ymd, FreshItem, FreshStore};
use crate::identity::{maturity_of, MaybeCaller, Maturity};
use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use crate::det::{HashMap, HashSet};
use std::sync::Arc;

mod candidates;
mod charts;
mod rows;

use candidates::*;
use charts::*;
use rows::*;

/// Netflix's New on Netflix and Worth the Wait hold 40 titles (note 97)
const ROW_TITLES: usize = 40;
const MIN_ROW_ITEMS: usize = 6;
/// A row of upcoming titles is shown from three (Netflix's Kids Coming Next Week had 3)
const MIN_UPCOMING_ITEMS: usize = 3;
const DESKTOP_ROWS: usize = 6;

#[derive(Deserialize)]
pub struct NewPopularQuery {
    pub rating: Option<String>,
    /// watching | top10_movies | top10_series | just_landed | coming_soon (mobile New & Hot tabs); none gives the desktop page
    pub tab: Option<String>,
    pub limit: Option<usize>,
    /// The visitor's session: with the six hour slot it decides the order of the rows and which languages and genres get one
    pub session_id: Option<String>,
}
















pub async fn new_popular(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(q): Query<NewPopularQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating));
    let catalog = state.catalog.load();
    let fresh = state.fresh.load();
    let today_n = today_days();
    let today = ymd(today_n);
    let year: i32 = today[..4].parse().unwrap_or(2026);
    let tab = q.tab.as_deref().map(str::trim).filter(|t| !t.is_empty());
    let limit = q.limit.unwrap_or(if tab.is_some() { 10 } else { ROW_TITLES }).clamp(1, ROW_TITLES);

    let slot = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600)) as usize;
    // the tabs of the phone are the same for everyone; the desktop page differs by session and six hour slot
    let seed_part = if tab.is_some() { "-".to_string() } else { format!("{}:{}", q.session_id.as_deref().unwrap_or("-"), slot) };
    let cache_key = format!("newpop:{}:{}:{}:{}:{}:{}", tab.unwrap_or("-"), limit, maturity.map_or("-", |m| m.as_str()), fresh.generated, today, seed_part);
    if let Some(hit) = state.feed_cache.get(&cache_key).await {
        if let Ok(v) = serde_json::from_str::<Value>(&hit) {
            return Json(v);
        }
    }

    let all = candidates(&catalog, &fresh, maturity, &state.profiles, year, &today);
    let pool: Vec<&Cand> = all.iter().filter(|c| c.allowed && has_art(c)).collect();

    let body = match tab {
        Some(t) => {
            let items: Vec<Cand> = match t {
                "top10_movies" => chart(&catalog, &fresh, maturity, &state.profiles, "movie", &today, year),
                "top10_series" => chart(&catalog, &fresh, maturity, &state.profiles, "tv", &today, year),
                "watching" => {
                    let mut v: Vec<&Cand> = pool.iter().copied().filter(|c| has_backdrop(c) && c.released(&today) && c.in_list("trending_day")).collect();
                    v.sort_by_key(|c| c.rank_in("trending_day"));
                    v.into_iter().take(limit).cloned().collect()
                }
                "just_landed" => {
                    // the most popular titles of the last 30 days, newest first
                    let v: Vec<&Cand> = pool.iter().copied().filter(|c| has_backdrop(c) && c.released(&today) && c.days_since(today_n).is_some_and(|d| d <= 30)).collect();
                    let mut top = take_top(v, limit);
                    top.sort_by(|a, b| b.release_date.cmp(&a.release_date).then_with(|| b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal)));
                    top
                }
                "coming_soon" => {
                    // the most anticipated titles of the next 90 days, soonest first
                    let v: Vec<&Cand> = pool.iter().copied().filter(|c| has_art(c) && c.upcoming(&today) && c.days_since(today_n).is_some_and(|d| d >= -90)).collect();
                    let mut top = take_top(v, limit);
                    top.sort_by(|a, b| a.release_date.cmp(&b.release_date).then_with(|| b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal)));
                    top
                }
                _ => Vec::new(),
            };
            let ranked = t == "top10_movies" || t == "top10_series";
            let items: Vec<Cand> = items.into_iter().take(limit).collect();
            let rows = if items.is_empty() {
                Vec::new()
            } else {
                vec![json!({
                    "id": format!("tab-{t}"), "row_type": "tab", "type": "", "title": t,
                    "items": items.iter().enumerate().map(|(i, c)| item_json(c, ranked.then_some(i + 1), today_n)).collect::<Vec<_>>(),
                })]
            };
            json!({ "surface": "new_popular", "tab": t, "generated": fresh.generated, "total_rows": rows.len(), "rows": rows })
        }
        None => {
            let rows = desktop_rows(&catalog, &fresh, &pool, maturity, &state.profiles, &today, today_n, year, limit, q.session_id.as_deref().unwrap_or("-"), slot);
            let mut seen: HashSet<(String, u32)> = HashSet::default();
            let mut out: Vec<Value> = Vec::new();
            for (i, row) in rows.into_iter().enumerate() {
                let ranked = row.row_type == "top_ten";
                // a Top 10 keeps every title (Netflix's charts share titles with New on Netflix); the other rows show a title once, in the first row that has it
                let items: Vec<Cand> = if ranked { row.items } else { row.items.into_iter().filter(|c| seen.insert(c.key())).take(limit).collect() };
                let min = if row.kind == "soon" { MIN_UPCOMING_ITEMS } else { MIN_ROW_ITEMS };
                if items.len() < min && !ranked {
                    continue;
                }
                if items.is_empty() {
                    continue;
                }
                out.push(json!({
                    "id": format!("newpop-{i}"), "row_type": row.row_type, "type": row.kind, "title": row.title, "title_key": row.key, "title_args": row.args,
                    "items": items.iter().enumerate().map(|(n, c)| item_json(c, ranked.then_some(n + 1), today_n)).collect::<Vec<_>>(),
                }));
                if out.len() >= DESKTOP_ROWS {
                    break;
                }
            }
            json!({ "surface": "new_popular", "generated": fresh.generated, "total_rows": out.len(), "rows": out })
        }
    };
    state.feed_cache.insert(cache_key, body.to_string()).await;
    Json(body)
}
