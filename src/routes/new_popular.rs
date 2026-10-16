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
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const ROW_TITLES: usize = 15;
const MIN_ROW_ITEMS: usize = 6;
const DESKTOP_ROWS: usize = 14;

#[derive(Deserialize)]
pub struct NewPopularQuery {
    pub rating: Option<String>,
    /// watching | top10_movies | top10_series | just_landed | coming_soon (mobile New & Hot tabs); none gives the desktop page
    pub tab: Option<String>,
    pub limit: Option<usize>,
    /// The visitor's session: with the six hour slot it decides the order of the rows and which languages and genres get one
    pub session_id: Option<String>,
}

/// One title as the page sees it, from the catalog, from fresh.json, or both (the catalog supplies certificates and genres, fresh.json the
/// current popularity and the release date).
#[derive(Clone)]
struct Cand {
    media_type: String,
    tmdb_id: u32,
    title: String,
    overview: String,
    poster_path: Option<String>,
    backdrop_path: Option<String>,
    vote_average: f32,
    vote_count: u32,
    popularity: f32,
    /// YYYY-MM-DD or empty
    release_date: String,
    release_year: Option<i32>,
    genres: Vec<String>,
    language: String,
    tv_type: Option<String>,
    cert: u8,
    cert_gb: u8,
    cert_us: u8,
    in_catalog: bool,
    allowed: bool,
    lists: Vec<String>,
    ranks: HashMap<String, usize>,
}

impl Cand {
    fn key(&self) -> (String, u32) {
        (self.media_type.clone(), self.tmdb_id)
    }
    fn released(&self, today: &str) -> bool {
        !self.release_date.is_empty() && self.release_date.as_str() <= today
    }
    fn upcoming(&self, today: &str) -> bool {
        !self.release_date.is_empty() && self.release_date.as_str() > today
    }
    fn days_since(&self, today: i64) -> Option<i64> {
        parse_days(&self.release_date).map(|d| today - d)
    }
    fn in_list(&self, tag: &str) -> bool {
        self.lists.iter().any(|l| l == tag)
    }
    fn rank_in(&self, tag: &str) -> usize {
        self.ranks.get(tag).copied().unwrap_or(usize::MAX)
    }
}

fn from_catalog(item: &CatalogItem, allowed: bool) -> Cand {
    Cand {
        media_type: item.media_type.clone(), tmdb_id: item.tmdb_id, title: item.title.clone(), overview: item.overview.clone(),
        poster_path: item.poster_path.clone(), backdrop_path: item.backdrop_path.clone(), vote_average: item.vote_average, vote_count: item.vote_count,
        popularity: item.popularity, release_date: String::new(), release_year: item.release_year, genres: item.genres.clone(),
        language: item.original_language.clone(), tv_type: item.tv_type.clone(), cert: item.cert, cert_gb: item.cert_gb, cert_us: item.cert_us,
        in_catalog: true, allowed, lists: Vec::new(), ranks: HashMap::new(),
    }
}

fn from_fresh(f: &FreshItem, catalog_item: Option<&CatalogItem>, allowed: bool) -> Cand {
    let mut c = match catalog_item {
        Some(item) => from_catalog(item, allowed),
        None => Cand {
            media_type: f.media_type.clone(), tmdb_id: f.id, title: f.title.clone(), overview: f.overview.clone(), poster_path: f.poster_path.clone(),
            backdrop_path: f.backdrop_path.clone(), vote_average: f.vote_average, vote_count: f.vote_count, popularity: f.popularity,
            release_date: String::new(), release_year: f.release_date.get(..4).and_then(|y| y.parse().ok()), genres: f.genres.clone(),
            language: f.original_language.clone().unwrap_or_default(), tv_type: None, cert: 0, cert_gb: 0, cert_us: 0, in_catalog: false, allowed,
            lists: Vec::new(), ranks: HashMap::new(),
        },
    };
    c.popularity = f.popularity.max(if catalog_item.is_some() { 0.0 } else { f.popularity });
    c.release_date = f.release_date.clone();
    c.release_year = f.release_date.get(..4).and_then(|y| y.parse().ok()).or(c.release_year);
    if c.backdrop_path.is_none() {
        c.backdrop_path = f.backdrop_path.clone();
    }
    c.lists = f.lists.clone();
    c.ranks = f.ranks.clone();
    c
}

/// Whether a profile may see a title. A catalog title follows the same rules as every other page; a title only fresh.json knows has no
/// certificate, so only an unrestricted profile sees it.
fn catalog_allowed(item: &CatalogItem, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore) -> bool {
    if item.adult || (crate::page::safety_nets() && item.flags & crate::catalog::FLAG_SUGGESTIVE != 0) {
        return false;
    }
    maturity.is_none_or(|m| crate::page::maturity_allows_with(m, profiles, item))
}

fn fresh_only_allowed(f: &FreshItem, maturity: Option<Maturity>) -> bool {
    !f.adult && maturity.is_none_or(|m| m == Maturity::Adult)
}

/// All candidates: every fresh title, plus the catalog's own newest titles (release year this year or last) for rows that need no date.
fn candidates(catalog: &CatalogMap, fresh: &FreshStore, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, year: i32) -> Vec<Cand> {
    let mut out: Vec<Cand> = Vec::new();
    let mut seen: HashSet<(String, u32)> = HashSet::new();
    for f in &fresh.items {
        let cat = resolve_key(catalog, f.id, Some(&f.media_type)).and_then(|k| catalog.get(&k));
        let allowed = match cat {
            Some(item) => catalog_allowed(item, maturity, profiles),
            None => fresh_only_allowed(f, maturity),
        };
        let c = from_fresh(f, cat, allowed);
        seen.insert(c.key());
        out.push(c);
    }
    for item in catalog.values() {
        if item.release_year.is_some_and(|y| y >= year - 1 && y <= year) && crate::page::passes_member_quality(item) && seen.insert((item.media_type.clone(), item.tmdb_id)) {
            out.push(from_catalog(item, catalog_allowed(item, maturity, profiles)));
        }
    }
    out
}

fn has_art(c: &Cand) -> bool {
    c.poster_path.as_deref().is_some_and(|p| !p.is_empty())
}

fn has_backdrop(c: &Cand) -> bool {
    has_art(c) && c.backdrop_path.as_deref().is_some_and(|p| !p.is_empty())
}

fn by_popularity(a: &&Cand, b: &&Cand) -> std::cmp::Ordering {
    b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.tmdb_id.cmp(&b.tmdb_id)).then_with(|| a.media_type.cmp(&b.media_type))
}

struct RowOut {
    title: String,
    /// Translation key of the title (`row.np_new_movies`) and its arguments; `title` is the English fallback
    key: &'static str,
    args: Value,
    row_type: &'static str,
    kind: &'static str,
    items: Vec<Cand>,
}

fn item_json(c: &Cand, rank: Option<usize>, today: i64) -> Value {
    json!({
        "id": c.tmdb_id, "media_type": c.media_type, "title": c.title,
        "release_year": c.release_year.map(|y| y.to_string()),
        "release_date": if c.release_date.is_empty() { Value::Null } else { json!(c.release_date) },
        "days_until": parse_days(&c.release_date).map(|d| d - today),
        "genres": c.genres.join(", "), "overview": c.overview, "poster_path": c.poster_path, "backdrop_path": c.backdrop_path,
        "vote_average": c.vote_average, "tv_type": c.tv_type, "language": c.language,
        "rating": crate::catalog::cert_label(c.cert), "rating_gb": crate::catalog::gb_label(c.cert_gb), "rating_us": crate::catalog::us_label(c.cert_us),
        "rank": rank,
        "badges": crate::contract::item_badges(rank, (!c.release_date.is_empty()).then_some(c.release_date.as_str()), today),
    })
}

/// The Top 10: the most popular released titles of one type, by popularity refreshed daily.
fn chart(catalog: &CatalogMap, fresh: &FreshStore, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, media_type: &str, today: &str, year: i32) -> Vec<Cand> {
    chart_where(catalog, fresh, maturity, profiles, today, year, |i| i.media_type == media_type)
}

/// The Top 10 of the titles that pass `keep`: a language, a genre, a type.
fn chart_where(catalog: &CatalogMap, fresh: &FreshStore, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, today: &str, year: i32, keep: impl Fn(&CatalogItem) -> bool) -> Vec<Cand> {
    let mut items: Vec<&CatalogItem> = catalog
        .values()
        .filter(|i| keep(i) && crate::page::chart_ok(i) && catalog_allowed(i, maturity, profiles))
        .filter(|i| i.release_year.is_none_or(|y| y <= year) && !fresh.is_upcoming(&i.media_type, i.tmdb_id, today))
        .collect();
    items.sort_unstable_by(|a, b| b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
    items.truncate(10);
    items
        .into_iter()
        .map(|i| {
            let mut c = from_catalog(i, true);
            if let Some(f) = fresh.get(&i.media_type, i.tmdb_id) {
                c.release_date = f.release_date.clone();
            }
            c
        })
        .collect()
}

fn take_top(mut v: Vec<&Cand>, n: usize) -> Vec<Cand> {
    v.sort_by(by_popularity);
    v.truncate(n);
    v.into_iter().cloned().collect()
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
    let limit = q.limit.unwrap_or(if tab.is_some() { 10 } else { ROW_TITLES }).clamp(1, 30);

    let slot = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600)) as usize;
    // the tabs of the phone are the same for everyone; the desktop page differs by session and six hour slot
    let seed_part = if tab.is_some() { "-".to_string() } else { format!("{}:{}", q.session_id.as_deref().unwrap_or("-"), slot) };
    let cache_key = format!("newpop:{}:{}:{}:{}:{}:{}", tab.unwrap_or("-"), limit, maturity.map_or("-", |m| m.as_str()), fresh.generated, today, seed_part);
    if let Some(hit) = state.feed_cache.get(&cache_key).await {
        if let Ok(v) = serde_json::from_str::<Value>(&hit) {
            return Json(v);
        }
    }

    let all = candidates(&catalog, &fresh, maturity, &state.profiles, year);
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
            let mut seen: HashSet<(String, u32)> = HashSet::new();
            let mut out: Vec<Value> = Vec::new();
            for (i, row) in rows.into_iter().enumerate() {
                // a title appears once on the page, in the first row that has it (the two Top 10 rows may share nothing anyway)
                let items: Vec<Cand> = row.items.into_iter().filter(|c| seen.insert(c.key())).take(limit).collect();
                if items.len() < MIN_ROW_ITEMS.min(10) && row.row_type != "top_ten" {
                    continue;
                }
                if items.is_empty() {
                    continue;
                }
                let ranked = row.row_type == "top_ten";
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

#[allow(clippy::too_many_arguments)]
fn desktop_rows(catalog: &CatalogMap, fresh: &FreshStore, pool: &[&Cand], maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, today: &str, today_n: i64, year: i32, limit: usize, session: &str, slot: usize) -> Vec<RowOut> {
    let mut rows: Vec<RowOut> = Vec::new();
    // rows are cut to `limit` after titles shared with an earlier row are removed, so each starts from a wider list
    let wide = limit * 4;
    let recent = |c: &&Cand, days: i64| c.released(today) && c.days_since(today_n).is_some_and(|d| d <= days);

    // the charts are built here and placed among the other rows at the end: how many there are, which, and where changes with the session and the slot
    let mut charts: Vec<RowOut> = vec![
        RowOut { title: "Top 10 Movies Today".into(), key: "row.np_top10_movies", args: Value::Null, row_type: "top_ten", kind: "chart", items: chart(catalog, fresh, maturity, profiles, "movie", today, year) },
        RowOut { title: "Top 10 Series Today".into(), key: "row.np_top10_series", args: Value::Null, row_type: "top_ten", kind: "chart", items: chart(catalog, fresh, maturity, profiles, "tv", today, year) },
    ];

    // what is being watched today, in TMDB's own order
    let mut trending: Vec<&Cand> = pool.iter().copied().filter(|c| c.released(today) && c.in_list("trending_day")).collect();
    trending.sort_by_key(|c| c.rank_in("trending_day"));
    rows.push(RowOut { title: "Trending Today".into(), key: "row.np_trending_today", args: Value::Null, row_type: "vibe", kind: "trending", items: trending.into_iter().take(limit).cloned().collect() });

    let new_movies: Vec<&Cand> = pool.iter().copied().filter(|c| c.media_type == "movie" && recent(c, 45)).collect();
    rows.push(RowOut { title: "New Movies".into(), key: "row.np_new_movies", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(new_movies, wide) });

    let episodes: Vec<&Cand> = pool.iter().copied().filter(|c| c.media_type == "tv" && (c.in_list("airing_today") || c.in_list("on_the_air")) && c.vote_average >= 6.0 && !matches!(c.tv_type.as_deref(), Some("Talk Show") | Some("News") | Some("Soap"))).collect();
    rows.push(RowOut { title: "New Episodes This Week".into(), key: "row.np_new_episodes", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(episodes, wide) });

    let mut soon_movies: Vec<Cand> = take_top(pool.iter().copied().filter(|c| c.media_type == "movie" && c.upcoming(today) && c.days_since(today_n).is_some_and(|d| d >= -120)).collect(), wide);
    soon_movies.sort_by(|a, b| a.release_date.cmp(&b.release_date));
    rows.push(RowOut { title: "Coming Soon to Cinemas".into(), key: "row.np_coming_soon_cinemas", args: Value::Null, row_type: "vibe", kind: "soon", items: soon_movies });

    let acclaimed: Vec<&Cand> = pool.iter().copied().filter(|c| recent(c, 270) && c.vote_count >= 100 && c.vote_average >= 7.2).collect();
    rows.push(RowOut { title: "New and Acclaimed".into(), key: "row.np_new_acclaimed", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(acclaimed, wide) });

    let mut soon_tv: Vec<Cand> = take_top(pool.iter().copied().filter(|c| c.media_type == "tv" && c.upcoming(today) && c.days_since(today_n).is_some_and(|d| d >= -150)).collect(), wide);
    soon_tv.sort_by(|a, b| a.release_date.cmp(&b.release_date));
    rows.push(RowOut { title: "Coming Soon: Series".into(), key: "row.np_coming_soon_series", args: Value::Null, row_type: "vibe", kind: "soon", items: soon_tv });

    let new_tv: Vec<&Cand> = pool.iter().copied().filter(|c| c.media_type == "tv" && recent(c, 90)).collect();
    rows.push(RowOut { title: "New Series".into(), key: "row.np_new_series", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(new_tv, wide) });

    // languages with enough new and trending titles each get a row, the biggest first
    let fresh_released: Vec<&Cand> = pool.iter().copied().filter(|c| recent(c, 120)).collect();
    let mut by_lang: HashMap<&str, usize> = HashMap::new();
    for c in &fresh_released {
        if c.language != "en" && crate::row_label::language_label(&c.language).is_some() {
            *by_lang.entry(c.language.as_str()).or_insert(0) += 1;
        }
    }
    let mut langs: Vec<(&str, usize)> = by_lang.into_iter().filter(|(_, n)| *n >= MIN_ROW_ITEMS).collect();
    langs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    // the biggest language also gets a Top 10 of its own among the charts
    if let Some((code, _)) = langs.first() {
        let (code, label) = (*code, crate::row_label::language_label(code).unwrap_or(code));
        charts.push(RowOut { title: format!("Top 10 {label} Titles Today"), key: "row.np_top10_language", args: json!({ "language": label }), row_type: "top_ten", kind: "chart", items: chart_where(catalog, fresh, maturity, profiles, today, year, |i| i.original_language == code) });
    }
    // the four biggest languages compete, two get a row: which two changes with the session and the slot
    for (code, _) in pick_by_noise(langs, 4, 2, session, slot) {
        let v: Vec<&Cand> = fresh_released.iter().copied().filter(|c| c.language == code).collect();
        rows.push(RowOut { title: format!("New in {}", crate::row_label::language_label(code).unwrap_or(code)), key: "row.np_new_in_language", args: json!({ "language": crate::row_label::language_label(code).unwrap_or(code) }), row_type: "vibe", kind: "language", items: take_top(v, wide) });
    }

    // genres with the most new titles
    let mut by_genre: HashMap<&str, usize> = HashMap::new();
    for c in pool.iter().filter(|c| recent(c, 90)) {
        for g in &c.genres {
            if crate::row_label::genre_noun(g).is_some() && g != "Drama" {
                *by_genre.entry(g.as_str()).or_insert(0) += 1;
            }
        }
    }
    let mut genres: Vec<(&str, usize)> = by_genre.into_iter().filter(|(_, n)| *n >= MIN_ROW_ITEMS + 4).collect();
    genres.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    if let Some((g, _)) = genres.first() {
        let (g, noun) = (*g, crate::row_label::genre_noun(g).unwrap_or(g));
        charts.push(RowOut { title: format!("Top 10 {noun} Today"), key: "row.np_top10_genre", args: json!({ "genre": noun }), row_type: "top_ten", kind: "chart", items: chart_where(catalog, fresh, maturity, profiles, today, year, |i| i.genres.iter().any(|x| x == g)) });
    }
    for (g, _) in pick_by_noise(genres, 5, 3, session, slot) {
        let v: Vec<&Cand> = pool.iter().copied().filter(|c| recent(c, 90) && c.genres.iter().any(|x| x == g)).collect();
        let noun = crate::row_label::genre_noun(g).unwrap_or(g);
        rows.push(RowOut { title: format!("New {noun}"), key: "row.np_new_genre", args: json!({ "genre": noun }), row_type: "vibe", kind: "genre", items: take_top(v, wide) });
    }

    // good titles of the last three years that are not yet everywhere: rated well by enough people, not among the most popular. Chosen by how well they are rated and how many rated them (not by
    // popularity, which would bring the same few every time), with a little noise per session and slot, at most three of a genre and films and series alternating
    let mut gems: Vec<(f32, &Cand)> = pool
        .iter()
        .copied()
        .filter(|c| c.in_catalog && !c.upcoming(today) && c.release_year.is_some_and(|y| y >= year - 2) && c.vote_average >= 7.2 && c.vote_count >= 80 && c.popularity < 60.0 && c.popularity >= 5.0)
        .map(|c| (c.vote_average + 0.35 * (c.vote_count as f32).ln() + 1.6 * crate::page::entry_noise(session, c.tmdb_id as usize + slot), c))
        .collect();
    gems.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.tmdb_id.cmp(&b.1.tmdb_id)));
    let mut per_genre: HashMap<&str, usize> = HashMap::new();
    let (mut films, mut series): (Vec<&Cand>, Vec<&Cand>) = (Vec::new(), Vec::new());
    for (_, c) in gems {
        let g = c.genres.first().map_or("", |g| g.as_str());
        let n = per_genre.entry(g).or_insert(0);
        if *n >= 3 { continue; }
        *n += 1;
        if c.media_type == "tv" { series.push(c) } else { films.push(c) }
    }
    let mut mixed: Vec<Cand> = Vec::new();
    let (mut fi, mut si) = (films.into_iter(), series.into_iter());
    while mixed.len() < wide {
        match (fi.next(), si.next()) {
            (None, None) => break,
            (f, t) => { mixed.extend(f.cloned()); mixed.extend(t.cloned()); }
        }
    }
    let (gem_title, gem_key) = [("Under the Radar", "row.np_under_the_radar"), ("Hidden Gems", "row.np_hidden_gems"), ("Worth a Closer Look", "row.np_closer_look")][slot % 3];
    rows.push(RowOut { title: gem_title.into(), key: gem_key, args: Value::Null, row_type: "vibe", kind: "gems", items: mixed });

    // the order of the other rows moves a little with the session and the slot
    let mut keyed: Vec<(f32, RowOut)> = rows.into_iter().enumerate().map(|(i, r)| (i as f32 + (crate::page::entry_noise(session, i + slot * 31) - 0.5) * 6.0, r)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ordered: Vec<RowOut> = keyed.into_iter().map(|(_, r)| r).collect();

    // between one and four charts (the nearest to the news first), spread out: the first after zero to two other rows, then every three to five rows, never next to each other
    charts.retain(|c| !c.items.is_empty());
    let want = 1 + (crate::page::entry_noise(session, 7 + slot) * 4.0) as usize;
    let mut chosen: Vec<(f32, RowOut)> = charts.into_iter().enumerate().map(|(i, c)| (i as f32 * 0.6 + crate::page::entry_noise(session, 100 + i + slot * 13) * 2.0, c)).collect();
    chosen.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let start = (crate::page::entry_noise(session, 11 + slot) * 3.0) as usize;
    let gap = 3 + (crate::page::entry_noise(session, 13 + slot) * 3.0) as usize;
    for (n, (_, chart_row)) in chosen.into_iter().take(want).enumerate() {
        let at = (start + n * gap).min(ordered.len());
        ordered.insert(at, chart_row);
    }
    ordered
}

/// `take` of the first `pool` candidates, chosen by their place plus noise from the session and the slot.
fn pick_by_noise<T>(mut v: Vec<(T, usize)>, pool: usize, take: usize, session: &str, slot: usize) -> Vec<(T, usize)> {
    v.truncate(pool);
    let mut keyed: Vec<(f32, (T, usize))> = v.into_iter().enumerate().map(|(i, x)| (i as f32 + crate::page::entry_noise(session, i * 17 + slot) * 2.5, x)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().take(take).map(|(_, x)| x).collect()
}

