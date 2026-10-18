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
use crate::identity::{maturity_of, MaybeCaller};

/// Candidates kept per surface/genre before the session-seeded draw.
const POOL_SIZE: usize = 50;
const MIN_VOTE_AVERAGE: f32 = 6.5;
const MIN_POPULARITY: f32 = 15.0;
/// Formats that make poor hero banners.
const EXCLUDED_GENRES: [&str; 5] = ["News", "Talk", "Soap", "Reality", "Documentary"];

#[derive(Deserialize)]
pub struct HeroQuery {
    /// Age rating limit of a local profile (U, PG, 12, 15 or 18). Used only when no verified profile is sent; a verified profile's own setting wins.
    pub rating: Option<String>,
    /// home | films | movies | series | tv
    #[serde(default = "default_surface")]
    pub surface: String,
    pub genre: Option<String>,
    pub session_id: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Local hour of the client (0 to 23): the night hours favour the late night lines
    pub hour: Option<u8>,
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

/// The six hour slot of the day the banner lines rotate with.
fn hook_slot() -> usize {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600)) as usize
}

fn hero_item_json(item: &CatalogItem, tags: Vec<String>) -> Value {
    json!({
        "tags":          tags,
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
        "rating":        crate::catalog::cert_label(item.cert),
        "rating_gb":     crate::catalog::gb_label(item.cert_gb),
        "rating_us":     crate::catalog::us_label(item.cert_us),
    })
}

/// Weighted draw (Efraimidis-Spirakis: key = u^(1/w), highest keys win) of `n` titles; without a session the order stays by quality.
fn weighted_draw<'a>(pool: Vec<&'a CatalogItem>, n: usize, session: &str) -> Vec<&'a CatalogItem> {
    if session.is_empty() {
        return pool.into_iter().take(n).collect();
    }
    let mut keyed: Vec<(f64, &CatalogItem)> = pool
        .into_iter()
        .map(|i| {
            let w = (quality_score(i) as f64).max(0.01);
            (session_uniform(session, i.id).powf(1.0 / w), i)
        })
        .collect();
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().take(n).map(|(_, i)| i).collect()
}

/// The hero titles of a pool: mostly the cinematic best (the top `POOL_SIZE` by quality), with about a quarter discoveries (well rated titles that are less
/// known, from the same pool so they still fit the page), the discoveries placed third and sixth in the carousel.
fn draw_hero<'a>(mut pool: Vec<&'a CatalogItem>, limit: usize, session: &str) -> Vec<&'a CatalogItem> {
    let by_score_desc = |a: &&CatalogItem, b: &&CatalogItem| quality_score(b).partial_cmp(&quality_score(a)).unwrap_or(std::cmp::Ordering::Equal);
    let mut rest: Vec<&CatalogItem> = Vec::new();
    if pool.len() > POOL_SIZE {
        pool.select_nth_unstable_by(POOL_SIZE, by_score_desc);
        rest = pool.split_off(POOL_SIZE);
    }
    pool.sort_by(by_score_desc);
    // discoveries: the best rated of what is left, not the most popular
    rest.retain(|i| i.vote_average >= 7.2 && i.vote_count >= 150);
    rest.sort_by(|a, b| (b.vote_average * b.popularity.ln_1p().sqrt()).partial_cmp(&(a.vote_average * a.popularity.ln_1p().sqrt())).unwrap_or(std::cmp::Ordering::Equal));
    rest.truncate(40);
    let discoveries = if session.is_empty() { 0 } else { (limit / 4).min(rest.len()) };
    let mut main = weighted_draw(pool, limit - discoveries, session);
    let found = weighted_draw(rest, discoveries, &format!("{session}|discover"));
    for (n, d) in found.into_iter().enumerate() {
        main.insert((2 + 3 * n).min(main.len()), d);
    }
    main
}

/// GET /api/hero-feed?surface=&genre=&session_id=&limit=
/// Ranked hero carousel. The top quality pool for the surface/genre is sampled
/// with a session-seeded weighted draw so each session gets a stable but
/// different rotation.
pub async fn hero_feed(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(q): Query<HeroQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(crate::identity::Maturity::from_rating));
    let surface = q.surface.to_lowercase();
    let genre = q.genre.as_deref().map(str::trim).filter(|g| !g.is_empty()).map(str::to_lowercase);
    // The client sends a genre id or name; both resolve through genre_spec for this surface
    let hero_surface = surface_media_type(&surface).unwrap_or("both");
    let genre_spec = genre.as_deref().map(|g| crate::genre_spec::resolve(g, hero_surface));
    let nonscripted_ok = genre_spec.as_ref().is_some_and(|s| s.includes_nonscripted());
    // A verified caller's session is its profile (or user) id, whatever the query says
    let session = caller.as_ref().map(|c| c.session_key()).or_else(|| q.session_id.clone()).unwrap_or_default();
    let limit = q.limit.clamp(1, 20);

    let session_state = if session.is_empty() { None } else { state.sessions.get(&session) };
    let version = session_state.as_ref().map_or(0, |s| s.lock().unwrap().version);
    let cache_key = format!("hero:{surface}:{}:{session}:{limit}:v{version}:m{}:s{}:n{}", genre.as_deref().unwrap_or(""), maturity.map_or("-", |m| m.as_str()), hook_slot(), q.hour.is_some_and(|h| h >= 22 || h <= 4));
    if let Some(hit) = state.feed_cache.get(&cache_key).await {
        if let Ok(v) = serde_json::from_str::<Value>(&hit) {
            return Json(v);
        }
    }

    let catalog = state.catalog.load();
    let media_type = surface_media_type(&surface);

    let seen = session_state.as_ref().map(|s| s.lock().unwrap().seen_ids().clone()).unwrap_or_default();

    let pool: Vec<&CatalogItem> = catalog
        .values()
        .filter(|i| {
            i.vote_average >= MIN_VOTE_AVERAGE
                && i.popularity >= MIN_POPULARITY
                && i.has_backdrop()
                && i.has_poster()
                && i.has_overview()
                && !seen.contains(&i.id)
                && (nonscripted_ok || !i.genres.iter().any(|g| EXCLUDED_GENRES.iter().any(|x| g.eq_ignore_ascii_case(x))))
                && media_type.is_none_or(|m| i.media_type == m)
                && (!crate::page::safety_nets() || i.flags & crate::catalog::FLAG_SUGGESTIVE == 0)
                && genre_spec.as_ref().is_none_or(|spec| spec.matches(i))
                && maturity.is_none_or(|m| m.allows(i))
        })
        .collect();

    // Films and series are drawn separately on the home page (series have larger popularity numbers, so one pool would fill with series) and
    // alternate in the carousel; the home draw uses its own salt so it is not the Films or Series carousel again.
    // the draw rotates every six hours for the same session
    let drawn_session = if session.is_empty() { String::new() } else { format!("{session}|{}", hook_slot()) };
    let chosen: Vec<&CatalogItem> = if media_type.is_none() {
        let salt = format!("{drawn_session}|home");
        let (movies, series): (Vec<&CatalogItem>, Vec<&CatalogItem>) = pool.into_iter().partition(|i| i.media_type == "movie");
        let (m, t) = (draw_hero(movies, limit, &salt), draw_hero(series, limit, &salt));
        let series_first = !session.is_empty() && session_uniform(&drawn_session, 7) < 0.5;
        let (mut first, mut second) = if series_first { (t.into_iter(), m.into_iter()) } else { (m.into_iter(), t.into_iter()) };
        let mut out: Vec<&CatalogItem> = Vec::with_capacity(limit);
        while out.len() < limit {
            let (a, b) = (first.next(), second.next());
            if a.is_none() && b.is_none() { break; }
            out.extend(a);
            if out.len() < limit { out.extend(b); }
        }
        out
    } else {
        draw_hero(pool, limit, &drawn_session)
    };

    let fresh = state.fresh.load();
    let env = crate::hooks::HookEnv {
        catalog: &catalog,
        vibe: &state.vibe,
        taxonomy: &state.taxonomy,
        profiles: &state.profiles,
        fresh: &fresh,
        today_md: crate::taxonomy::today_md(),
        today_days: crate::fresh::today_days(),
        late_night: q.hour.is_some_and(|h| h >= 22 || h <= 4),
        session: &session,
        slot: hook_slot(),
        kids: maturity.is_some_and(|m| m.is_kids()),
    };
    let items: Vec<Value> = chosen
        .iter()
        .map(|i| {
            let mut v = hero_item_json(i, crate::contract::item_tags(state.profiles.get(i.id), &i.genres, 3));
            v["hooks"] = Value::Array(crate::hooks::hero_hooks(i, &env));
            v
        })
        .collect();
    let body = json!({
        "surface": surface,
        "genre":   genre,
        "count":   items.len(),
        "items":   items,
    });

    state.feed_cache.insert(cache_key, body.to_string()).await;
    Json(body)
}
