use crate::app::AppState;
use crate::identity::{maturity_of, MaybeCaller};
use crate::page::{PageContext, Row};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;

/// Query parameters for GET /api/feed
#[derive(Deserialize, Debug)]
pub struct FeedQuery {
    /// Age rating limit of a local profile (U, PG, 12, 15 or 18). Used only when no verified profile is sent; a verified profile's own setting wins.
    pub rating: Option<String>,
    /// "movie", "tv", "both", or the client surfaces "home", "films", "series"
    pub surface: Option<String>,
    /// Sparse taste vector: "28:0.45,80:0.31,878:0.15" (genre_id:weight pairs)
    pub taste: Option<String>,
    /// Client alias for `taste`. Also accepts genre names: "Action:0.6,Drama:0.4" or "Action,Drama"
    pub affinity: Option<String>,
    /// Genre name filter applied to every row
    pub genre: Option<String>,
    /// The collection a page is the page of (the id of a mood, such as `tearjerkers`): the page opens with its header and its own rows (the page Explore all opens)
    pub collection: Option<String>,
    /// Original language code of a language page (ko, ja, es ...); every row holds only titles in that language
    pub language: Option<String>,
    /// Client session id (reserved for session-aware feeds)
    pub session_id: Option<String>,
    /// Number of rows to generate (default: 20)
    pub rows: Option<usize>,
    /// Client alias for `rows`
    pub limit: Option<usize>,
    /// Rows to skip, for infinite scroll (default 0). The page is built once up to
    /// offset + rows and cached, so later pages continue it without repeats.
    pub offset: Option<usize>,
    /// Local hour of the client (0 to 23). From 22 to 4 the page leads with late night moods.
    pub hour: Option<u8>,
    /// The profile's signals (see signals.rs): saved, recent (most recent first) and liked titles as `movie:550,tv:1399`
    pub saved: Option<String>,
    pub recent: Option<String>,
    pub liked: Option<String>,
}

/// Row policy derived in ai-engine/config/row_architecture.json (surfaces block).
/// Initial rows cover about three desktop screens; the cap is Netflix's ~40 rows per
/// page, reduced for genres whose quality pool cannot fill that many rows without
/// repeating titles across two sessions (pool / (15 titles x 2)).
const NETFLIX_ROWS_CAP: usize = 60;
const INITIAL_ROWS: usize = 12;
const INITIAL_ROWS_GENRE: usize = 10;
const MIN_ROWS_FLOOR: usize = 8; // two desktop screens
const ROW_TITLES: usize = 15;
const SESSION_TURNOVER: usize = 2;
/// Below this a genre has too few rows to be its own page.
pub const MIN_GENRE_PAGE_ROWS: usize = 8;

/// Rows a genre page can support for this surface, from the size of its quality pool.
pub fn genre_row_cap(catalog: &crate::catalog::CatalogMap, profiles: &crate::profiles::ProfileStore, surface: &str, spec: &crate::genre_spec::GenreSpec, maturity: Option<crate::identity::Maturity>) -> usize {
    let pool = catalog
        .values()
        .filter(|i| {
            (surface == "both" || i.media_type == surface)
                && crate::page::genre_ok_with(Some(spec), profiles, i)
                && crate::page::audience_fits_with(Some(spec), maturity, profiles, i)
                && maturity.is_none_or(|m| crate::page::maturity_allows_with(m, profiles, i))
                && crate::page::member_quality_for(Some(spec), i)
                && crate::page::meets_vote_floor(i, true)
        })
        .count();
    // a judged niche page is small by nature: its titles may repeat across two visits
    let per_row = if spec.is_judged() { ROW_TITLES } else { ROW_TITLES * SESSION_TURNOVER };
    (pool / per_row).min(NETFLIX_ROWS_CAP)
}

/// Rows a language page can support: the quality titles in that language the profile may see, 30 per row.
pub fn language_row_cap(catalog: &crate::catalog::CatalogMap, profiles: &crate::profiles::ProfileStore, surface: &str, language: &str, maturity: Option<crate::identity::Maturity>) -> usize {
    let pool = catalog
        .values()
        .filter(|i| {
            (surface == "both" || i.media_type == surface)
                && crate::page::language_matches(language, &i.original_language)
                && crate::page::audience_fits_with(None, maturity, profiles, i)
                && maturity.is_none_or(|m| crate::page::maturity_allows_with(m, profiles, i))
                && crate::page::passes_member_quality(i)
                && crate::page::meets_vote_floor(i, true)
        })
        .count();
    (pool / (ROW_TITLES * SESSION_TURNOVER)).min(NETFLIX_ROWS_CAP)
}

/// The pool size only changes when the catalog reloads, so it is cached per surface, genre rule and catalog size.
fn genre_row_cap_cached(catalog: &crate::catalog::CatalogMap, profiles: &crate::profiles::ProfileStore, surface: &str, spec: &crate::genre_spec::GenreSpec, maturity: Option<crate::identity::Maturity>) -> usize {
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};
    static CAPS: OnceLock<Mutex<std::collections::HashMap<String, (usize, Instant)>>> = OnceLock::new();
    let key = format!("{surface}|{spec:?}|{}|{}|{}", catalog.len(), profiles.profiles.len(), maturity.map_or("-", |m| m.as_str()));
    let map = CAPS.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some((v, at)) = map.lock().unwrap().get(&key) {
        if at.elapsed() < Duration::from_secs(600) {
            return *v;
        }
    }
    let v = genre_row_cap(catalog, profiles, surface, spec, maturity);
    map.lock().unwrap().insert(key, (v, Instant::now()));
    v
}

fn language_row_cap_cached(catalog: &crate::catalog::CatalogMap, profiles: &crate::profiles::ProfileStore, surface: &str, language: &str, maturity: Option<crate::identity::Maturity>) -> usize {
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};
    static CAPS: OnceLock<Mutex<std::collections::HashMap<String, (usize, Instant)>>> = OnceLock::new();
    let key = format!("{surface}|{language}|{}|{}|{}", catalog.len(), profiles.profiles.len(), maturity.map_or("-", |m| m.as_str()));
    let map = CAPS.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some((v, at)) = map.lock().unwrap().get(&key) {
        if at.elapsed() < Duration::from_secs(600) {
            return *v;
        }
    }
    let v = language_row_cap(catalog, profiles, surface, language, maturity);
    map.lock().unwrap().insert(key, (v, Instant::now()));
    v
}

fn normalize_surface(s: &str) -> &'static str {
    match s {
        "movie" | "movies" | "films" | "film" => "movie",
        "tv" | "series" | "shows" => "tv",
        _ => "both",
    }
}

/// Cache key: normalise taste weights to 1 decimal, sort by genre_id, cap at 5.
/// This ensures "action:0.80" and "action:0.8001" share the same key.
fn cache_key(surface: &str, genre: &str, rows: usize, taste: &str) -> String {
    let mut pairs: Vec<(String, String)> = taste
        .split(',')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, ':');
            let genre = parts.next()?.trim().to_string();
            let weight: f32 = parts.next().and_then(|w| w.trim().parse().ok()).unwrap_or(1.0);
            let rounded = format!("{:.1}", weight);
            Some((genre, rounded))
        })
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs.truncate(5);
    let taste_bucket = pairs
        .iter()
        .map(|(g, w)| format!("{g}:{w}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{surface}|{genre}|{rows}|{taste_bucket}")
}

/// Parse the taste query parameter into a genre affinity list.
/// Input: "28:0.45,80:0.31" → [("Action", 0.45), ("Drama", 0.31)]
/// Genre IDs are mapped to names using the TMDB genre taxonomy.
fn parse_taste(taste_str: &str) -> Vec<(String, f32)> {
    // TMDB genre ID → name map (subset)
    let genre_map: &[(&str, &str)] = &[
        ("28", "Action"), ("12", "Adventure"), ("16", "Animation"),
        ("35", "Comedy"), ("80", "Crime"), ("99", "Documentary"),
        ("18", "Drama"), ("10751", "Family"), ("14", "Fantasy"),
        ("36", "History"), ("27", "Horror"), ("10402", "Music"),
        ("9648", "Mystery"), ("10749", "Romance"), ("878", "Science Fiction"),
        ("53", "Thriller"), ("10752", "War"), ("37", "Western"),
    ];

    taste_str
        .split(',')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, ':');
            let genre_id = parts.next()?.trim();
            if genre_id.is_empty() {
                return None;
            }
            let weight: f32 = parts.next().and_then(|w| w.trim().parse().ok()).unwrap_or(1.0);

            let name = genre_map
                .iter()
                .find(|(id, _)| *id == genre_id)
                .map(|(_, name)| name.to_string())
                .unwrap_or_else(|| genre_id.to_string());

            Some((name, weight.clamp(0.0, 1.0)))
        })
        .take(5)
        .collect()
}

/// GET /api/feed
/// Main discovery feed endpoint. Returns a page of thematic recommendation rows.
pub async fn feed(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(params): Query<FeedQuery>,
) -> impl IntoResponse {
    // A verified caller's session is its profile (or user) id, whatever the query says
    let session_id: Option<String> = caller.as_ref().map(|c| c.session_key()).or_else(|| params.session_id.clone());
    let maturity = maturity_of(&caller).or_else(|| params.rating.as_deref().and_then(crate::identity::Maturity::from_rating));
    let surface = normalize_surface(params.surface.as_deref().unwrap_or("both"));
    let taste_str = params.taste.as_deref().or(params.affinity.as_deref()).unwrap_or("");
    let genre = params.genre.as_deref().map(str::trim).filter(|g| !g.is_empty());
    let language = params.language.as_deref().map(str::trim).filter(|l| !l.is_empty() && l.len() <= 3 && l.chars().all(|c| c.is_ascii_alphabetic())).map(str::to_lowercase);
    let initial = if genre.is_some() || language.is_some() { INITIAL_ROWS_GENRE } else { INITIAL_ROWS };
    let per_page = params.rows.or(params.limit).unwrap_or(initial).clamp(MIN_ROWS_FLOOR, NETFLIX_ROWS_CAP);
    let offset = params.offset.unwrap_or(0).min(NETFLIX_ROWS_CAP);

    // Session memory: fall back to the session's learned taste when the client
    // sends none, and drop titles the session has already started.
    let session_state = session_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .and_then(|sid| state.sessions.get(sid));
    let (session_taste, session_seen, session_version) = match &session_state {
        Some(s) => {
            let s = s.lock().unwrap();
            (s.taste(), s.seen_ids().clone(), s.version)
        }
        None => (Vec::new(), Default::default(), 0),
    };

    let mut key = cache_key(surface, &genre.unwrap_or("").to_lowercase(), offset + per_page, taste_str);
    // Entry driven pages explore per session, so the session is part of the key
    // the choice of rows rotates every six hours for the same session, so a returning visitor does not meet the same page all week
    let slot = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600));
    let noise_session = format!("{}|{slot}", session_id.as_deref().unwrap_or(""));
    key = format!("{key}|t:{slot}|s:{}", session_id.as_deref().unwrap_or(""));
    if session_version > 0 {
        key = format!("{key}|{}|v{session_version}", session_id.as_deref().unwrap_or(""));
    }
    let late_night = params.hour.is_some_and(|h| h >= 22 || h <= 4);
    key = format!("{key}|n:{}", late_night as u8);
    let collection = params.collection.as_deref().map(str::trim).filter(|c| !c.is_empty() && c.len() <= 48 && c.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'));
    if let Some(c) = collection { key = format!("{key}|c:{c}"); }
    let signals = crate::signals::Signals::parse(params.saved.as_deref(), params.recent.as_deref(), params.liked.as_deref());
    if !signals.is_empty() {
        key = format!("{key}|g:{}", signals.fingerprint());
    }
    // pages differ per maturity
    key = format!("{key}|m:{}|l:{}", maturity.map_or("-", |m| m.as_str()), language.as_deref().unwrap_or(""));

    // Check cache first
    if let Some(cached) = state.feed_cache.get(&key).await {
        return (StatusCode::OK, Json(serde_json::from_str::<serde_json::Value>(&cached).unwrap()));
    }

    let catalog = state.catalog.load();
    let fresh = state.fresh.load();
    let genre_spec = genre.map(|g| crate::genre_spec::resolve(g, surface));
    // the rows the page's own titles can fill; the page is topped up with neighbouring rows to the floor, but the page is only offered when its own cap is enough
    let own_cap = match (&genre_spec, language.as_deref()) {
        (Some(spec), None) => genre_row_cap_cached(&catalog, &state.profiles, surface, spec, maturity),
        _ => usize::MAX,
    };
    let cap = match (&genre_spec, language.as_deref()) {
        (Some(_), None) => own_cap.max(crate::page::THIN_PAGE_FLOOR),
        (_, Some(lang)) => language_row_cap_cached(&catalog, &state.profiles, surface, lang, maturity),
        (None, None) => NETFLIX_ROWS_CAP,
    };
    let target_rows = (offset + per_page).min(cap.max(MIN_ROWS_FLOOR)).max(MIN_ROWS_FLOOR);

    // Build taste query vector from client-provided genre affinities
    let mut taste_affinities = parse_taste(taste_str);
    if taste_affinities.is_empty() {
        taste_affinities = session_taste;
    }
    let taste_vec = if taste_affinities.is_empty() {
        // Cold start: no taste signal → use global average (zero vector → centroid blend fails gracefully)
        vec![0.0f32; crate::vectors::DIMS]
    } else {
        state.vault.blend_taste(&taste_affinities)
    };

    // A Kids profile's home comes from the Kids row vocabulary when there is one
    let kids_entries = maturity.is_some_and(|m| m.is_kids()) && genre.is_none() && language.is_none() && state.taxonomy_kids.entries.len() >= 60;
    // Assemble page
    let ctx = PageContext {
        catalog:    &catalog,
        vectors:    &state.vectors,
        hnsw:       &state.hnsw,
        seeds:      &state.seeds,
        groups:     &state.groups,
        group_names: &state.group_names,
        vibe:       &state.vibe,
        late_night,
        vault:      &state.vault,
        taxonomy:   if kids_entries { &state.taxonomy_kids } else { &state.taxonomy },
        profiles:   &state.profiles,
        topics:     &state.topics,
        fresh:      &fresh,
        today_ymd:  crate::fresh::ymd(crate::fresh::today_days()),
        language:   language.as_deref(),
        animation:  None,
        kids_entries,
        surface,
        genre,
        genre_spec: genre_spec.as_ref(),
        taste:      taste_vec,
        target_rows,
        mean:       &state.mean,
        today:      crate::taxonomy::today_md(),
        session:    if session_id.is_some() { noise_session.as_str() } else { "" },
        maturity,
        fill_pool:  None,
        skip:       None,
        signals:    &signals,
    };

    let (page_rows, collection_header) = match collection.and_then(|c| ctx.build_mood_collection(c)) {
        Some((rows, header)) => (rows, Some(header)),
        None => (ctx.build_page().await, None),
    };
    let all_rows = to_client_rows(page_rows, &session_seen, &state.profiles, &fresh);
    let has_more = all_rows.len() > offset + per_page && offset + per_page < cap;
    let rows: Vec<serde_json::Value> = all_rows.into_iter().skip(offset).take(per_page).collect();
    let response = json!({
        "surface":     surface,
        "genre":       genre,
        "row_cap":     cap,
        "genre_page_supported": (genre.is_none() && language.is_none()) || (if genre_spec.is_some() && language.is_none() { own_cap } else { cap }) >= MIN_GENRE_PAGE_ROWS,
        "language":    language,
        "total_rows":  rows.len(),
        "offset":      offset,
        "next_offset": offset + rows.len(),
        "has_more":    has_more,
        "header":      collection_header.or_else(|| genre_spec.as_ref().and_then(|g| g.event_id()).and_then(|ev| crate::personal::event_header(&state.vibe, &catalog, ev))),
        "chips":       if genre.is_none() && language.is_none() { ctx.season_chips() } else { Vec::new() },
        "rows":        rows,
    });
    let response_str = serde_json::to_string(&response).unwrap();

    // Cache the result
    state.feed_cache.insert(key, response_str).await;

    (StatusCode::OK, Json(response))
}

/// Convert engine rows to the client FeedRow shape. Titles are deduplicated
/// across rows (earlier rows win) and rows left with too few items are dropped.
fn to_client_rows(rows: Vec<Row>, already_seen: &HashSet<u32>, profiles: &crate::profiles::ProfileStore, fresh: &crate::fresh::FreshStore) -> Vec<serde_json::Value> {
    const MIN_ROW_ITEMS: usize = 4;
    let mut seen: HashSet<u32> = HashSet::new();
    let mut out = Vec::new();
    let today = crate::fresh::today_days();

    for (i, row) in rows.into_iter().enumerate() {
        let chart = row.archetype == "trending";
        let items: Vec<serde_json::Value> = row
            .items
            .iter()
            .filter(|it| !already_seen.contains(&it.id) && seen.insert(it.id))
            .enumerate()
            .map(|(n, it)| {
                let release = fresh.get(&it.media_type, it.tmdb_id).map(|f| f.release_date.as_str());
                json!({
                    "id":            it.tmdb_id,
                    "media_type":    it.media_type,
                    "title":         it.title,
                    "release_year":  it.release_year.map(|y| y.to_string()),
                    "genres":        it.genres.join(", "),
                    "overview":      it.overview,
                    "poster_path":   it.poster_path,
                    "backdrop_path": it.backdrop_path,
                    "vote_average":  it.vote_average,
                    "tv_type":       it.tv_type,
                    "language":      it.language,
                    "rating":        crate::catalog::cert_label(it.cert),
                    "rating_gb":     crate::catalog::gb_label(it.cert_gb),
                    "rating_us":     crate::catalog::us_label(it.cert_us),
                    "runtime":       it.runtime,
                    "tags":          crate::contract::item_tags(profiles.get(it.id), &it.genres, 3),
                    "badges":        crate::contract::item_badges(chart.then_some(n + 1), release, today),
                })
            })
            .collect();
        if items.len() < MIN_ROW_ITEMS {
            continue;
        }
        let row_type = match row.archetype.as_str() { "trending" => "top_ten", "collection" => "collection", _ => "vibe" };
        let mut obj = json!({
            "id":        format!("{}-{}", row.archetype, i),
            "row_type":  row_type,
            "type":      if row.kind.is_empty() && row.archetype != "thematic" && row.archetype != "trending" { row.archetype.clone() } else { row.kind.clone() },
            "title":     row.title,
            "items":     items,
        });
        // the facts of a personal or collection row (translation key and arguments of the title, the reason, a collection's text and target) sit on the row itself
        if let (Some(serde_json::Value::Object(m)), Some(o)) = (&row.meta, obj.as_object_mut()) {
            for (k, v) in m { o.insert(k.clone(), v.clone()); }
        }
        out.push(obj);
    }
    out
}
