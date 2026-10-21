//! How many rows a page may have: the policy constants, and the row caps of genre and language pages (what their pool of titles can fill).
use crate::catalog::CatalogMap;
use crate::genre_spec::GenreSpec;
use crate::identity::Maturity;
use crate::profiles::ProfileStore;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Row policy derived in ai-engine/config/row_architecture.json (surfaces block).
/// Initial rows cover about three desktop screens; the cap is Netflix's ~40 rows per
/// page, reduced for genres whose quality pool cannot fill that many rows without
/// repeating titles across two sessions (pool / (15 titles x 2)).
pub const NETFLIX_ROWS_CAP: usize = 60;
pub const INITIAL_ROWS: usize = 12;
pub const INITIAL_ROWS_GENRE: usize = 10;
pub const MIN_ROWS_FLOOR: usize = 8; // two desktop screens
const ROW_TITLES: usize = 15;
const SESSION_TURNOVER: usize = 2;
/// Below this a genre has too few rows to be its own page.
pub const MIN_GENRE_PAGE_ROWS: usize = 8;

/// How many rows the page can have and how many are built for this request.
pub struct PagePlan {
    /// Rows the page's own titles can fill (a genre page is topped up with neighbouring rows to the floor, but offered only when this is enough); unlimited when the page has no genre
    pub own_cap: usize,
    /// Rows the page may show in total
    pub cap: usize,
    /// Rows to build now: everything up to the end of the requested page
    pub target_rows: usize,
}

impl PagePlan {
    pub fn new(catalog: &CatalogMap, profiles: &ProfileStore, surface: &str, genre_spec: Option<&GenreSpec>, language: Option<&str>, maturity: Option<Maturity>, wanted_rows: usize) -> Self {
        let own_cap = match (genre_spec, language) {
            (Some(spec), None) => genre_row_cap_cached(catalog, profiles, surface, spec, maturity),
            _ => usize::MAX,
        };
        let cap = match (genre_spec, language) {
            (Some(_), None) => own_cap.max(crate::page::THIN_PAGE_FLOOR),
            (_, Some(lang)) => language_row_cap_cached(catalog, profiles, surface, lang, maturity),
            (None, None) => NETFLIX_ROWS_CAP,
        };
        let target_rows = wanted_rows.min(cap.max(MIN_ROWS_FLOOR)).max(MIN_ROWS_FLOOR);
        Self { own_cap, cap, target_rows }
    }

    /// Whether the client should offer this page at all.
    pub fn supported(&self, has_genre_spec: bool, has_genre: bool, has_language: bool) -> bool {
        (!has_genre && !has_language) || (if has_genre_spec && !has_language { self.own_cap } else { self.cap }) >= MIN_GENRE_PAGE_ROWS
    }
}

/// Rows a genre page can support for this surface, from the size of its quality pool.
pub fn genre_row_cap(catalog: &CatalogMap, profiles: &ProfileStore, surface: &str, spec: &GenreSpec, maturity: Option<Maturity>) -> usize {
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
    // a children's page has a small pool and its rows may share titles (as on Netflix), so each row needs half as many
    // composed rows (note 101) keep only their first screens unique across the page, so a row costs the pool FIRST_SCREEN titles, not a whole row
    let per_row = if crate::requests::enabled() { crate::page::FIRST_SCREEN * SESSION_TURNOVER } else if spec.is_judged() || maturity.is_some_and(|m| m.is_kids()) { ROW_TITLES } else { ROW_TITLES * SESSION_TURNOVER };
    (pool / per_row).min(NETFLIX_ROWS_CAP)
}

/// Rows a language page can support: the quality titles in that language the profile may see, 30 per row.
pub fn language_row_cap(catalog: &CatalogMap, profiles: &ProfileStore, surface: &str, language: &str, maturity: Option<Maturity>) -> usize {
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

/// A cap only changes when the catalog reloads, so it is kept for ten minutes per kind of page, catalog size and profile count.
fn cached_cap(key: String, compute: impl FnOnce() -> usize) -> usize {
    static CAPS: OnceLock<Mutex<crate::det::HashMap<String, (usize, Instant)>>> = OnceLock::new();
    let map = CAPS.get_or_init(|| Mutex::new(crate::det::HashMap::default()));
    if let Some((v, at)) = map.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        if at.elapsed() < Duration::from_secs(600) {
            return *v;
        }
    }
    let v = compute();
    map.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (v, Instant::now()));
    v
}

fn genre_row_cap_cached(catalog: &CatalogMap, profiles: &ProfileStore, surface: &str, spec: &GenreSpec, maturity: Option<Maturity>) -> usize {
    let key = format!("genre|{surface}|{spec:?}|{}|{}|{}", catalog.len(), profiles.profiles.len(), maturity.map_or("-", |m| m.as_str()));
    cached_cap(key, || genre_row_cap(catalog, profiles, surface, spec, maturity))
}

fn language_row_cap_cached(catalog: &CatalogMap, profiles: &ProfileStore, surface: &str, language: &str, maturity: Option<Maturity>) -> usize {
    let key = format!("language|{surface}|{language}|{}|{}|{}", catalog.len(), profiles.profiles.len(), maturity.map_or("-", |m| m.as_str()));
    cached_cap(key, || language_row_cap(catalog, profiles, surface, language, maturity))
}
