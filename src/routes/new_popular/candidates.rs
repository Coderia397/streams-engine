//! The titles the page can show, from the catalog and from fresh.json, with the rules for who may see them.
use super::*;

/// One title as the page sees it, from the catalog, from fresh.json, or both (the catalog supplies certificates and genres, fresh.json the
/// current popularity and the release date).
#[derive(Clone)]
pub(super) struct Cand {
    pub(super) media_type: String,
    pub(super) tmdb_id: u32,
    pub(super) title: String,
    pub(super) overview: String,
    pub(super) poster_path: Option<String>,
    pub(super) backdrop_path: Option<String>,
    pub(super) vote_average: f32,
    pub(super) vote_count: u32,
    pub(super) popularity: f32,
    /// YYYY-MM-DD or empty
    pub(super) release_date: String,
    pub(super) release_year: Option<i32>,
    pub(super) genres: Vec<String>,
    pub(super) language: String,
    pub(super) tv_type: Option<String>,
    pub(super) cert: u8,
    pub(super) cert_gb: u8,
    pub(super) cert_us: u8,
    pub(super) in_catalog: bool,
    pub(super) allowed: bool,
    pub(super) lists: Vec<String>,
    pub(super) ranks: HashMap<String, usize>,
}

impl Cand {
    pub(super) fn key(&self) -> (String, u32) {
        (self.media_type.clone(), self.tmdb_id)
    }
    pub(super) fn released(&self, today: &str) -> bool {
        !self.release_date.is_empty() && self.release_date.as_str() <= today
    }
    pub(super) fn upcoming(&self, today: &str) -> bool {
        !self.release_date.is_empty() && self.release_date.as_str() > today
    }
    pub(super) fn days_since(&self, today: i64) -> Option<i64> {
        parse_days(&self.release_date).map(|d| today - d)
    }
    pub(super) fn in_list(&self, tag: &str) -> bool {
        self.lists.iter().any(|l| l == tag)
    }
    pub(super) fn rank_in(&self, tag: &str) -> usize {
        self.ranks.get(tag).copied().unwrap_or(usize::MAX)
    }
}

pub(super) fn from_catalog(item: &CatalogItem, allowed: bool) -> Cand {
    Cand {
        media_type: item.media_type.clone(), tmdb_id: item.tmdb_id, title: item.title.clone(), overview: item.overview.clone(),
        poster_path: item.poster_path.clone(), backdrop_path: item.backdrop_path.clone(), vote_average: item.vote_average, vote_count: item.vote_count,
        popularity: item.popularity, release_date: String::new(), release_year: item.release_year, genres: item.genres.clone(),
        language: item.original_language.clone(), tv_type: item.tv_type.clone(), cert: item.cert, cert_gb: item.cert_gb, cert_us: item.cert_us,
        in_catalog: true, allowed, lists: Vec::new(), ranks: HashMap::default(),
    }
}

pub(super) fn from_fresh(f: &FreshItem, catalog_item: Option<&CatalogItem>, allowed: bool) -> Cand {
    let mut c = match catalog_item {
        Some(item) => from_catalog(item, allowed),
        None => Cand {
            media_type: f.media_type.clone(), tmdb_id: f.id, title: f.title.clone(), overview: f.overview.clone(), poster_path: f.poster_path.clone(),
            backdrop_path: f.backdrop_path.clone(), vote_average: f.vote_average, vote_count: f.vote_count, popularity: f.popularity,
            release_date: String::new(), release_year: f.release_date.get(..4).and_then(|y| y.parse().ok()), genres: f.genres.clone(),
            language: f.original_language.clone().unwrap_or_default(), tv_type: None, cert: 0, cert_gb: 0, cert_us: 0, in_catalog: false, allowed,
            lists: Vec::new(), ranks: HashMap::default(),
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
pub(super) fn catalog_allowed(item: &CatalogItem, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore) -> bool {
    if item.adult || (crate::page::safety_nets() && item.flags & crate::catalog::FLAG_SUGGESTIVE != 0) {
        return false;
    }
    maturity.is_none_or(|m| crate::page::maturity_allows_with(m, profiles, item))
}

pub(super) fn fresh_only_allowed(f: &FreshItem, maturity: Option<Maturity>) -> bool {
    !f.adult && maturity.is_none_or(|m| m == Maturity::Adult)
}

/// All candidates: every fresh title, plus the catalog's own newest titles (release year this year or last) for rows that need no date.
pub(super) fn candidates(catalog: &CatalogMap, fresh: &FreshStore, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, year: i32) -> Vec<Cand> {
    let mut out: Vec<Cand> = Vec::new();
    let mut seen: HashSet<(String, u32)> = HashSet::default();
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

pub(super) fn has_art(c: &Cand) -> bool {
    c.poster_path.as_deref().is_some_and(|p| !p.is_empty())
}

pub(super) fn has_backdrop(c: &Cand) -> bool {
    has_art(c) && c.backdrop_path.as_deref().is_some_and(|p| !p.is_empty())
}

pub(super) fn by_popularity(a: &&Cand, b: &&Cand) -> std::cmp::Ordering {
    b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.tmdb_id.cmp(&b.tmdb_id)).then_with(|| a.media_type.cmp(&b.media_type))
}

pub(super) fn take_top(mut v: Vec<&Cand>, n: usize) -> Vec<Cand> {
    v.sort_by(by_popularity);
    v.truncate(n);
    v.into_iter().cloned().collect()
}
