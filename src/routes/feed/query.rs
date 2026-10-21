//! What the client asks for: the raw query, and the request normalised once so no other part of the feed re-reads the query text.
use super::policy::{INITIAL_ROWS, INITIAL_ROWS_GENRE, MIN_ROWS_FLOOR, NETFLIX_ROWS_CAP};
use crate::identity::{maturity_of, Caller, Maturity};
use crate::signals::Signals;
use serde::Deserialize;

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

/// The request after the rules of the endpoint are applied: who asks, for which page, how many rows.
pub struct FeedRequest {
    /// A verified caller's session is its profile (or user) id, whatever the query says
    pub session_id: Option<String>,
    pub maturity: Option<Maturity>,
    /// "movie", "tv" or "both"
    pub surface: &'static str,
    /// The taste text as sent (`taste`, else `affinity`)
    pub taste: String,
    pub genre: Option<String>,
    pub language: Option<String>,
    /// Only a plain id: letters, digits, underscore, hyphen, at most 48 characters
    pub collection: Option<String>,
    pub per_page: usize,
    pub offset: usize,
    /// The client's local time is between 22:00 and 04:59
    pub late_night: bool,
    pub signals: Signals,
}

impl FeedRequest {
    pub fn parse(q: &FeedQuery, caller: &Option<Caller>) -> Self {
        let session_id = caller.as_ref().map(|c| c.session_key()).or_else(|| q.session_id.clone());
        let maturity = maturity_of(caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating));
        let surface = normalize_surface(q.surface.as_deref().unwrap_or("both"));
        let taste = q.taste.as_deref().or(q.affinity.as_deref()).unwrap_or("").to_string();
        let genre = q.genre.as_deref().map(str::trim).filter(|g| !g.is_empty()).map(str::to_string);
        let language = q.language.as_deref().map(str::trim).filter(|l| !l.is_empty() && l.len() <= 3 && l.chars().all(|c| c.is_ascii_alphabetic())).map(str::to_lowercase);
        let initial = if genre.is_some() || language.is_some() { INITIAL_ROWS_GENRE } else { INITIAL_ROWS };
        let per_page = q.rows.or(q.limit).unwrap_or(initial).clamp(MIN_ROWS_FLOOR, NETFLIX_ROWS_CAP);
        let offset = q.offset.unwrap_or(0).min(NETFLIX_ROWS_CAP);
        let collection = q.collection.as_deref().map(str::trim).filter(|c| !c.is_empty() && c.len() <= 48 && c.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')).map(str::to_string);
        Self {
            session_id,
            maturity,
            surface,
            taste,
            genre,
            language,
            collection,
            per_page,
            offset,
            late_night: q.hour.is_some_and(|h| h >= 22 || h <= 4),
            signals: Signals::parse(q.saved.as_deref(), q.recent.as_deref(), q.liked.as_deref()),
        }
    }
}

pub(super) fn normalize_surface(s: &str) -> &'static str {
    match s {
        "movie" | "movies" | "films" | "film" => "movie",
        "tv" | "series" | "shows" => "tv",
        _ => "both",
    }
}

/// Parse the taste query parameter into a genre affinity list.
/// Input: "28:0.45,80:0.31" → [("Action", 0.45), ("Drama", 0.31)]
/// Genre IDs are mapped to names using the TMDB genre taxonomy.
pub(super) fn parse_taste(taste_str: &str) -> Vec<(String, f32)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surfaces_have_one_spelling() {
        assert_eq!(normalize_surface("films"), "movie");
        assert_eq!(normalize_surface("series"), "tv");
        assert_eq!(normalize_surface("home"), "both");
        assert_eq!(normalize_surface(""), "both");
    }

    #[test]
    fn taste_maps_genre_ids_and_keeps_names() {
        let t = parse_taste("28:0.6, Horror:2, :1");
        assert_eq!(t, vec![("Action".to_string(), 0.6), ("Horror".to_string(), 1.0)]);
    }

    #[test]
    fn paging_and_collection_rules_apply_once() {
        let q = FeedQuery { rating: None, surface: Some("series".into()), taste: None, affinity: Some("Drama".into()), genre: Some(" Crime ".into()), collection: Some("bad id!".into()),
            language: Some("KO".into()), session_id: Some("s1".into()), rows: None, limit: Some(500), offset: Some(500), hour: Some(23), saved: None, recent: None, liked: None };
        let r = FeedRequest::parse(&q, &None);
        assert_eq!((r.surface, r.taste.as_str(), r.genre.as_deref(), r.language.as_deref()), ("tv", "Drama", Some("Crime"), Some("ko")));
        assert_eq!((r.per_page, r.offset, r.late_night), (NETFLIX_ROWS_CAP, NETFLIX_ROWS_CAP, true));
        assert!(r.collection.is_none());
    }
}
