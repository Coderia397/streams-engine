//! Keys of titles: TMDB numbers movies and series separately, so the engine keys every title by `item_key`.
use super::*;

/// TMDB numbers movies and TV series separately, so the same id exists in both. The engine keys every
/// title by `item_key`: movies keep their TMDB id, TV series get the top bit set.
pub const TV_BIT: u32 = 1 << 31;

#[inline]
pub fn item_key(media_type: &str, tmdb_id: u32) -> u32 {
    if media_type == "tv" { tmdb_id | TV_BIT } else { tmdb_id }
}

/// Resolve a client supplied (id, media_type) to an internal key that exists in the catalog.
/// Without a media type the movie is tried first, then the series.
pub fn resolve_key(catalog: &CatalogMap, tmdb_id: u32, media_type: Option<&str>) -> Option<u32> {
    match media_type {
        Some(mt) => Some(item_key(mt, tmdb_id)).filter(|k| catalog.contains_key(k)),
        None => [tmdb_id, tmdb_id | TV_BIT].into_iter().find(|k| catalog.contains_key(k)),
    }
}
