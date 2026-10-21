//! The Top 10 charts: the most popular released titles of a type, a language or a genre.
use super::*;

/// The Top 10: the most popular released titles of one type, by popularity refreshed daily.
pub(super) fn chart(catalog: &CatalogMap, fresh: &FreshStore, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, media_type: &str, today: &str, year: i32) -> Vec<Cand> {
    chart_where(catalog, fresh, maturity, profiles, today, year, |i| i.media_type == media_type)
}

/// The Top 10 of the titles that pass `keep`: a language, a genre, a type.
pub(super) fn chart_where(catalog: &CatalogMap, fresh: &FreshStore, maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, today: &str, year: i32, keep: impl Fn(&CatalogItem) -> bool) -> Vec<Cand> {
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
