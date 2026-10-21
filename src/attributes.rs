//! Attribute rows: rows defined by a fact of the catalog instead of by a topic group, such as the runtime ("Films Under 90 Minutes"), the format ("Limited Series") or the
//! source ("Based on Books"). The titles of each kind are listed once at catalog load (`Catalog::attr_keys`, best quality first); a page takes the kinds that suit its surface,
//! checks every title against the page's own rules (age limit, genre) and rotates the order a little per session.
use crate::catalog::CatalogItem;

pub struct AttrSpec {
    pub id: &'static str,
    /// "movie", "tv" or "both": the kind of title the row holds
    pub media: &'static str,
    /// Row title on a films page, a series page and a page of both
    pub title_movie: &'static str,
    pub title_tv: &'static str,
    pub title_both: &'static str,
    pub test: fn(&CatalogItem) -> bool,
}

fn keyword(it: &CatalogItem, words: &[&str]) -> bool {
    it.keywords.iter().any(|k| words.contains(&k.as_str()))
}

pub const SPECS: &[AttrSpec] = &[
    AttrSpec { id: "half_hour_comedy", media: "tv", title_movie: "", title_tv: "Half Hour Comedies", title_both: "Half Hour Comedies", test: |i| i.media_type == "tv" && (5..=32).contains(&i.runtime) && i.genres.iter().any(|g| g == "Comedy") },
    AttrSpec { id: "limited_series", media: "tv", title_movie: "", title_tv: "Limited Series", title_both: "Limited Series", test: |i| i.media_type == "tv" && i.tv_type.as_deref() == Some("Miniseries") },
    AttrSpec { id: "books", media: "both", title_movie: "Films Based on Books", title_tv: "Series Based on Books", title_both: "Based on Books", test: |i| keyword(i, &["based on novel or book", "based on young adult novel"]) },
    AttrSpec { id: "true_story", media: "both", title_movie: "Films Based on True Stories", title_tv: "Series Based on True Stories", title_both: "Based on True Stories", test: |i| keyword(i, &["based on true story"]) },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_ids_are_unique_and_titles_are_set_for_their_media() {
        let mut ids: Vec<&str> = SPECS.iter().map(|s| s.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), SPECS.len());
        for s in SPECS {
            assert!(!s.title_both.is_empty());
            if s.media != "tv" { assert!(!s.title_movie.is_empty(), "{}", s.id); }
            if s.media != "movie" { assert!(!s.title_tv.is_empty(), "{}", s.id); }
        }
    }
}
