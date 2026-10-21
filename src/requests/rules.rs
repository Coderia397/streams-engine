//! The fact rules of a request: what a title must carry to be in the row. Every list is "any of"; an empty list or a missing bound asks nothing.
use crate::catalog::CatalogItem;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Rules {
    /// At least one of these TMDB genres (movie and series names, such as "Science Fiction" and "Sci-Fi & Fantasy")
    #[serde(default)]
    pub genres_any: Vec<String>,
    /// None of these genres
    #[serde(default)]
    pub genres_none: Vec<String>,
    /// At least one of these TMDB keywords (lower case, whole keyword)
    #[serde(default)]
    pub keywords_any: Vec<String>,
    /// Release year from and to, inclusive
    pub year_min: Option<i32>,
    pub year_max: Option<i32>,
    /// One of these original languages
    #[serde(default)]
    pub languages_any: Vec<String>,
    /// One of these origin countries
    #[serde(default)]
    pub countries_any: Vec<String>,
    /// Anime or not; none asks nothing
    pub anime: Option<bool>,
}

impl Rules {
    /// Does the title carry what the request asks? The cheap checks come first.
    pub fn passes(&self, it: &CatalogItem) -> bool {
        if self.anime.is_some_and(|a| a != it.is_anime) {
            return false;
        }
        if self.year_min.is_some_and(|y| it.release_year.is_none_or(|r| r < y)) || self.year_max.is_some_and(|y| it.release_year.is_none_or(|r| r > y)) {
            return false;
        }
        if !self.languages_any.is_empty() && !self.languages_any.iter().any(|l| *l == it.original_language) {
            return false;
        }
        if !self.countries_any.is_empty() && !it.countries.iter().any(|c| self.countries_any.contains(c)) {
            return false;
        }
        if !self.genres_any.is_empty() && !it.genres.iter().any(|g| self.genres_any.contains(g)) {
            return false;
        }
        if it.genres.iter().any(|g| self.genres_none.contains(g)) {
            return false;
        }
        if !self.keywords_any.is_empty() && !it.keywords.iter().any(|k| self.keywords_any.contains(k)) {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(genres: &[&str], keywords: &[&str], year: i32, lang: &str, countries: &[&str], anime: bool) -> CatalogItem {
        CatalogItem {
            id: 1, tmdb_id: 1, media_type: "movie".into(), title: "t".into(), release_year: Some(year), genres: genres.iter().map(|s| s.to_string()).collect(),
            overview: String::new(), tagline: None, director: None, poster_path: None, backdrop_path: None, vote_average: 7.0, popularity: 10.0, is_anime: anime,
            vote_count: 100, adult: false, original_language: lang.into(), tv_type: None, countries: countries.iter().map(|s| s.to_string()).collect(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(), cert: 0, cert_gb: 0, cert_us: 0, runtime: 100, flags: 0,
        }
    }

    #[test]
    fn every_rule_must_hold_and_empty_rules_ask_nothing() {
        let heist_thriller = item(&["Thriller", "Crime"], &["heist", "bank"], 1995, "en", &["US"], false);
        assert!(Rules::default().passes(&heist_thriller));
        let r = Rules { genres_any: vec!["Thriller".into()], keywords_any: vec!["heist".into()], year_min: Some(1990), year_max: Some(1999), ..Default::default() };
        assert!(r.passes(&heist_thriller));
        assert!(!r.passes(&item(&["Comedy"], &["heist"], 1995, "en", &["US"], false)), "wrong genre");
        assert!(!r.passes(&item(&["Thriller"], &["robbery"], 1995, "en", &["US"], false)), "no keyword");
        assert!(!r.passes(&item(&["Thriller"], &["heist"], 2001, "en", &["US"], false)), "outside the decade");
        let korean = Rules { languages_any: vec!["ko".into()], ..Default::default() };
        assert!(korean.passes(&item(&["Drama"], &[], 2010, "ko", &["KR"], false)) && !korean.passes(&heist_thriller));
        let british = Rules { languages_any: vec!["en".into()], countries_any: vec!["GB".into()], ..Default::default() };
        assert!(!british.passes(&heist_thriller) && british.passes(&item(&["Drama"], &[], 2010, "en", &["GB"], false)));
        let not_anime = Rules { anime: Some(false), genres_none: vec!["Animation".into()], ..Default::default() };
        assert!(!not_anime.passes(&item(&["Action"], &[], 2010, "ja", &["JP"], true)));
        assert!(!not_anime.passes(&item(&["Animation"], &[], 2010, "ja", &["JP"], false)));
        let no_year = CatalogItem { release_year: None, ..heist_thriller.clone() };
        assert!(!r.passes(&no_year), "a decade rule needs a known year");
    }
}
