use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn item(genres: &[&str], countries: &[&str], keywords: &[&str], year: i32) -> CatalogItem {
        CatalogItem {
            id: 1, tmdb_id: 1, media_type: "tv".into(), title: "t".into(), release_year: Some(year),
            genres: genres.iter().map(|s| s.to_string()).collect(), overview: String::new(), tagline: None, director: None,
            poster_path: None, backdrop_path: None, vote_average: 7.0, popularity: 10.0, is_anime: false, vote_count: 0,
            adult: false, original_language: match countries.first() { Some(&"JP") => "ja", Some(&"FR") => "fr", Some(&"ES") => "es", Some(&"KR") => "ko", _ => "en" }.into(), tv_type: None,
            countries: countries.iter().map(|s| s.to_string()).collect(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(), runtime: 0, flags: 0, cert: 0, cert_gb: 0, cert_us: 0,
        }
    }

    #[test]
    fn country_theme_genre() {
        let items: Vec<CatalogItem> = (0..10).map(|i| item(&["Mystery", "Drama"], &["JP"], &["detective"], 2010 + i)).collect();
        let refs: Vec<&CatalogItem> = items.iter().collect();
        let c = candidates(&refs);
        assert_eq!(c[0], "Japanese Detective Mysteries");
        assert!(c.contains(&"Detective Mysteries".to_string()));
    }

    #[test]
    fn a_country_needs_the_first_listed_country_and_a_fitting_language() {
        // Spanish language films with a French co production are not French films
        let mut items: Vec<CatalogItem> = (0..10).map(|i| item(&["Drama"], &["ES", "FR"], &[], 2010 + i)).collect();
        let refs: Vec<&CatalogItem> = items.iter().collect();
        assert!(candidates(&refs).iter().all(|c| !c.contains("French")), "{:?}", candidates(&refs));
        assert!(candidates(&refs).iter().any(|c| c.starts_with("Spanish")));
        // first country France but the film is in English: no country in the name
        for it in items.iter_mut() { it.countries = vec!["FR".into(), "US".into()]; it.original_language = "en".into(); }
        let refs: Vec<&CatalogItem> = items.iter().collect();
        assert!(candidates(&refs).iter().all(|c| !c.contains("French")));
    }

    #[test]
    fn standalone_theme_and_decade() {
        let items: Vec<CatalogItem> = (0..10).map(|i| item(&["Comedy"], &["US"], if i < 6 { &["sitcom"] } else { &[] }, 1990 + i)).collect();
        let refs: Vec<&CatalogItem> = items.iter().collect();
        let c = candidates(&refs);
        assert_eq!(c[0], "Sitcoms");
        assert!(c.contains(&"90s Comedies".to_string()));
    }

    #[test]
    fn anime_and_no_false_decade() {
        let mut items: Vec<CatalogItem> = (0..10).map(|i| item(&["Animation", "Romance"], &["JP"], &[], 1995 + i * 3)).collect();
        for it in &mut items { it.is_anime = true; }
        let refs: Vec<&CatalogItem> = items.iter().collect();
        let c = candidates(&refs);
        assert!(c.iter().any(|t| t == "Animation Anime" || t == "Romance Anime"), "{c:?}");
        assert!(!c.iter().any(|t| t.starts_with("90s") || t.starts_with("2000s") || t.starts_with("2010s")), "{c:?}");
        assert!(candidates(&refs[..3]).is_empty());
    }

    #[test]
    fn animated_rows_say_so() {
        let items: Vec<CatalogItem> = (0..10).map(|i| item(&["Animation", "Comedy"], &["US"], &[], 2010 + i)).collect();
        let refs: Vec<&CatalogItem> = items.iter().collect();
        let marked = mark_animated(vec!["Comedies".to_string(), "Animation Picks".to_string()], &refs);
        assert_eq!(marked, vec!["Animated Comedies".to_string(), "Animation Picks".to_string()]);
        let live: Vec<CatalogItem> = (0..10).map(|i| item(&["Comedy"], &["US"], &[], 2010 + i)).collect();
        let live_refs: Vec<&CatalogItem> = live.iter().collect();
        assert_eq!(mark_animated(vec!["Comedies".to_string()], &live_refs), vec!["Comedies".to_string()]);
    }

    #[test]
    fn tone_names_skip_repeats() {
        let items: Vec<CatalogItem> = (0..8).map(|i| item(&["Crime", "Drama"], &["US"], &[], 2000 + i)).collect();
        let refs: Vec<&CatalogItem> = items.iter().collect();
        assert_eq!(tone_title("Dark", &refs), Some("Dark Crime Stories".to_string()));
        let thr: Vec<CatalogItem> = (0..8).map(|i| item(&["Thriller"], &["US"], &[], 2000 + i)).collect();
        let thr_refs: Vec<&CatalogItem> = thr.iter().collect();
        assert_eq!(tone_title("Suspenseful", &thr_refs), None);
    }
}

#[cfg(test)]
mod name_fit_tests {
    use super::*;

    fn item(year: i32, genres: &[&str]) -> CatalogItem {
        CatalogItem {
            id: year as u32, tmdb_id: year as u32, media_type: "movie".into(), title: format!("T{year}"), release_year: Some(year), genres: genres.iter().map(|g| g.to_string()).collect(), overview: String::new(), tagline: None,
            director: None, poster_path: None, backdrop_path: None, vote_average: 7.0, popularity: 10.0, is_anime: false, vote_count: 100, adult: false, original_language: "en".into(), tv_type: None,
            countries: vec![], keywords: vec![], cert: 0, cert_gb: 0, cert_us: 0, runtime: 100, flags: 0,
        }
    }

    #[test]
    fn a_decade_anywhere_in_the_name_must_be_true() {
        let in_2000s: Vec<CatalogItem> = (2000..2010).map(|y| item(y, &["Comedy", "Romance"])).collect();
        let spread: Vec<CatalogItem> = (1950..2020).step_by(7).map(|y| item(y, &["Comedy", "Romance"])).collect();
        let a: Vec<&CatalogItem> = in_2000s.iter().collect();
        let b: Vec<&CatalogItem> = spread.iter().collect();
        assert!(name_fits("Romantic Comedies of the 2000s", &a));
        assert!(!name_fits("Romantic Comedies of the 2000s", &b));
        assert!(name_fits("Romantic Comedies", &b));
    }

    #[test]
    fn a_rom_com_name_needs_romance_and_comedy() {
        let both: Vec<CatalogItem> = (2000..2010).map(|y| item(y, &["Comedy", "Romance"])).collect();
        let horror: Vec<CatalogItem> = (2000..2010).map(|y| item(y, &["Horror", "Fantasy"])).collect();
        let a: Vec<&CatalogItem> = both.iter().collect();
        let b: Vec<&CatalogItem> = horror.iter().collect();
        assert!(name_fits("Holiday Rom-Coms", &a));
        assert!(!name_fits("Spooky Fantasy Rom-Coms", &b));
    }

    #[test]
    fn animated_and_cartoon_names_need_animation() {
        let live: Vec<CatalogItem> = (2000..2010).map(|y| item(y, &["Comedy", "Family"])).collect();
        let toons: Vec<CatalogItem> = (2000..2010).map(|y| item(y, &["Animation", "Adventure", "Family"])).collect();
        let a: Vec<&CatalogItem> = live.iter().collect();
        let b: Vec<&CatalogItem> = toons.iter().collect();
        assert!(!name_fits("Animated Bear Adventures", &a));
        assert!(name_fits("Animated Bear Adventures", &b));
        assert!(!name_fits("Cartoon Capers", &a));
        assert!(name_fits("Cartoon Capers", &b));
    }

    #[test]
    fn a_nationality_word_needs_titles_from_there() {
        let mk = |country: &str, lang: &str| -> Vec<CatalogItem> {
            (2000..2010).map(|y| { let mut it = item(y, &["Thriller", "Crime"]); it.countries = vec![country.into()]; it.original_language = lang.into(); it }).collect()
        };
        let korean = mk("KR", "ko");
        let american = mk("US", "en");
        let british = mk("GB", "en");
        let k: Vec<&CatalogItem> = korean.iter().collect();
        let u: Vec<&CatalogItem> = american.iter().collect();
        let g: Vec<&CatalogItem> = british.iter().collect();
        assert!(name_fits("Korean Thrillers", &k));
        assert!(!name_fits("Korean Thrillers", &u));
        assert!(name_fits("British Crime Stories", &g));
        assert!(!name_fits("British Crime Stories", &u));
        assert!(name_fits("American Crime Stories", &u));
        // a name without a nationality is not affected, and a language alone counts (Korean titles with no country listed)
        assert!(name_fits("Crime Stories", &u));
        let mut no_country = mk("KR", "ko");
        for it in no_country.iter_mut() { it.countries.clear(); }
        let n: Vec<&CatalogItem> = no_country.iter().collect();
        assert!(name_fits("Korean Thrillers", &n));
    }
}
