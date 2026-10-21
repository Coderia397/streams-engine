//! Rules about row titles: which names are fit for a page and which words carry meaning.
use super::*;

/// Significant words of a title (3 or more letters, not a number, not a filler word), lower case, without repeats.
/// The candidate names of a row that keep their promise for its titles (genre, nationality and decade words, see row_label::name_fits), in order; the fallback, which
/// promises nothing but the page itself, when none does. A derived name such as "Documentary & Animation" over one animated title out of fifteen is never shown.
pub(crate) fn keep_promises(candidates: Vec<String>, items: &[&CatalogItem], fallback: &str) -> Vec<String> {
    let mut kept: Vec<String> = candidates.into_iter().filter(|c| crate::row_label::name_fits(c, items)).collect();
    kept.push(fallback.to_string());
    kept
}

pub(crate) fn title_words(title: &str) -> Vec<String> {
    const FILLER: &[&str] = &[
        "the", "and", "for", "you", "your", "with", "from", "that", "this", "are", "was", "not", "but", "his", "her", "its", "our",
        "their", "who", "why", "how", "what", "when", "where", "into", "over", "under", "off", "out", "one", "two", "new", "old",
        "man", "men", "day", "night", "life", "love", "last", "first", "les", "der", "die", "das", "los", "las", "part", "chapter",
        "season", "episode", "movie", "film", "story", "special",
    ];
    let mut out: Vec<String> = Vec::new();
    for w in title.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if w.chars().count() >= 3 && !w.chars().all(|c| c.is_ascii_digit()) && !FILLER.contains(&w) && !out.iter().any(|o| o == w) {
            out.push(w.to_string());
        }
    }
    out
}

/// A row title that names adult content ("Adult Film Anthologies", "Erotic Thrillers"); "Young Adult" and "Adult Animation" are fine.
pub(crate) fn is_suggestive_title(title: &str) -> bool {
    let t = title.to_lowercase();
    ["adult film", "adult video", "adult entertainment", "adult content"].iter().any(|p| t.contains(p))
        || t.split(|c: char| !c.is_alphanumeric()).any(|w| matches!(w, "erotic" | "erotica" | "sexy" | "porn" | "nudity" | "nude" | "softcore" | "hentai" | "ecchi" | "harem" | "stripper"))
}

/// A topic row title made only of genre and format words ("Comedies", "Documentary & Drama", "Thriller Picks"), which says nothing about how the titles feel.
pub(crate) fn is_plain_genre_title(title: &str) -> bool {
    const WORDS: &[&str] = &[
        "action", "adventure", "animation", "animated", "anime", "comedy", "comedies", "crime", "documentary", "documentaries", "drama", "dramas", "family", "fantasy", "history", "horror", "music",
        "musical", "mystery", "mysteries", "romance", "romantic", "sci-fi", "science", "fiction", "thriller", "thrillers", "war", "western", "westerns", "reality", "kids", "films", "movies", "series",
        "shows", "tv", "picks", "more", "&", "and", "of", "the",
    ];
    let mut any = false;
    for w in title.split_whitespace() {
        let lw = w.to_lowercase();
        if !WORDS.contains(&lw.as_str()) { return false; }
        any = true;
    }
    any
}

/// True when a row title names films on a series page or series on a films page.
pub(crate) fn entry_names_other_kind(title: &str, surface: &str) -> bool {
    let t = title.to_lowercase();
    let films = t.split(' ').any(|w| matches!(w, "film" | "films" | "movie" | "movies" | "cinema"));
    let series = t.split(' ').any(|w| matches!(w, "series" | "show" | "shows" | "sitcom" | "sitcoms" | "episodes" | "tv" | "(tv)" | "docuseries" | "miniseries"));
    (surface == "tv" && films) || (surface == "movie" && series)
}

impl<'a> PageContext<'a> {
    /// A vibe name is shown only when it keeps its promise for the titles of the row and does not name the other kind of title (Series on a films page).
    pub(crate) fn vibe_name_ok(&self, name: &str, items: &[&CatalogItem]) -> bool {
        crate::row_label::name_fits(name, items) && !entry_names_other_kind(name, self.surface)
    }
}

#[cfg(test)]
mod suggestive_tests {
    use super::is_suggestive_title;

    #[test]
    fn adult_content_names_are_found_and_genre_names_are_kept() {
        assert!(is_suggestive_title("Adult Film Anthologies"));
        assert!(is_suggestive_title("Erotic Thrillers"));
        assert!(!is_suggestive_title("Young Adult Dramas"));
        assert!(!is_suggestive_title("Adult Animation"));
        assert!(!is_suggestive_title("Cozy Comfort Watches"));
    }
}
