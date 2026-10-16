/// Generate a row title. Title is always derived AFTER retrieval (Spotify principle).
///
/// Chain (first non-None wins):
/// 1. taxonomy Hungarian assignment title (see taxonomy.rs) — passed as `taxonomy_title`
/// 2. era and genre mix ("90s Crime", "Crime & Thriller")
/// 3. `{genre} Picks`
/// 4. `More {genre}`
///
/// The caller (page.rs) handles uniqueness — this function just generates candidates.
pub fn derive_title(
    dominant_genre: &str,
    second_genre: Option<&str>,
    era: Option<&str>,
    anchor_title: Option<&str>,
    taxonomy_title: Option<String>,
) -> Vec<String> {
    let mut candidates = Vec::new();

    // 1. Taxonomy alignment winner
    if let Some(t) = taxonomy_title {
        candidates.push(t);
    }

    // 2. Data derived titles: era and genre mix
    if !dominant_genre.is_empty() {
        if let Some(e) = era {
            candidates.push(format!("{e} {dominant_genre}"));
        }
        if let Some(g2) = second_genre {
            candidates.push(format!("{dominant_genre} & {g2}"));
        }
    }

    // 3. Genre Picks
    if !dominant_genre.is_empty() {
        candidates.push(format!("{dominant_genre} Picks"));
        // 3b. Like <lead title> keeps otherwise identical fallbacks unique
        if let Some(t) = anchor_title {
            candidates.push(format!("Like {t}"));
        }
        // 4. More Genre
        candidates.push(format!("More {dominant_genre}"));
    } else {
        candidates.push("More Picks".to_string());
    }

    candidates
}

/// Derive the BYW row title for an anchor item.
pub fn byw_title(anchor_title: &str) -> String {
    format!("Because You Watched {anchor_title}")
}

/// Clamp a title to 4 words maximum (hard rule from platform research).
/// Splits on whitespace and rejoins.
pub fn clamp_words(title: &str, max_words: usize) -> String {
    // "Like <title>" rows carry a full title, so allow room for it
    let max_words = if title.starts_with("Like ") { max_words + 3 } else { max_words };
    let mut words: Vec<&str> = title.split_whitespace().take(max_words).collect();
    // Never end on a dangling connector like "&"
    while words.last().is_some_and(|w| matches!(w.to_lowercase().as_str(), "&" | "and" | "of" | "the" | "to" | "in" | "for" | "with" | "a" | "an" | "from" | "on")) {
        words.pop();
    }
    words.join(" ")
}
