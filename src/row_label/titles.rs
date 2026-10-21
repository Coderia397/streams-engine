//! The candidate titles of a row, most specific first.
use super::*;

pub fn theme_title(theme: &str, items: &[&CatalogItem]) -> Option<String> {
    let words: Vec<&str> = theme.split_whitespace().collect();
    if words.is_empty() || words.len() > 3 || matches!(theme, "entertainment" | "comedy" | "drama" | "humor" | "action" | "family") {
        return None;
    }
    let mut count: HashMap<&str, usize> = HashMap::default();
    for it in items {
        let mut seen: Vec<&str> = Vec::new();
        for g in &it.genres {
            let k = genre_key(g);
            if !seen.contains(&k) {
                seen.push(k);
                *count.entry(k).or_insert(0) += 1;
            }
        }
    }
    let top = count.into_iter().filter(|(g, c)| noun(g).is_some() && *c * 2 >= items.len()).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0))).and_then(|(g, _)| noun(g));
    let title_case = words.iter().map(|w| { let mut c = w.chars(); c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default() }).collect::<Vec<_>>().join(" ");
    // a one word theme is an idea, not a kind of title ("Identity Action" reads wrong): "Stories of Identity". Longer themes name a kind of title.
    if words.len() == 1 {
        return Some(format!("Stories of {title_case}"));
    }
    themed(&title_case, top.unwrap_or("Stories"))
}

/// A row with no keyword or theme name can still be named by the tone that sets it apart ("Dark Crime Stories"). The tone and the genre
/// noun must not say the same thing ("Suspenseful Thrillers", "Romantic Romances").
pub fn tone_title(tone: &str, items: &[&CatalogItem]) -> Option<String> {
    let n = items.len();
    if n < 4 {
        return None;
    }
    let mut count: HashMap<&str, usize> = HashMap::default();
    for it in items {
        let mut seen: Vec<&str> = Vec::new();
        for g in &it.genres {
            let k = genre_key(g);
            if !seen.contains(&k) {
                seen.push(k);
                *count.entry(k).or_insert(0) += 1;
            }
        }
    }
    let top = count.into_iter().filter(|(g, c)| noun(g).is_some() && *c * 2 >= n).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0))).and_then(|(g, _)| noun(g))?;
    let lower = top.to_lowercase();
    let stem: String = tone.to_lowercase().chars().take(5).collect();
    let same = lower.contains(&stem) || matches!((tone, top), ("Suspenseful", "Thrillers") | ("Mysterious", "Mysteries") | ("Action-Packed", "Action") | ("Satirical", "Comedies") | ("Absurd", "Comedies") | ("Feel-Good", "Family Favorites"));
    if same {
        return None;
    }
    themed(tone, top)
}

/// Title candidates for a row, most specific first. Empty when nothing specific can be said.
/// A row that is mostly animated titles says so in every name it could take ("Animated Comedies", not "Comedies"). Anime rows already name
/// themselves, and a name that already mentions animation is left alone.
pub fn mark_animated(names: Vec<String>, items: &[&CatalogItem]) -> Vec<String> {
    let n = items.len();
    let animated = items.iter().filter(|i| !i.is_anime && i.genres.iter().any(|g| g == "Animation")).count();
    let anime = items.iter().filter(|i| i.is_anime).count();
    if n < 4 || animated * 10 < n * 7 || anime * 10 >= n * 6 {
        return names;
    }
    names
        .into_iter()
        .map(|s| if s.contains("Animat") || s.contains("Anime") || s.contains("Cartoon") { s } else { format!("Animated {s}") })
        .collect()
}

pub fn candidates(items: &[&CatalogItem]) -> Vec<String> {
    let n = items.len();
    if n < 4 {
        return Vec::new();
    }
    let share = |count: usize| count as f32 / n as f32;

    // genres (movie naming), most common first
    let mut gcount: HashMap<&str, usize> = HashMap::default();
    for it in items {
        let mut seen: Vec<&str> = Vec::new();
        for g in &it.genres {
            let k = genre_key(g);
            if !seen.contains(&k) {
                seen.push(k);
                *gcount.entry(k).or_insert(0) += 1;
            }
        }
    }
    // Drama is the catch all genre: it only leads when nothing more specific is nearly as common
    let mut genres: Vec<(&str, usize)> = gcount.into_iter().filter(|(g, _)| noun(g).is_some()).collect();
    let weight = |g: &str, c: usize| if g == "Drama" { c as f32 * 0.8 } else { c as f32 };
    genres.sort_by(|a, b| weight(b.0, b.1).partial_cmp(&weight(a.0, a.1)).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(b.0)));
    // anime rows are all Animation: name them by the next genre instead ("Comedy Anime", not "Animation Anime")
    let anime_row = share(items.iter().filter(|i| i.is_anime).count()) >= 0.6;
    if anime_row {
        genres.retain(|g| g.0 != "Animation");
    }
    let top_genre = genres.first().map(|g| g.0);
    let second_genre = genres.get(1).filter(|g| share(g.1) >= 0.3).map(|g| g.0);

    // anime share
    let anime = share(items.iter().filter(|i| i.is_anime).count()) >= 0.6;

    // country (origin, else language), skipping the US default
    // The country a viewer would name is the first listed one, and only when the original language fits it (a Spanish
    // language film with a French co-producer is not a French film; one country counted per title).
    let mut ccount: HashMap<&str, usize> = HashMap::default();
    for it in items {
        if let Some(c) = primary_country(it) {
            if c != "US" {
                *ccount.entry(c).or_insert(0) += 1;
            }
        }
    }
    let country_adj = ccount
        .into_iter()
        .filter(|(_, c)| share(*c) >= 0.6)
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0)))
        .and_then(|(c, _)| demonym(c));

    // themes: phrase -> number of items carrying one of its keywords
    let mut tcount: Vec<(&str, bool, usize)> = Vec::new();
    for &(kw, phrase, standalone) in THEMES {
        if tcount.iter().any(|t| t.0 == phrase) {
            continue;
        }
        let count = items
            .iter()
            .filter(|it| THEMES.iter().filter(|t| t.1 == phrase).any(|t| it.keywords.iter().any(|k| k == t.0)))
            .count();
        let _ = kw;
        tcount.push((phrase, standalone, count));
    }
    let min_theme = ((n as f32) * 0.4).ceil() as usize;
    let mut themes: Vec<(&str, bool, usize)> = tcount.into_iter().filter(|t| t.2 >= min_theme.max(4)).collect();
    themes.sort_by_key(|t| std::cmp::Reverse(t.2));

    // decade, strict
    let mut dcount: HashMap<i32, usize> = HashMap::default();
    let mut dtotal = 0usize;
    for it in items {
        if let Some(y) = it.release_year {
            *dcount.entry(y / 10 * 10).or_insert(0) += 1;
            dtotal += 1;
        }
    }
    let decade = dcount
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .filter(|(_, c)| dtotal > 0 && *c * 10 >= dtotal * 7)
        .map(|(d, _)| decade_label(d));

    let reality = share(items.iter().filter(|i| i.tv_type.as_deref() == Some("Reality")).count()) >= 0.6;

    let mut out: Vec<String> = Vec::new();
    let mut push = |s: String| { if !out.contains(&s) { out.push(s); } };
    let gnoun = top_genre.and_then(noun);
    // "Space Historical Dramas" reads badly: a theme before History uses plain Dramas
    let theme_noun = if top_genre == Some("History") { Some("Dramas") } else { gnoun };

    if anime {
        // a standalone anime theme first (Mecha Anime), then the genre
        if let Some(t) = themes.iter().find(|t| t.1 && t.0.contains("Anime")) {
            push(t.0.to_string());
        }
        if let Some(g) = top_genre {
            let name = match g { "Science Fiction" => "Sci-Fi", other => other };
            push(format!("{name} Anime"));
        }
        for t in themes.iter().filter(|t| !t.1) {
            if let Some(g) = gnoun { if let Some(s) = themed(t.0, g) { push(s); } }
        }
        push("Anime Favorites".to_string());
    }
    if reality {
        for t in themes.iter().filter(|t| t.1) {
            push(t.0.to_string());
        }
        push(match top_genre { Some("Comedy") => "Funny Reality Shows".to_string(), _ => "Reality Shows".to_string() });
    }
    let is_city = |p: &str| matches!(p, "Paris" | "London" | "Tokyo" | "New York" | "Los Angeles");
    if let (Some(c), Some(t)) = (country_adj, themes.iter().find(|t| !t.1 && !is_city(t.0))) {
        if let Some(g) = theme_noun { if let Some(s) = themed(t.0, g) { push(format!("{c} {s}")); } }
    }
    for t in themes.iter().filter(|t| !t.1) {
        if let Some(g) = theme_noun { if let Some(s) = themed(t.0, g) { push(s); } }
    }
    for t in themes.iter().filter(|t| t.1) {
        push(match country_adj { Some(c) if !t.0.starts_with(c) => format!("{c} {}", t.0), _ => t.0.to_string() });
        push(t.0.to_string());
    }
    if let (Some(c), Some(g)) = (country_adj, gnoun) {
        push(format!("{c} {g}"));
        if let Some(g2) = second_genre.and_then(noun) { push(format!("{c} {g} & {g2}")); }
    }
    // old films: "Classic Comedies" when at least 70 percent are from before 1980
    let old_share = share(items.iter().filter(|i| i.release_year.is_some_and(|y| y < 1980)).count());
    if let Some(g) = gnoun {
        if old_share >= 0.7 && items.iter().all(|i| i.media_type == "movie") {
            push(format!("Classic {g}"));
        }
    }
    if let (Some(d), Some(g)) = (decade.as_ref(), gnoun) {
        push(format!("{d} {g}"));
    }
    if let (Some(g), Some(nn)) = (top_genre, gnoun) {
        let top_share = genres.first().map_or(0.0, |x| share(x.1));
        if top_share >= 0.8 && matches!(g, "Western" | "Horror" | "Fantasy" | "Animation" | "Action" | "Romance" | "Comedy" | "Thriller" | "Mystery" | "War") {
            push(nn.to_string());
        }
    }
    if let (Some(g), Some(g2)) = (top_genre, second_genre) {
        let short = |x: &str| if x == "Science Fiction" { "Sci-Fi".to_string() } else { x.to_string() };
        if g != g2 { push(format!("{} & {}", short(g), short(g2))); }
    }
    // clients show at most 4 words: a longer candidate would be cut mid phrase
    out.retain(|t| t.split_whitespace().count() <= 4);
    out
}
