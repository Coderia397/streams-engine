//! Row titles from what is in the row. Candidates run from most to least specific; the page builder takes the first one
//! not yet used on the page (page.rs `claim_title`) and falls back to the older generic titles in titles.rs.
//!
//!   {country adjective} {theme} {genre}   Korean Romantic Comedies, Japanese Detective Mysteries
//!   {theme} {genre}                       Time Travel Sci-Fi, High School Comedies, Serial Killer Thrillers
//!   {theme}                               Based on True Stories, Sitcoms, Mecha Anime
//!   {genre} Anime                         Romance Anime
//!   {country adjective} {genre}           British Crime Stories, French Thrillers
//!   {decade} {genre}                      90s Action (only when at least 70 percent of the row is in that decade)
//!
//! Themes come from TMDB keywords: a theme counts when at least 40 percent of the row carries one of its keywords.

use crate::catalog::CatalogItem;
use std::collections::HashMap;

/// (TMDB keyword, phrase, standalone). Standalone phrases are a whole title; the others go before a genre noun.
const THEMES: &[(&str, &str, bool)] = &[
    ("based on true story", "Based on True Stories", true),
    ("true crime", "True Crime", true),
    ("coming of age", "Coming of Age Stories", true),
    ("slice of life", "Slice of Life", true),
    ("sitcom", "Sitcoms", true),
    ("sketch comedy", "Sketch Comedy", true),
    ("mockumentary", "Mockumentaries", true),
    ("musical", "Musicals", true),
    ("broadway musical", "Musicals", true),
    ("miniseries", "Miniseries", true),
    ("anthology", "Anthologies", true),
    ("telenovela", "Telenovelas", true),
    ("dating", "Dating Shows", true),
    ("game show", "Game and Variety Shows", true),
    ("variety show", "Game and Variety Shows", true),
    ("nature documentary", "Nature Documentaries", true),
    ("sports documentary", "Sports Documentaries", true),
    ("short film", "Short Films", true),
    ("black and white", "Black and White Classics", true),
    ("mecha", "Mecha Anime", true),
    ("isekai", "Isekai Adventures", true),
    ("shounen", "Shonen Anime", true),
    ("seinen", "Seinen Anime", true),
    ("shoujo", "Shojo Anime", true),
    ("wuxia", "Wuxia Epics", true),
    ("tokusatsu", "Tokusatsu Heroes", true),
    ("donghua", "Chinese Animation", true),
    ("kaiju", "Kaiju Mayhem", true),
    ("boys' love (bl)", "Boys' Love", true),
    ("girls' love (gl)", "Girls' Love", true),
    ("based on manga", "Manga Adaptations", true),
    ("based on light novel", "Light Novel Adaptations", true),
    ("based on novel or book", "Book Adaptations", true),
    ("based on young adult novel", "Young Adult Adaptations", true),
    ("based on comic", "Comic Book Adaptations", true),
    ("based on video game", "Video Game Adaptations", true),
    ("lgbt", "LGBTQ Stories", true),
    ("gay theme", "LGBTQ Stories", true),
    ("romcom", "Romantic Comedies", true),
    ("heist", "Heist", false),
    ("serial killer", "Serial Killer", false),
    ("psychological thriller", "Psychological", false),
    ("film noir", "Noir", false),
    ("neo-noir", "Noir", false),
    ("time travel", "Time Travel", false),
    ("post-apocalyptic future", "Post Apocalyptic", false),
    ("dystopia", "Dystopian", false),
    ("superhero", "Superhero", false),
    ("super power", "Superpowered", false),
    ("martial arts", "Martial Arts", false),
    ("high school", "High School", false),
    ("school life", "High School", false),
    ("japanese high school", "High School", false),
    ("university", "Campus", false),
    ("revenge", "Revenge", false),
    ("spy", "Spy", false),
    ("survival", "Survival", false),
    ("zombie", "Zombie", false),
    ("vampire", "Vampire", false),
    ("witch", "Witchcraft", false),
    ("ghost", "Ghost", false),
    ("paranormal", "Paranormal", false),
    ("ghost hunter", "Ghost Hunting", false),
    ("haunted house", "Haunted House", false),
    ("haunting", "Haunting", false),
    ("demon", "Demon", false),
    ("supernatural", "Supernatural", false),
    ("monster", "Monster", false),
    ("alien", "Alien", false),
    ("alien invasion", "Alien", false),
    ("robot", "Robot", false),
    ("artificial intelligence (a.i.)", "A.I.", false),
    ("space", "Space", false),
    ("world war ii", "World War II", false),
    ("period drama", "Period", false),
    ("costume drama", "Period", false),
    ("19th century", "Period", false),
    ("historical drama", "Historical", false),
    ("war", "War", false),
    ("love triangle", "Love Triangle", false),
    ("forbidden love", "Forbidden Love", false),
    ("first love", "First Love", false),
    ("wedding", "Wedding", false),
    ("dark comedy", "Dark", false),
    ("workplace comedy", "Workplace", false),
    ("satire", "Satirical", false),
    ("parody", "Spoof", false),
    ("slapstick comedy", "Slapstick", false),
    ("detective", "Detective", false),
    ("murder mystery", "Murder Mystery", false),
    ("murder investigation", "Murder Mystery", false),
    ("murder", "Murder", false),
    ("police", "Police", false),
    ("police detective", "Police", false),
    ("fbi", "FBI", false),
    ("gangster", "Gangster", false),
    ("mafia", "Gangster", false),
    ("organized crime", "Gangster", false),
    ("prison", "Prison", false),
    ("lawyer", "Legal", false),
    ("medical drama", "Medical", false),
    ("hospital", "Medical", false),
    ("politics", "Political", false),
    ("conspiracy", "Conspiracy", false),
    ("corruption", "Corruption", false),
    ("sports", "Sports", false),
    ("football (soccer)", "Soccer", false),
    ("wrestling", "Wrestling", false),
    ("cooking", "Food", false),
    ("food", "Food", false),
    ("family relationships", "Family", false),
    ("friendship", "Friendship", false),
    ("female friendship", "Friendship", false),
    ("male friendship", "Friendship", false),
    ("biography", "Biographical", false),
    ("road trip", "Road Trip", false),
    ("new york city", "New York", false),
    ("london, england", "London", false),
    ("paris, france", "Paris", false),
    ("tokyo, japan", "Tokyo", false),
    ("los angeles, california", "Los Angeles", false),
];

fn genre_key(g: &str) -> &str {
    match g {
        "Action & Adventure" => "Action",
        "Sci-Fi & Fantasy" => "Science Fiction",
        // TMDB's TV genre is mostly political drama (Bodyguard, The Politician's Husband), not war
        "War & Politics" => "Politics",
        "Kids" => "Family",
        other => other,
    }
}

/// The title ready noun of a genre as TMDB names it ("Science Fiction" gives "Sci-Fi"). None for genres that do not make a row name.
pub fn genre_noun(g: &str) -> Option<&'static str> {
    noun(genre_key(g))
}

/// English name of an original language code, for row names such as "New in Korean".
pub fn language_label(code: &str) -> Option<&'static str> {
    Some(match code {
        "en" => "English", "es" => "Spanish", "fr" => "French", "de" => "German", "it" => "Italian", "pt" => "Portuguese", "ru" => "Russian",
        "ja" => "Japanese", "ko" => "Korean", "zh" | "cn" => "Chinese", "ar" => "Arabic", "hi" => "Hindi", "tr" => "Turkish", "nl" => "Dutch",
        "pl" => "Polish", "sv" => "Swedish", "da" => "Danish", "no" | "nb" => "Norwegian", "th" => "Thai", "id" => "Indonesian", "ta" => "Tamil",
        "te" => "Telugu", "fi" => "Finnish", "he" => "Hebrew", "el" => "Greek", "cs" => "Czech", "hu" => "Hungarian", "uk" => "Ukrainian",
        "fa" => "Persian", "vi" => "Vietnamese", "ro" => "Romanian", "ml" => "Malayalam", "bn" => "Bengali", "tl" => "Filipino",
        _ => return None,
    })
}

/// Plural, title ready noun for a genre. None for genres that do not combine with a theme.
fn noun(g: &str) -> Option<&'static str> {
    Some(match g {
        "Action" => "Action",
        "Adventure" => "Adventures",
        "Animation" => "Animation",
        "Comedy" => "Comedies",
        "Crime" => "Crime Stories",
        "Documentary" => "Documentaries",
        "Drama" => "Dramas",
        "Family" => "Family Favorites",
        "Fantasy" => "Fantasy",
        "History" => "Historical Dramas",
        "Horror" => "Horror",
        "Music" => "Musicals",
        "Mystery" => "Mysteries",
        "Romance" => "Romances",
        "Science Fiction" => "Sci-Fi",
        "Thriller" => "Thrillers",
        "War" => "War Stories",
        "Politics" => "Political Dramas",
        "Western" => "Westerns",
        _ => return None,
    })
}

fn demonym(country: &str) -> Option<&'static str> {
    Some(match country {
        "JP" => "Japanese", "KR" => "Korean", "CN" => "Chinese", "HK" => "Hong Kong", "TW" => "Taiwanese", "TH" => "Thai",
        "IN" => "Indian", "ID" => "Indonesian", "PH" => "Filipino", "VN" => "Vietnamese", "TR" => "Turkish",
        "GB" => "British", "IE" => "Irish", "FR" => "French", "DE" => "German", "IT" => "Italian", "ES" => "Spanish",
        "PT" => "Portuguese", "NL" => "Dutch", "BE" => "Belgian", "SE" => "Swedish", "DK" => "Danish", "NO" => "Norwegian",
        "FI" => "Finnish", "PL" => "Polish", "RU" => "Russian", "UA" => "Ukrainian", "GR" => "Greek", "CZ" => "Czech",
        "HU" => "Hungarian", "RO" => "Romanian", "MX" => "Mexican", "BR" => "Brazilian", "AR" => "Argentine",
        "CO" => "Colombian", "CL" => "Chilean", "CA" => "Canadian", "AU" => "Australian", "NZ" => "New Zealand",
        "ZA" => "South African", "NG" => "Nigerian", "EG" => "Egyptian", "IL" => "Israeli", "IR" => "Iranian",
        _ => return None,
    })
}

fn decade_label(decade: i32) -> String {
    if decade >= 2000 { format!("{decade}s") } else { format!("{}s", decade % 100) }
}

/// Languages that fit a country. A country that is not listed accepts any language.
fn fits_language(country: &str, language: &str) -> bool {
    let langs: &[&str] = match country {
        "JP" => &["ja"], "KR" => &["ko"], "CN" => &["zh", "cn"], "HK" => &["zh", "cn"], "TW" => &["zh", "cn"], "TH" => &["th"],
        "IN" => &["hi", "ta", "te", "ml", "kn", "bn", "mr", "pa", "en"], "ID" => &["id"], "PH" => &["tl", "en"], "VN" => &["vi"], "TR" => &["tr"],
        "FR" => &["fr"], "DE" => &["de"], "IT" => &["it"], "ES" => &["es", "ca", "gl", "eu"], "PT" => &["pt"], "NL" => &["nl"], "BE" => &["nl", "fr"],
        "SE" => &["sv"], "DK" => &["da"], "NO" => &["no", "nb", "nn"], "FI" => &["fi"], "PL" => &["pl"], "RU" => &["ru"], "UA" => &["uk", "ru"],
        "GR" => &["el"], "CZ" => &["cs"], "HU" => &["hu"], "RO" => &["ro"], "MX" => &["es"], "BR" => &["pt"], "AR" => &["es"], "CO" => &["es"], "CL" => &["es"],
        "IL" => &["he"], "IR" => &["fa"], "EG" => &["ar"],
        "GB" | "IE" | "CA" | "AU" | "NZ" | "ZA" => &["en"],
        _ => return true,
    };
    language.is_empty() || langs.contains(&language)
}

/// The one country that names a title: the first listed country, if the title's original language fits it.
pub fn primary_country(it: &CatalogItem) -> Option<&str> {
    let first = it.countries.first()?;
    fits_language(first, &it.original_language).then_some(first.as_str())
}

/// "found family" with a row of mostly dramas becomes "Found Family Dramas"; with no genre noun, "Found Family Stories". None for a theme that is
/// empty, longer than three words, or too generic to name a row alone.
/// A theme in front of a genre noun, unless the two repeat a word ("Found Family Family Favorites").
/// Singular of a plural English word, enough to tell "mystery" from "mysteries" and "drama" from "dramas".
fn stem(w: &str) -> String {
    let w = w.to_lowercase();
    if let Some(b) = w.strip_suffix("ies") { format!("{b}y") } else if let Some(b) = w.strip_suffix("es") { b.to_string() } else if let Some(b) = w.strip_suffix('s') { b.to_string() } else { w }
}

fn themed(theme: &str, noun: &str) -> Option<String> {
    let first = stem(noun.split_whitespace().next().unwrap_or(""));
    if theme.split_whitespace().any(|w| stem(w) == first) {
        return None;
    }
    Some(format!("{theme} {noun}"))
}

pub fn theme_title(theme: &str, items: &[&CatalogItem]) -> Option<String> {
    let words: Vec<&str> = theme.split_whitespace().collect();
    if words.is_empty() || words.len() > 3 || matches!(theme, "entertainment" | "comedy" | "drama" | "humor" | "action" | "family") {
        return None;
    }
    let mut count: HashMap<&str, usize> = HashMap::new();
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
    let mut count: HashMap<&str, usize> = HashMap::new();
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
    let mut gcount: HashMap<&str, usize> = HashMap::new();
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
    let mut ccount: HashMap<&str, usize> = HashMap::new();
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
        .max_by_key(|(_, c)| *c)
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
    let mut dcount: HashMap<i32, usize> = HashMap::new();
    let mut dtotal = 0usize;
    for it in items {
        if let Some(y) = it.release_year {
            *dcount.entry(y / 10 * 10).or_insert(0) += 1;
            dtotal += 1;
        }
    }
    let decade = dcount
        .into_iter()
        .max_by_key(|(_, c)| *c)
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

/// Nationality and language words that make a promise about where titles come from: (word, origin country codes, original language codes). English and Greek are left out:
/// English is the language of most titles from every country, and "Greek" mostly names myths.
const ORIGIN_WORDS: &[(&str, &[&str], &[&str])] = &[
    ("american", &["US"], &[]), ("british", &["GB"], &[]), ("irish", &["IE"], &[]), ("canadian", &["CA"], &[]), ("australian", &["AU"], &[]),
    ("korean", &["KR"], &["ko"]), ("japanese", &["JP"], &["ja"]), ("chinese", &["CN", "HK", "TW"], &["zh", "cn"]), ("taiwanese", &["TW"], &[]),
    ("thai", &["TH"], &["th"]), ("indian", &["IN"], &["hi", "ta", "te", "ml", "kn", "bn", "mr", "pa"]), ("bollywood", &["IN"], &["hi"]),
    ("french", &["FR"], &["fr"]), ("german", &["DE"], &["de"]), ("italian", &["IT"], &["it"]), ("spanish", &["ES"], &["es"]),
    ("portuguese", &["PT", "BR"], &["pt"]), ("brazilian", &["BR"], &["pt"]), ("mexican", &["MX"], &["es"]), ("turkish", &["TR"], &["tr"]),
    ("russian", &["RU"], &["ru"]), ("swedish", &["SE"], &["sv"]), ("danish", &["DK"], &["da"]), ("norwegian", &["NO"], &["no", "nb", "nn"]),
    ("dutch", &["NL"], &["nl"]), ("polish", &["PL"], &["pl"]),
];

/// Does a written row title keep its promise? Each genre word in it (comedy, crime, war ...) must be a genre of at least 40 percent of the items shown, a nationality or language
/// word ("Korean", "British") must be the origin of at least 40 percent, and a leading decade
/// ("90s", "1980s") must hold at least 60 percent of their release years. Used before a name written for a group is shown as the row title.
pub fn name_fits(name: &str, items: &[&CatalogItem]) -> bool {
    if items.is_empty() {
        return false;
    }
    let n = items.len() as f32;
    const GENRE_WORDS: &[(&str, &str)] = &[
        ("action", "action"), ("adventure", "adventure"), ("animat", "animation"), ("cartoon", "animation"), ("anime", "animation"), ("comed", "comedy"), ("crime", "crime"), ("documentar", "documentary"),
        ("drama", "drama"), ("family", "family"), ("fantasy", "fantasy"), ("histor", "history"), ("horror", "horror"), ("music", "music"), ("mystery", "mystery"), ("mysteries", "mystery"),
        ("romanc", "romance"), ("rom-com", "romance"), ("rom-com", "comedy"), ("romcom", "romance"), ("romcom", "comedy"), ("sci-fi", "sci"), ("thriller", "thriller"), ("war", "war"), ("western", "western"), ("reality", "reality"),
    ];
    for w in name.split(|c: char| !c.is_alphabetic() && c != '-').filter(|w| !w.is_empty()) {
        let lw = w.to_lowercase();
        if lw.starts_with("warm") || lw.starts_with("warn") || lw.starts_with("ward") || lw.starts_with("warr") { continue; }
        for (k, g) in GENRE_WORDS {
            if lw.starts_with(k) {
                let have = items.iter().filter(|i| i.genres.iter().any(|x| x.to_lowercase().contains(g))).count() as f32;
                if have < 0.4 * n {
                    return false;
                }
            }
        }
    }
    // a nationality or language word ("Korean", "British", "Spanish") must be the origin of at least 40 percent of the items; two such words together count as one group
    let mut origin: Vec<&(&str, &[&str], &[&str])> = Vec::new();
    for w in name.split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty()) {
        let lw = w.to_lowercase();
        if let Some(rule) = ORIGIN_WORDS.iter().find(|r| r.0 == lw) {
            origin.push(rule);
        }
    }
    if !origin.is_empty() {
        let have = items
            .iter()
            .filter(|i| origin.iter().any(|(_, countries, langs)| primary_country(i).is_some_and(|c| countries.contains(&c)) || langs.contains(&i.original_language.as_str())))
            .count() as f32;
        if have < 0.4 * n {
            return false;
        }
    }
    // every decade named anywhere in the title ("90s", "2000s", "of the 1950s and 1960") must hold the years of the items: together they cover at least 60 percent
    let mut decades: Vec<i32> = Vec::new();
    for w in name.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
        let lw = w.to_lowercase();
        let Some(digits) = lw.strip_suffix('s').or(Some(lw.as_str())).map(|d| d.to_string()) else { continue };
        let is_decade = lw.ends_with('s') && (digits.len() == 2 || digits.len() == 4) && digits.chars().all(|c| c.is_ascii_digit());
        let bare_year = !lw.ends_with('s') && digits.len() == 4 && digits.chars().all(|c| c.is_ascii_digit()) && !name.contains(&format!("{digits}s")) && (name.to_lowercase().contains(" to ") || name.to_lowercase().contains(" and "));
        if is_decade || bare_year {
            let d: i32 = digits.parse().unwrap_or(0);
            decades.push(if digits.len() == 4 { d / 10 * 10 } else if d >= 30 { 1900 + d } else { 2000 + d });
        }
    }
    if !decades.is_empty() {
        let years: Vec<i32> = items.iter().filter_map(|i| i.release_year).collect();
        if years.is_empty() || (years.iter().filter(|y| decades.iter().any(|d| (*d..*d + 10).contains(*y))).count() as f32) < 0.6 * years.len() as f32 {
            return false;
        }
    }
    true
}

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
