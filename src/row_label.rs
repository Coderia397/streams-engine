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

/// Plural, title ready noun for a genre. None for genres that do not combine with a theme.
fn noun(g: &str) -> Option<&'static str> {
    Some(match g {
        "Action" => "Action",
        "Adventure" => "Adventures",
        "Animation" => "Animation",
        "Comedy" => "Comedies",
        "Crime" => "Crime Stories",
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

/// Title candidates for a row, most specific first. Empty when nothing specific can be said.
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
    let mut ccount: HashMap<&str, usize> = HashMap::new();
    for it in items {
        for c in &it.countries {
            if c != "US" {
                *ccount.entry(c.as_str()).or_insert(0) += 1;
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
    themes.sort_by(|a, b| b.2.cmp(&a.2));

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
            if let Some(g) = gnoun { push(format!("{} {}", t.0, g)); }
        }
        push("Anime Favorites".to_string());
    }
    if reality {
        for t in themes.iter().filter(|t| t.1) {
            push(t.0.to_string());
        }
        push(match top_genre { Some("Comedy") => "Funny Reality Shows".to_string(), _ => "Reality Shows".to_string() });
    }
    if let (Some(c), Some(t)) = (country_adj, themes.iter().find(|t| !t.1)) {
        if let Some(g) = gnoun { push(format!("{c} {} {g}", t.0)); }
    }
    for t in themes.iter().filter(|t| !t.1) {
        if let Some(g) = gnoun { push(format!("{} {}", t.0, g)); }
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
    let old_share = share(items.iter().filter(|i| i.release_year.map_or(false, |y| y < 1980)).count());
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(genres: &[&str], countries: &[&str], keywords: &[&str], year: i32) -> CatalogItem {
        CatalogItem {
            id: 1, tmdb_id: 1, media_type: "tv".into(), title: "t".into(), release_year: Some(year),
            genres: genres.iter().map(|s| s.to_string()).collect(), overview: String::new(), tagline: None, director: None,
            poster_path: None, backdrop_path: None, vote_average: 7.0, popularity: 10.0, is_anime: false, vote_count: 0,
            adult: false, original_language: "en".into(), tv_type: None,
            countries: countries.iter().map(|s| s.to_string()).collect(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(), runtime: 0,
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
}
