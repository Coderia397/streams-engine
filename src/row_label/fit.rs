//! Does a written row title keep its promise? The check a name for a group has to pass before it is shown.
use super::*;

/// Nationality and language words that make a promise about where titles come from: (word, origin country codes, original language codes). English and Greek are left out:
/// English is the language of most titles from every country, and "Greek" mostly names myths.
pub(crate) const ORIGIN_WORDS: &[(&str, &[&str], &[&str])] = &[
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
