//! Words for row titles: genre nouns, nationality words, decade labels, and the facts of a title they rest on.
use super::*;

pub(crate) fn genre_key(g: &str) -> &str {
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
pub(crate) fn noun(g: &str) -> Option<&'static str> {
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

pub(crate) fn demonym(country: &str) -> Option<&'static str> {
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

pub(crate) fn decade_label(decade: i32) -> String {
    if decade >= 2000 { format!("{decade}s") } else { format!("{}s", decade % 100) }
}

/// Languages that fit a country. A country that is not listed accepts any language.
pub(crate) fn fits_language(country: &str, language: &str) -> bool {
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
pub(crate) fn stem(w: &str) -> String {
    let w = w.to_lowercase();
    if let Some(b) = w.strip_suffix("ies") { format!("{b}y") } else if let Some(b) = w.strip_suffix("es") { b.to_string() } else if let Some(b) = w.strip_suffix('s') { b.to_string() } else { w }
}

pub(crate) fn themed(theme: &str, noun: &str) -> Option<String> {
    let first = stem(noun.split_whitespace().next().unwrap_or(""));
    if theme.split_whitespace().any(|w| stem(w) == first) {
        return None;
    }
    Some(format!("{theme} {noun}"))
}
