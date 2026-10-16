//! Model written profiles of titles (ai-engine/profiles): audience, themes, tone, who it is about, LGBTQ+ presence and content flags, and
//!
//! File in the catalog folder: `profiles_v1.jsonl` (one `{"k":"movie:550","p":{...}}` per line). Without it the engine runs as before.
//! Profile vectors of earlier runs (`profile_vectors_v1.*`, another embedding space) are not read.
//!
//! What a page may trust: `audience` and `themes` are used. `lgbtq` "present" is not trusted (the model marks too many titles, measured
//! 143 of 1,000 on the first part, among them a late night talk show); only "central" counts.
use crate::catalog::item_key;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;


pub const KIDS: u8 = 1;
pub const FAMILY: u8 = 2;
pub const TEEN: u8 = 4;
pub const ADULT: u8 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lgbtq {
    Central,
    Present,
    None,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct TitleProfile {
    /// Bit set of the audiences the model named (KIDS, FAMILY, TEEN, ADULT)
    pub audience: u8,
    pub lgbtq: Lgbtq,
    pub themes: Vec<String>,
    pub tone: Vec<String>,
    pub content: Vec<String>,
}

impl TitleProfile {
    /// A children's or family title for a limited profile when the title has no age certificate. Measured against the real certificates of 1,525
    /// popular titles that have one: audience kids or family only (no teen, no adult named) and no heavy content flag gave 303 of 303 correct
    /// (all U or PG), recall 20 percent. Looser rules let 12 and 15 titles through (precision 0.89), so they are not used.
    pub fn suits_children(&self) -> bool {
        self.audience & (KIDS | FAMILY) != 0
            && self.audience & (TEEN | ADULT) == 0
            && !self.content.iter().any(|c| matches!(c.as_str(), "violence" | "sex" | "nudity" | "horror" | "drugs" | "abuse" | "suicide" | "strong language" | "war"))
    }

    /// Suits teenagers: no flag for sex, nudity or abuse, and the model did not name only adults.
    pub fn suits_teens(&self) -> bool {
        self.audience != ADULT && !self.content.iter().any(|c| matches!(c.as_str(), "sex" | "nudity" | "abuse"))
    }
}

/// "kids", "family, teen", ["teen", "adult"] ... to the audience bit set. Unknown words are ignored.
pub fn parse_audience(v: &Value) -> u8 {
    let words: Vec<String> = match v {
        Value::String(s) => s.split(',').map(|w| w.trim().to_ascii_lowercase()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(|w| w.trim().to_ascii_lowercase())).collect(),
        _ => Vec::new(),
    };
    words.iter().fold(0, |acc, w| {
        acc | match w.as_str() {
            "kids" | "kid" | "children" => KIDS,
            "family" => FAMILY,
            "teen" | "teens" | "teenager" | "teenagers" => TEEN,
            "adult" | "adults" => ADULT,
            _ => 0,
        }
    })
}

fn parse_lgbtq(v: Option<&Value>) -> Lgbtq {
    match v.and_then(Value::as_str).map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("central") => Lgbtq::Central,
        Some("present") => Lgbtq::Present,
        Some("none") => Lgbtq::None,
        _ => Lgbtq::Unknown,
    }
}

fn list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect(),
        Some(Value::String(s)) => s.split(',').map(|x| x.trim().to_ascii_lowercase()).filter(|x| !x.is_empty()).collect(),
        _ => Vec::new(),
    }
}

/// None when the profile is empty (the model gave no usable answer).
pub fn parse_profile(p: &Value) -> Option<TitleProfile> {
    let audience = parse_audience(p.get("audience")?);
    if audience == 0 {
        return None;
    }
    Some(TitleProfile {
        audience,
        lgbtq: parse_lgbtq(p.get("lgbtq")),
        themes: list(p.get("themes")),
        tone: list(p.get("tone")),
        content: list(p.get("content")),
    })
}

pub struct ProfileStore {
    /// internal item key -> position in `profiles`
    pub index: HashMap<u32, usize>,
    pub profiles: Vec<TitleProfile>,
    /// share of all profiles carrying each row-name tone (by display name); the baseline a row's tone is compared against
    pub tone_share: HashMap<&'static str, f32>,
    /// share of all profiles carrying each theme phrase
    pub theme_share: HashMap<String, f32>,
}

/// Tone words (as the model writes them) that make a good row name, with the word shown. Several spellings share one display name.
const TONE_NAMES: &[(&str, &str)] = &[
    ("dark", "Dark"), ("gritty", "Gritty"), ("heartwarming", "Heartwarming"), ("heartfelt", "Heartwarming"),
    ("lighthearted", "Feel-Good"), ("light-hearted", "Feel-Good"), ("whimsical", "Whimsical"), ("quirky", "Quirky"),
    ("nostalgic", "Nostalgic"), ("epic", "Epic"), ("uplifting", "Uplifting"), ("inspirational", "Inspiring"), ("inspiring", "Inspiring"),
    ("satirical", "Satirical"), ("absurd", "Absurd"), ("absurdist", "Absurd"), ("unsettling", "Unsettling"), ("romantic", "Romantic"),
    ("gentle", "Gentle"), ("playful", "Playful"), ("mysterious", "Mysterious"), ("melancholic", "Melancholy"),
    ("action-packed", "Action-Packed"), ("suspenseful", "Suspenseful"), ("tense", "Suspenseful"), ("sarcastic", "Sarcastic"),
];

fn tone_display(word: &str) -> Option<&'static str> {
    TONE_NAMES.iter().find(|(w, _)| *w == word).map(|(_, d)| *d)
}

impl ProfileStore {
    pub fn empty() -> Self {
        Self { index: HashMap::new(), profiles: Vec::new(), tone_share: HashMap::new(), theme_share: HashMap::new() }
    }

    /// The theme that sets a row apart: carried by at least `min_share` of the row's titles and at least `min_lift` times more often than across
    /// all profiles. A theme every kind of title shares ("entertainment") never qualifies. Ties go to the larger lift, then the name.
    pub fn row_theme_lift(&self, profiles: &[&TitleProfile], min_share: f32, min_lift: f32) -> Option<(String, f32)> {
        if profiles.is_empty() {
            return None;
        }
        let mut count: HashMap<&str, usize> = HashMap::new();
        for p in profiles {
            let mut seen: Vec<&str> = Vec::new();
            for t in &p.themes {
                if !seen.contains(&t.as_str()) {
                    seen.push(t);
                    *count.entry(t).or_insert(0) += 1;
                }
            }
        }
        count
            .into_iter()
            .filter_map(|(th, n)| {
                let share = n as f32 / profiles.len() as f32;
                let base = self.theme_share.get(th).copied().unwrap_or(0.0).max(1.0 / (self.profiles.len().max(1) as f32));
                (share >= min_share && share / base >= min_lift).then(|| (th.to_string(), share, share / base))
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)))
            .map(|(th, share, _)| (th, share))
    }

    /// The tone that sets a row apart: the display tone carried by at least `min_share` of the row's titles and at least `min_lift` times
    /// more often than across all profiles. None when no tone stands out (common tones such as "dramatic" never qualify).
    pub fn row_tone(&self, profiles: &[&TitleProfile], min_share: f32, min_lift: f32) -> Option<&'static str> {
        if profiles.is_empty() {
            return None;
        }
        let mut count: HashMap<&'static str, usize> = HashMap::new();
        for p in profiles {
            let mut seen: Vec<&'static str> = Vec::new();
            for w in &p.tone {
                if let Some(d) = tone_display(w) {
                    if !seen.contains(&d) {
                        seen.push(d);
                        *count.entry(d).or_insert(0) += 1;
                    }
                }
            }
        }
        count
            .into_iter()
            .filter_map(|(d, n)| {
                let share = n as f32 / profiles.len() as f32;
                let base = self.tone_share.get(d).copied().unwrap_or(0.0).max(0.005);
                (share >= min_share && share / base >= min_lift).then_some((d, share / base))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(a.0)))
            .map(|(d, _)| d)
    }

    /// True when title profiles are loaded (with or without their vectors)
    pub fn has_profiles(&self) -> bool { !self.profiles.is_empty() }

    pub fn get(&self, key: u32) -> Option<&TitleProfile> {
        self.index.get(&key).map(|&i| &self.profiles[i])
    }

    /// Reads the profile files from the catalog folder. Missing files give an empty store.
    pub fn load(dir: &str) -> Self {
        let dir = Path::new(dir);
        let Ok(text) = std::fs::read_to_string(dir.join("profiles_v1.jsonl")) else { return Self::empty() };
        let mut store = Self::empty();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(row) = serde_json::from_str::<Value>(line) else { continue };
            let (Some(k), Some(p)) = (row.get("k").and_then(Value::as_str), row.get("p")) else { continue };
            let (Some(profile), Some((media, id))) = (parse_profile(p), k.split_once(':')) else { continue };
            let Ok(id) = id.parse::<u32>() else { continue };
            store.index.insert(item_key(media, id), store.profiles.len());
            store.profiles.push(profile);
        }
        let mut tone_count: HashMap<&'static str, usize> = HashMap::new();
        for p in &store.profiles {
            let mut seen: Vec<&'static str> = Vec::new();
            for w in &p.tone {
                if let Some(d) = tone_display(w) {
                    if !seen.contains(&d) {
                        seen.push(d);
                        *tone_count.entry(d).or_insert(0) += 1;
                    }
                }
            }
        }
        let total = store.profiles.len().max(1) as f32;
        let mut theme_count: HashMap<String, usize> = HashMap::new();
        for p in &store.profiles {
            let mut seen: Vec<&str> = Vec::new();
            for th in &p.themes {
                if !seen.contains(&th.as_str()) {
                    seen.push(th);
                    *theme_count.entry(th.clone()).or_insert(0) += 1;
                }
            }
        }
        store.theme_share = theme_count.into_iter().map(|(th, n)| (th, n as f32 / total)).collect();
        store.tone_share = tone_count.into_iter().map(|(d, n)| (d, n as f32 / total)).collect();
        store
    }
}

/// The theme that names a row: the theme phrase found in the largest share of the row's titles, with that share. None when no phrase
/// reaches `min_share` (then the row has no single honest theme name).
pub fn row_theme(profiles: &[&TitleProfile], min_share: f32) -> Option<(String, f32)> {
    if profiles.is_empty() {
        return None;
    }
    let mut count: HashMap<&str, usize> = HashMap::new();
    for p in profiles {
        let mut seen: Vec<&str> = Vec::new();
        for t in &p.themes {
            if !seen.contains(&t.as_str()) {
                seen.push(t);
                *count.entry(t).or_insert(0) += 1;
            }
        }
    }
    let (theme, n) = count.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0)))?;
    let share = n as f32 / profiles.len() as f32;
    (share >= min_share).then(|| (theme.to_string(), share))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn audience_words_become_a_set() {
        assert_eq!(parse_audience(&json!("kids, family")), KIDS | FAMILY);
        assert_eq!(parse_audience(&json!(["teen", "Adult"])), TEEN | ADULT);
        assert_eq!(parse_audience(&json!("adult")), ADULT);
        assert_eq!(parse_audience(&json!("nonsense")), 0);
        assert_eq!(parse_audience(&json!(null)), 0);
    }

    #[test]
    fn an_empty_or_unreadable_profile_is_dropped() {
        assert!(parse_profile(&json!({})).is_none());
        assert!(parse_profile(&json!({"audience": "grown ups"})).is_none());
        let p = parse_profile(&json!({"audience": "family", "lgbtq": "present", "themes": ["Friendship ", "magic"], "content": ["humor"]})).unwrap();
        assert_eq!(p.themes, vec!["friendship", "magic"]);
        assert_eq!(p.lgbtq, Lgbtq::Present);
    }

    #[test]
    fn children_means_a_child_audience_without_heavy_content() {
        let ok = parse_profile(&json!({"audience": "kids, family", "content": ["humor"]})).unwrap();
        let violent = parse_profile(&json!({"audience": "kids, family", "content": ["violence"]})).unwrap();
        let mixed = parse_profile(&json!({"audience": "family, adult", "content": []})).unwrap();
        let teen = parse_profile(&json!({"audience": "family, teen", "content": []})).unwrap();
        assert!(ok.suits_children() && !violent.suits_children() && !mixed.suits_children() && !teen.suits_children());
    }

    #[test]
    fn teens_exclude_adult_only_and_sexual_content() {
        let adult_only = parse_profile(&json!({"audience": "adult"})).unwrap();
        let teen = parse_profile(&json!({"audience": "teen, adult", "content": ["violence"]})).unwrap();
        let sexual = parse_profile(&json!({"audience": "teen, adult", "content": ["sex"]})).unwrap();
        assert!(!adult_only.suits_teens() && teen.suits_teens() && !sexual.suits_teens());
    }

    #[test]
    fn a_row_theme_needs_enough_titles_to_carry_it() {
        let mk = |themes: &[&str]| parse_profile(&json!({"audience": "adult", "themes": themes})).unwrap();
        let a = mk(&["found family", "grief"]);
        let b = mk(&["found family", "loss"]);
        let c = mk(&["revenge"]);
        let row = [&a, &b, &c];
        let (name, share) = row_theme(&row, 0.6).unwrap();
        assert_eq!(name, "found family");
        assert!((share - 2.0 / 3.0).abs() < 1e-6);
        assert!(row_theme(&row, 0.8).is_none());
        assert!(row_theme(&[], 0.5).is_none());
    }

    #[test]
    fn missing_files_give_an_empty_store() {
        let s = ProfileStore::load("/definitely/not/here");
        assert!(s.profiles.is_empty() && s.get(1).is_none());
    }

    #[test]
    fn a_tone_must_stand_out_to_name_a_row() {
        let mk = |tone: &[&str]| parse_profile(&json!({"audience": "adult", "tone": tone})).unwrap();
        let mut store = ProfileStore::empty();
        store.tone_share.insert("Dark", 0.13);
        store.tone_share.insert("Feel-Good", 0.13);
        let dark: Vec<TitleProfile> = (0..10).map(|_| mk(&["dark", "dramatic"])).collect();
        let refs: Vec<&TitleProfile> = dark.iter().collect();
        assert_eq!(store.row_tone(&refs, 0.35, 2.5), Some("Dark"));
        // a tone every title carries is not distinctive
        store.tone_share.insert("Dark", 0.9);
        assert_eq!(store.row_tone(&refs, 0.35, 2.5), None);
        // "dramatic" is not a naming tone
        let plain: Vec<TitleProfile> = (0..10).map(|_| mk(&["dramatic"])).collect();
        let plain_refs: Vec<&TitleProfile> = plain.iter().collect();
        assert_eq!(store.row_tone(&plain_refs, 0.35, 2.5), None);
    }
}
