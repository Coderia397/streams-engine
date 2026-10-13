//! Model written profiles of titles (ai-engine/profiles): audience, themes, tone, who it is about, LGBTQ+ presence and content flags, and
//! the vector of each profile (Qwen3-Embedding-4B, the model of the catalog vectors, float16, 2,560 numbers).
//!
//! Files in the catalog folder: `profiles_v1.jsonl` (one `{"k":"movie:550","p":{...}}` per line), `profile_vectors_v1.bin` (float16, in the
//! order of `profile_vectors_v1.keys`, one key per line). Without them the engine runs as before.
//!
//! What a page may trust: `audience` and `themes` are used. `lgbtq` "present" is not trusted (the model marks too many titles, measured
//! 143 of 1,000 on the first part, among them a late night talk show); only "central" counts.
use crate::catalog::item_key;
use half::f16;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub const DIMS: usize = 2560;

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
    pub summary: String,
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
        summary: p.get("summary").and_then(Value::as_str).unwrap_or("").to_string(),
    })
}

pub struct ProfileStore {
    /// internal item key -> position in `profiles` and `vectors`
    pub index: HashMap<u32, usize>,
    pub profiles: Vec<TitleProfile>,
    /// float16 bits, `DIMS` per profile; empty when the vectors file is missing
    pub vectors: Vec<u16>,
    /// share of all profiles carrying each row-name tone (by display name); the baseline a row's tone is compared against
    pub tone_share: HashMap<&'static str, f32>,
    /// share of all profiles carrying each theme phrase
    pub theme_share: HashMap<String, f32>,
    /// internal item key of each profile, in the order of `profiles` and `vectors`
    pub keys: Vec<u32>,
    /// Mean of all profile vectors, each vector's dot product with it, and the length of each vector after the mean is removed, so the
    /// centred cosine of two profiles needs one dot product (raw pairs share a high baseline cosine, centred ones do not)
    pub mean: Vec<f32>,
    pub dot_mean: Vec<f32>,
    pub norm_centred: Vec<f32>,
    mean_sq: f32,
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
        Self { index: HashMap::new(), profiles: Vec::new(), vectors: Vec::new(), tone_share: HashMap::new(), theme_share: HashMap::new(), keys: Vec::new(), mean: Vec::new(), dot_mean: Vec::new(), norm_centred: Vec::new(), mean_sq: 0.0 }
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

    pub fn get(&self, key: u32) -> Option<&TitleProfile> {
        self.index.get(&key).map(|&i| &self.profiles[i])
    }

    /// Computes the mean vector and the per vector numbers behind `nearest`.
    fn prepare_centring(&mut self) {
        use rayon::prelude::*;
        let n = self.vectors.len() / DIMS;
        if n == 0 {
            return;
        }
        let mut mean = vec![0f64; DIMS];
        for row in self.vectors.chunks_exact(DIMS) {
            for (m, &b) in mean.iter_mut().zip(row) {
                *m += f16::from_bits(b).to_f32() as f64;
            }
        }
        let mean: Vec<f32> = mean.into_iter().map(|m| (m / n as f64) as f32).collect();
        let mean_sq: f32 = mean.iter().map(|m| m * m).sum();
        let stats: Vec<(f32, f32)> = self
            .vectors
            .par_chunks_exact(DIMS)
            .map(|row| {
                let (mut dm, mut sq) = (0f32, 0f32);
                for (&b, &m) in row.iter().zip(&mean) {
                    let x = f16::from_bits(b).to_f32();
                    dm += x * m;
                    sq += x * x;
                }
                // |x - m|^2 = |x|^2 - 2 x.m + m.m
                (dm, (sq - 2.0 * dm + mean_sq).max(1e-12).sqrt())
            })
            .collect();
        self.dot_mean = stats.iter().map(|s| s.0).collect();
        self.norm_centred = stats.iter().map(|s| s.1).collect();
        self.mean = mean;
        self.mean_sq = mean_sq;
    }

    /// The profiles most like the one of `key` by centred cosine, best first, the title itself left out. Empty when it has no profile vector.
    pub fn nearest(&self, key: u32, k: usize) -> Vec<(u32, f32)> {
        use rayon::prelude::*;
        let Some(&pos) = self.index.get(&key) else { return Vec::new() };
        if self.norm_centred.len() != self.profiles.len() || self.vectors.len() < (pos + 1) * DIMS {
            return Vec::new();
        }
        let a: Vec<f32> = self.vectors[pos * DIMS..(pos + 1) * DIMS].iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        let (am, an) = (self.dot_mean[pos], self.norm_centred[pos]);
        let mut scored: Vec<(u32, f32)> = self
            .vectors
            .par_chunks_exact(DIMS)
            .enumerate()
            .filter(|(i, _)| *i != pos)
            .map(|(i, row)| {
                let dot: f32 = row.iter().zip(&a).map(|(&b, &x)| f16::from_bits(b).to_f32() * x).sum();
                (self.keys[i], (dot - am - self.dot_mean[i] + self.mean_sq) / (an * self.norm_centred[i]))
            })
            .collect();
        scored.sort_unstable_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0)));
        scored.truncate(k);
        scored
    }

    /// Centred cosine of the profiles of two titles, None when either has no profile vector.
    pub fn centred_cos(&self, a: u32, b: u32) -> Option<f32> {
        let (&pa, &pb) = (self.index.get(&a)?, self.index.get(&b)?);
        if self.norm_centred.len() != self.profiles.len() || self.vectors.len() < (pa.max(pb) + 1) * DIMS {
            return None;
        }
        let dot: f32 = self.vectors[pa * DIMS..(pa + 1) * DIMS].iter().zip(&self.vectors[pb * DIMS..(pb + 1) * DIMS]).map(|(&x, &y)| f16::from_bits(x).to_f32() * f16::from_bits(y).to_f32()).sum();
        Some((dot - self.dot_mean[pa] - self.dot_mean[pb] + self.mean_sq) / (self.norm_centred[pa] * self.norm_centred[pb]))
    }

    pub fn vector(&self, key: u32) -> Option<Vec<f32>> {
        let i = *self.index.get(&key)?;
        let s = self.vectors.get(i * DIMS..(i + 1) * DIMS)?;
        Some(s.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
    }

    /// Reads the profile files from the catalog folder. Missing files give an empty store.
    pub fn load(dir: &str) -> Self {
        let dir = Path::new(dir);
        let Ok(text) = std::fs::read_to_string(dir.join("profiles_v1.jsonl")) else { return Self::empty() };
        let keys: Vec<String> = std::fs::read_to_string(dir.join("profile_vectors_v1.keys")).map(|t| t.lines().map(str::to_string).collect()).unwrap_or_default();
        let bytes = std::fs::read(dir.join("profile_vectors_v1.bin")).unwrap_or_default();
        let vector_rows = if bytes.len() == keys.len() * DIMS * 2 { keys.len() } else { 0 };
        let vector_pos: HashMap<&str, usize> = keys.iter().enumerate().take(vector_rows).map(|(i, k)| (k.as_str(), i)).collect();
        let mut store = Self::empty();
        let mut vectors: Vec<u16> = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(row) = serde_json::from_str::<Value>(line) else { continue };
            let (Some(k), Some(p)) = (row.get("k").and_then(Value::as_str), row.get("p")) else { continue };
            let (Some(profile), Some((media, id))) = (parse_profile(p), k.split_once(':')) else { continue };
            let Ok(id) = id.parse::<u32>() else { continue };
            let Some(&pos) = vector_pos.get(k) else { continue };
            store.index.insert(item_key(media, id), store.profiles.len());
            store.keys.push(item_key(media, id));
            store.profiles.push(profile);
            vectors.extend((0..DIMS).map(|d| u16::from_le_bytes([bytes[(pos * DIMS + d) * 2], bytes[(pos * DIMS + d) * 2 + 1]])));
        }
        store.vectors = vectors;
        store.prepare_centring();
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
        assert!(s.profiles.is_empty() && s.vector(1).is_none());
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
