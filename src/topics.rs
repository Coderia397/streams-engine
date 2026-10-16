//! Topic rows: rows named for what a title is about (Courtroom & Legal Dramas, Heists & Capers, Found Family ...), from the memberships the Colab judge accepted
//! (`topics_v1.json`, kept as `topic:<slug>` keywords and indexed in the catalog) and the names and media kinds of `topics_meta.json` in the catalog folder.
//! Without those files no topic row is built. A page gets a few of them, chosen by the session and the six hour slot, only from topics that fill a row on that page.
use crate::page::{entry_noise, PageContext, Row};
use serde_json::json;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Topic {
    pub name: String,
    /// One sentence of what belongs in the topic (the words of a search are matched against it too)
    pub definition: String,
    /// `movie`, `tv` or both
    pub media: Vec<String>,
}

#[derive(Debug, Default)]
pub struct TopicStore {
    pub topics: HashMap<String, Topic>,
    /// word -> in how many topic texts (name and definition) it occurs, for weighting query words by how telling they are
    df: HashMap<String, u32>,
}

impl TopicStore {
    pub fn load(dir: &str) -> Self {
        let Some(raw) = std::fs::read_to_string(std::path::Path::new(dir).join("topics_meta.json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { return Self::default() };
        let mut topics = HashMap::new();
        for (slug, v) in raw.as_object().into_iter().flatten() {
            let Some(name) = v.get("name").and_then(|n| n.as_str()) else { continue };
            let media = v.get("media").and_then(|m| m.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let definition = v.get("definition").and_then(|d| d.as_str()).unwrap_or_default().to_string();
            topics.insert(slug.clone(), Topic { name: name.to_string(), definition, media });
        }
        let mut df: HashMap<String, u32> = HashMap::new();
        for t in topics.values() {
            let words: std::collections::HashSet<String> = words_of(&format!("{} {}", t.name, t.definition)).into_iter().collect();
            for w in words { *df.entry(w).or_insert(0) += 1; }
        }
        Self { topics, df }
    }

    /// The topics a free text search describes ("movies about space", "scary", "something funny"), best first with their score: each query word that starts a word of the topic's name
    /// counts twice, one in the definition once, each weighted by how rare the word is among the topics (so "movies" or "about" count for almost nothing). A topic needs a score of at
    /// least `ln(topics + 1)`, which one telling word in the name reaches and a common word does not.
    pub fn match_query(&self, query: &str) -> Vec<(&str, f32)> {
        let n = self.topics.len().max(1) as f32;
        // a telling word in a name (rarer than about 1 in 8 topic texts) is enough; the bar grows with the number of topics
        let min_score = (n + 1.0).ln();
        let qw: Vec<String> = words_of(query);
        if qw.is_empty() { return Vec::new(); }
        let idf = |w: &str| -> f32 { ((n + 1.0) / (1.0 + *self.df.get(w).unwrap_or(&0) as f32)).ln().max(0.0) };
        let mut out: Vec<(&str, f32)> = self
            .topics
            .iter()
            .filter_map(|(slug, t)| {
                let name_words = words_of(&t.name);
                let def_words = words_of(&t.definition);
                let mut score = 0.0f32;
                for q in &qw {
                    let in_name = name_words.iter().any(|w| w.starts_with(q.as_str()) || q.starts_with(w.as_str()) && w.len() >= 4);
                    let in_def = def_words.iter().any(|w| w == q);
                    if in_name { score += 2.0 * idf(q); } else if in_def { score += idf(q); }
                }
                (score >= min_score).then_some((slug.as_str(), score))
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(b.0)));
        out
    }
}

/// Lower case words of a text with a light plural stem, words of three letters or more.
fn words_of(text: &str) -> Vec<String> {
    crate::people::normalise(text)
        .split(' ')
        .filter(|w| w.chars().count() >= 3)
        .map(|w| w.strip_suffix('s').filter(|s| s.chars().count() >= 3).unwrap_or(w).to_string())
        .collect()
}

impl PageContext<'_> {
    /// Up to `want` topic rows for this page, none whose title is in `skip_titles`; each is at least 12 titles that no earlier row of the page has.
    pub(crate) fn build_topic_rows(&self, taken: &HashSet<u32>, want: usize, skip_titles: &HashSet<String>) -> Vec<Row> {
        if want == 0 || self.topics.topics.is_empty() { return Vec::new(); }
        let mut order: Vec<(&String, &Topic)> = self.topics.topics.iter().collect();
        order.sort_by(|a, b| entry_noise(self.session, 31_000 + b.0.len() * 7 + b.0.bytes().map(|x| x as usize).sum::<usize>()).partial_cmp(&entry_noise(self.session, 31_000 + a.0.len() * 7 + a.0.bytes().map(|x| x as usize).sum::<usize>())).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new();
        let mut used = taken.clone();
        for (slug, topic) in order {
            if rows.len() >= want { break; }
            if skip_titles.contains(&topic.name.to_lowercase()) { continue; }
            if self.surface != "both" && !topic.media.iter().any(|m| m == self.surface) { continue; }
            let Some(keys) = self.catalog.topic_keys.get(slug) else { continue };
            let items = self.eligible(keys, &used, false);
            if items.len() < 12 { continue; }
            if let Some(mut r) = self.row_of(&topic.name, "topic", &items) {
                r.meta = Some(json!({ "title_key": format!("topic.{slug}"), "title_args": {}, "reason": { "kind": "topic", "topic": slug } }));
                r.items.iter().for_each(|i| { used.insert(i.id); });
                rows.push(r);
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_names_and_media_are_read_from_the_meta_file() {
        let dir = std::env::temp_dir().join(format!("streams-topics-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("topics_meta.json"), r#"{"heists": {"name": "Heists & Capers", "media": ["movie", "tv"]}, "k_drama": {"name": "K-dramas", "media": ["tv"]}, "broken": {"media": []}}"#).unwrap();
        let store = TopicStore::load(dir.to_str().unwrap());
        assert_eq!(store.topics.len(), 2, "an entry without a name is skipped");
        assert_eq!(store.topics["heists"].name, "Heists & Capers");
        assert_eq!(store.topics["k_drama"].media, vec!["tv".to_string()]);
        assert_eq!(store.topics["heists"].definition, "");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_free_text_query_finds_the_topic_that_describes_it() {
        let mut topics = HashMap::new();
        for (slug, name, def) in [("space", "Space & Aliens", "A story set in space or about aliens"), ("heists", "Heists & Capers", "A story about a robbery"), ("legal", "Courtroom Dramas", "A drama about lawyers and trials"), ("a", "Family Stories", "A story about family"), ("b", "War Stories", "A story about war")] {
            topics.insert(slug.to_string(), Topic { name: name.into(), definition: def.into(), media: vec![] });
        }
        let mut df = HashMap::new();
        for t in topics.values() { for w in words_of(&format!("{} {}", t.name, t.definition)).into_iter().collect::<std::collections::HashSet<_>>() { *df.entry(w).or_insert(0) += 1; } }
        let store = TopicStore { topics, df };
        assert_eq!(store.match_query("movies about space").first().map(|m| m.0), Some("space"));
        assert!(store.match_query("movies about").is_empty(), "common words alone name no topic");
        assert_eq!(store.match_query("courtroom").first().map(|m| m.0), Some("legal"));
    }

    #[test]
    fn a_missing_file_gives_an_empty_store() {
        assert!(TopicStore::load("/definitely/not/here").topics.is_empty());
    }
}
