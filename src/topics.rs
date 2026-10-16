//! Topic rows: rows named for what a title is about (Courtroom & Legal Dramas, Heists & Capers, Found Family ...), from the memberships the Colab judge accepted
//! (`topics_v1.json`, kept as `topic:<slug>` keywords and indexed in the catalog) and the names and media kinds of `topics_meta.json` in the catalog folder.
//! Without those files no topic row is built. A page gets a few of them, chosen by the session and the six hour slot, only from topics that fill a row on that page.
use crate::page::{entry_noise, PageContext, Row};
use serde_json::json;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Topic {
    pub name: String,
    /// `movie`, `tv` or both
    pub media: Vec<String>,
}

#[derive(Debug, Default)]
pub struct TopicStore {
    pub topics: HashMap<String, Topic>,
}

impl TopicStore {
    pub fn load(dir: &str) -> Self {
        let Some(raw) = std::fs::read_to_string(std::path::Path::new(dir).join("topics_meta.json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { return Self::default() };
        let mut topics = HashMap::new();
        for (slug, v) in raw.as_object().into_iter().flatten() {
            let Some(name) = v.get("name").and_then(|n| n.as_str()) else { continue };
            let media = v.get("media").and_then(|m| m.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            topics.insert(slug.clone(), Topic { name: name.to_string(), media });
        }
        Self { topics }
    }
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
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_gives_an_empty_store() {
        assert!(TopicStore::load("/definitely/not/here").topics.is_empty());
    }
}
