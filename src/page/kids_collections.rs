//! Banner collections of the Kids pages, made from the Kids row vocabulary and nothing else: any row can lead a collection (its name is the collection's name, its description the sentence), and the
//! collection is that row together with the vocabulary rows nearest to it. The collections of the calendar rows (Halloween, Christmas, Diwali, Eid ...) show only inside their dates and are favoured then.
//! Which collections a page gets depends on the session and on the profile's taste, so nothing is a fixed list. The banner carries the name as its logo, the sentence and Explore all, which opens a
//! page with one row for each row of the collection.
use super::entry::{title_tokens, titles_alike};
use super::*;

/// Neighbouring rows that lead a collection with its own row.
const NEIGHBOURS: usize = 5;
/// Titles taken from each row of a collection for its banner.
const PER_ROW: usize = 4;
/// Two collections on a page are further apart than this (centred cosine of their lead rows).
const APART: f32 = 0.55;

/// The part of a name that identifies a collection in an address.
pub(crate) fn slug(title: &str) -> String {
    let words: Vec<String> = title.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect();
    format!("kids_{}", words.join("_"))
}

/// A sentence for the banner: the first sentence of the row's description, cut at a word when it is long.
fn sentence(description: &str) -> String {
    let first = description.split(". ").next().unwrap_or(description).trim().trim_end_matches('.');
    if first.chars().count() <= 120 {
        return first.to_string();
    }
    let cut: String = first.chars().take(120).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.clone(), |(a, _)| a.to_string());
    format!("{}…", cut.trim_end_matches(|c: char| c == ',' || c == ' '))
}

/// The vocabulary rows that may lead a collection: not the film and series twins of a theme, not the editorial rows.
fn leads_a_collection(kind: &str) -> bool {
    !matches!(kind, "variant" | "editorial" | "animation" | "style" | "")
}

impl<'a> PageContext<'a> {
    fn kids_entry_open(&self, e: &crate::taxonomy::TaxonomyEntry) -> bool {
        !e.centered.is_empty() && e.season.is_none_or(|s| s.active_on(self.today))
    }

    /// The vocabulary rows nearest to row `idx` (centred cosine), without rows that carry the same name, out of season rows or rows that need a title kind this page does not have.
    fn kids_neighbours(&self, idx: usize) -> Vec<usize> {
        let me = &self.taxonomy.entries[idx];
        let my_tokens = title_tokens(&me.title);
        let mut scored: Vec<(f32, usize)> = self
            .taxonomy
            .entries
            .iter()
            .enumerate()
            .filter(|(i, e)| *i != idx && self.kids_entry_open(e) && !titles_alike(&my_tokens, &title_tokens(&e.title)))
            .map(|(i, e)| (cosine_sim(&me.centered, &e.centered), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out: Vec<usize> = Vec::new();
        let mut tokens: Vec<HashSet<String>> = vec![my_tokens];
        for (_, i) in scored {
            if out.len() >= NEIGHBOURS { break; }
            let t = title_tokens(&self.taxonomy.entries[i].title);
            if tokens.iter().any(|x| titles_alike(x, &t)) { continue; }
            tokens.push(t);
            out.push(i);
        }
        out
    }

    /// Titles that fill a vocabulary row for this profile, nearest first: the entry's neighbours kept by the page's filters.
    fn kids_lead_titles(&self, guard: &Option<crate::hnsw::HnswIndex>, idx: usize, taken: &HashSet<u32>) -> Vec<&CatalogItem> {
        let e = &self.taxonomy.entries[idx];
        self.retrieve_sync(guard, &e.embedding, ANN_ENTRY_K)
            .into_iter()
            .filter_map(|(id, _)| self.catalog.get(&id))
            .filter(|i| !taken.contains(&i.id) && self.matches_filters(i) && self.member_ok(i) && self.reliable(i))
            .collect()
    }

    /// The banners of a Kids page: two, a third now and then. A row is chosen to lead by its taste fit, a share of chance that changes every six hours, and a boost when it is a calendar row in season.
    pub(crate) async fn build_kids_collection_rows(&self, taken: &HashSet<u32>, want: usize) -> Vec<Row> {
        let slot = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600)) as usize;
        let seed = format!("{}|{}", self.session, slot);
        let count = (2 + (entry_noise(&seed, 99_991) > 0.6) as usize).min(want);
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let taste_c: Vec<f32> = if has_taste {
            let mut c: Vec<f32> = self.taste.iter().zip(self.mean).map(|(x, m)| x - m).collect();
            l2_normalize(&mut c);
            c
        } else {
            Vec::new()
        };
        let mut themes: Vec<(f32, usize)> = self
            .taxonomy
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| self.kids_entry_open(e) && (e.kind == "calendar" || leads_a_collection(&e.kind)))
            .map(|(i, e)| {
                let season = if e.kind == "calendar" && e.season.is_some() { 0.9 } else { 0.0 };
                let taste = if has_taste { 1.0 * cosine_sim(&taste_c, &e.centered) } else { 0.0 };
                (entry_noise(&seed, 31_000 + i) + season + taste, i)
            })
            .collect();
        themes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let guard = self.hnsw.read().await;
        let mut rows: Vec<Row> = Vec::new();
        let mut used = taken.clone();
        let mut chosen: Vec<usize> = Vec::new();
        for (_, idx) in themes.into_iter().take(80) {
            if rows.len() >= count { break; }
            let e = &self.taxonomy.entries[idx];
            // a collection that sits next to one already on the page waits
            if chosen.iter().any(|&c| cosine_sim(&self.taxonomy.entries[c].centered, &e.centered) > APART) { continue; }
            let mut lead: Vec<&CatalogItem> = Vec::new();
            let mut seen: HashSet<u32> = HashSet::default();
            for i in std::iter::once(idx).chain(self.kids_neighbours(idx)) {
                for t in self.kids_lead_titles(&guard, i, &used).into_iter().filter(|t| seen.insert(t.id)).take(PER_ROW) { lead.push(t); }
            }
            if lead.len() < 8 { continue; }
            if let Some(mut r) = self.row_of_any(&e.title, "collection", &lead) {
                let id = slug(&e.title);
                r.archetype = "collection".to_string();
                r.kind = "collection".to_string();
                r.meta = Some(crate::personal::collection_meta(&id, "collection.kids", "Streams Kids", &sentence(&e.vibe_query), serde_json::json!({ "collection": id }), &r));
                r.items.iter().for_each(|i| { used.insert(i.id); });
                chosen.push(idx);
                rows.push(r);
            }
        }
        rows
    }

    /// The page Explore all opens for a Kids collection: one row for the leading row and each of its neighbours, and the header (label, name, sentence, pictures, colour).
    pub(crate) async fn build_kids_collection(&self, id: &str) -> Option<(Vec<Row>, serde_json::Value)> {
        let idx = self.taxonomy.entries.iter().position(|e| slug(&e.title) == id && self.kids_entry_open(e))?;
        let e = &self.taxonomy.entries[idx];
        let guard = self.hnsw.read().await;
        let mut rows: Vec<Row> = Vec::new();
        let mut used: HashSet<u32> = HashSet::default();
        for i in std::iter::once(idx).chain(self.kids_neighbours(idx)) {
            let items = self.kids_lead_titles(&guard, i, &used);
            if items.len() < 8 { continue; }
            if let Some(r) = self.row_of(&self.taxonomy.entries[i].title, "vibe", &items) {
                r.items.iter().for_each(|t| { used.insert(t.id); });
                rows.push(r);
            }
        }
        let images: Vec<String> = rows.iter().flat_map(|r| r.items.iter()).filter_map(|i| i.backdrop_path.clone()).filter(|p| !p.is_empty()).take(8).collect();
        if rows.is_empty() || images.is_empty() { return None; }
        let hue = id.bytes().fold(17u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) % 360;
        Some((rows, serde_json::json!({ "id": id, "label_key": "collection.kids", "label": "Streams Kids", "title": e.title, "description": sentence(&e.vibe_query), "images": images, "hue": hue })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_are_addresses() {
        assert_eq!(slug("Shh, Secrets and Mysteries"), "kids_shh_secrets_and_mysteries");
        assert_eq!(slug("Ages 2 to 4: Play and Learn"), "kids_ages_2_to_4_play_and_learn");
    }

    #[test]
    fn a_long_description_is_cut_at_a_word() {
        let s = sentence("Gentle, colourful shows for toddlers and preschoolers that teach colours, shapes, numbers, letters and kindness through songs and play. Second sentence.");
        assert!(s.ends_with('…') && s.chars().count() <= 122, "{s}");
        assert_eq!(sentence("Short one. Another."), "Short one");
    }

    #[test]
    fn variants_and_editorial_rows_do_not_lead() {
        assert!(!leads_a_collection("variant") && !leads_a_collection("editorial") && leads_a_collection("theme") && leads_a_collection("animal"));
    }
}
