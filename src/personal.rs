//! Rows made for one profile from the signals the client sends (signals.rs): Continue Watching, My List, Because You Watched, More Like, Today's Top Picks for You, Your Next
//! Watch and Gems for You. Also the New Releases row and the collection rows (an event or a mood shown as a banner with a few lead titles). A profile without signals still gets
//! Today's Top Picks for You and Gems for You, built from the age limit, the quality of the catalog and a per session rotation.
//! Every row carries `meta` with a translation key and arguments (`title_key`, `title_args`) and, for rows about a title, a `reason`, so a client can write the title in its own language.
use crate::catalog::CatalogItem;
use crate::page::{entry_noise, PageContext, Row};
use crate::vectors::{cosine_sim, l2_normalize, DIMS};
use serde_json::json;
use std::collections::HashSet;

const NEW_RELEASE_DAYS: i64 = 45;
const GEM_MIN_RATING: f32 = 7.4;
const GEM_MAX_POPULARITY: f32 = 30.0;

/// A title is a gem when it is rated well by enough people but few have found it yet.
pub fn is_gem(item: &CatalogItem) -> bool {
    item.vote_average >= GEM_MIN_RATING && item.popularity < GEM_MAX_POPULARITY && item.vote_count >= 150
}

fn meta(key: &str, args: serde_json::Value, reason: Option<serde_json::Value>) -> Option<serde_json::Value> {
    let mut m = json!({ "title_key": key, "title_args": args });
    if let (Some(r), Some(obj)) = (reason, m.as_object_mut()) {
        obj.insert("reason".to_string(), r);
    }
    Some(m)
}

impl<'a> PageContext<'a> {
    /// A title of the profile's own lists (saved, recent): any surface rule and age rule of the page, but not the audience rule (a parent's list may hold children's titles).
    fn personal_ok(&self, item: &CatalogItem) -> bool {
        !item.adult
            && (self.surface == "both" || item.media_type == self.surface || item.media_type == "both")
            && self.genre_spec.is_none_or(|g| g.matches(item))
            && self.maturity.is_none_or(|m| self.maturity_allows(m, item))
            && (!crate::page::safety_nets() || self.allowed_by_nets(item))
    }

    fn vector_of(&self, key: u32) -> Option<Vec<f32>> {
        let ix = self.vectors.index_of(key)?;
        if self.vectors.is_zero_row(ix) {
            return None;
        }
        Some(self.vectors.get_f32(ix))
    }

    /// The mean of the vectors of some titles, each with a weight, normalised; None when none has a vector.
    fn taste_of(&self, parts: &[(u32, f32)]) -> Option<Vec<f32>> {
        let mut sum = vec![0f32; DIMS];
        let mut any = false;
        for (key, w) in parts {
            if let Some(v) = self.vector_of(*key) {
                for (a, x) in sum.iter_mut().zip(&v) {
                    *a += w * x;
                }
                any = true;
            }
        }
        if !any {
            return None;
        }
        l2_normalize(&mut sum);
        Some(sum)
    }

    async fn neighbour_keys(&self, vec: &[f32], k: usize) -> Vec<u32> {
        let guard = self.hnsw.read().await;
        self.retrieve_sync(&guard, vec, k).into_iter().map(|(id, _)| id).collect()
    }

    /// The titles from `keys` this page may show, ranked by closeness to `taste` plus a little quality, best first. `gems_only` keeps the hidden favourites.
    fn rank_by_taste(&self, keys: &[u32], taken: &HashSet<u32>, taste: &[f32], gems_only: bool) -> Vec<&CatalogItem> {
        let mut cands: Vec<(f32, &CatalogItem)> = self
            .eligible(keys, taken, false)
            .into_iter()
            .filter(|i| !gems_only || is_gem(i))
            .take(400)
            .filter_map(|i| {
                let v = self.vector_of(i.id)?;
                Some((cosine_sim(taste, &v) + 0.25 * (crate::quality::item_quality(i) / 35.0).min(1.5), i))
            })
            .collect();
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.id.cmp(&b.1.id)));
        cands.into_iter().map(|c| c.1).collect()
    }

    /// The titles from `keys` without a taste (a profile with no history): quality order shaken per session.
    fn rank_by_session(&self, keys: &[u32], taken: &HashSet<u32>, gems_only: bool, salt: usize) -> Vec<&CatalogItem> {
        let mut cands: Vec<(f32, &CatalogItem)> = self
            .eligible(keys, taken, false)
            .into_iter()
            .filter(|i| !gems_only || is_gem(i))
            // up to 150 films and 150 series (series have larger popularity numbers and would fill the first 300 alone)
            .scan((0usize, 0usize), |n, i| {
                let slot = if i.media_type == "tv" { &mut n.1 } else { &mut n.0 };
                *slot += 1;
                Some((*slot <= 150).then_some(i))
            })
            .flatten()
            .enumerate()
            .map(|(rank, i)| (40.0 * entry_noise(self.session, salt + i.id as usize) - rank as f32, i))
            .collect();
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.id.cmp(&b.1.id)));
        cands.into_iter().map(|c| c.1).collect()
    }

    fn list_row(&self, keys: &[u32], title: &str, archetype: &str, key: &str, reason: &str) -> Option<Row> {
        let items: Vec<&CatalogItem> = keys.iter().filter_map(|k| self.catalog.get(k)).filter(|i| self.personal_ok(i)).take(15).collect();
        if items.len() < 3 {
            return None;
        }
        let mut r = self.row_of_any(title, "personal", &items)?;
        r.archetype = archetype.to_string();
        r.meta = meta(key, json!({}), Some(json!({ "kind": reason })));
        Some(r)
    }

    /// A row about one title (its neighbours); the row title and its reason name that title.
    async fn about_title_row(&self, anchor: &CatalogItem, taken: &HashSet<u32>, key: &str, kind: &str, title: String, archetype: &str) -> Option<Row> {
        let v = self.vector_of(anchor.id)?;
        let keys = self.neighbour_keys(&v, 400).await;
        let mut taken = taken.clone();
        taken.insert(anchor.id);
        let ranked = self.rank_by_taste(&keys, &taken, &v, false);
        let mut r = self.row_of(&title, "personal", &ranked)?;
        r.archetype = archetype.to_string();
        r.meta = meta(key, json!({ "title": anchor.title }), Some(json!({ "kind": kind, "title": anchor.title, "id": anchor.tmdb_id, "media_type": anchor.media_type })));
        Some(r)
    }

    /// All personal rows of this page. `taken` holds the titles other rows of the page already use; discovery rows avoid them and avoid the profile's own titles.
    pub(crate) async fn build_personal_rows(&self, taken: &HashSet<u32>) -> Vec<Row> {
        let sig = self.signals;
        let own: HashSet<u32> = sig.saved.iter().chain(&sig.recent).chain(&sig.liked).copied().collect();
        let mut rows: Vec<Row> = Vec::new();
        let on_home = self.genre.is_none() && self.language.is_none();
        if on_home {
            if let Some(r) = self.list_row(&sig.recent, "Continue Watching", "continue", "row.continue_watching", "continue") { rows.push(r); }
            if let Some(r) = self.list_row(&sig.saved, "My List", "list", "row.my_list", "saved") { rows.push(r); }
        }
        let mut used: HashSet<u32> = taken.clone();
        used.extend(own.iter().copied());
        // titles of the profile in the order of use for anchors: the latest watched first, then liked, then saved
        let anchors = |keys: &[u32], skip: &HashSet<u32>| -> Vec<&CatalogItem> {
            keys.iter().filter(|k| !skip.contains(*k)).filter_map(|k| self.catalog.get(k)).filter(|i| self.personal_ok(i)).collect()
        };
        let mut anchor_used: HashSet<u32> = HashSet::new();
        // Because You Watched <title>: up to two of the latest watched titles
        for a in anchors(&sig.recent, &anchor_used).into_iter().take(2) {
            anchor_used.insert(a.id);
            if let Some(r) = self.about_title_row(a, &used, "row.because_you_watched", "because_you_watched", format!("Because You Watched {}", a.title), "byw").await {
                r.items.iter().for_each(|i| { used.insert(i.id); });
                rows.push(r);
            }
        }
        // More Like <title>: a liked or saved title
        let liked_first: Vec<u32> = sig.liked.iter().chain(&sig.saved).copied().collect();
        if let Some(a) = anchors(&liked_first, &anchor_used).into_iter().next() {
            anchor_used.insert(a.id);
            if let Some(r) = self.about_title_row(a, &used, "row.more_like", "more_like", format!("More Like {}", a.title), "more_like").await {
                r.items.iter().for_each(|i| { used.insert(i.id); });
                rows.push(r);
            }
        }
        // the profile's taste: recent titles weigh most, then liked, then saved; none of it for a new profile
        let mut parts: Vec<(u32, f32)> = Vec::new();
        for (n, k) in sig.recent.iter().take(8).enumerate() { parts.push((*k, 1.0 * 0.85f32.powi(n as i32))); }
        for k in sig.liked.iter().take(10) { parts.push((*k, 1.2)); }
        for k in sig.saved.iter().take(10) { parts.push((*k, 0.8)); }
        let taste = self.taste_of(&parts);
        // Today's Top Picks for You
        let top = match &taste {
            Some(t) => {
                let keys = self.neighbour_keys(t, 800).await;
                self.rank_by_taste(&keys, &used, t, false)
            }
            None => blend_types(self.rank_by_session(&self.catalog.chart_keys[..self.catalog.chart_keys.len().min(1500)], &used, false, 100_003)),
        };
        if let Some(mut r) = self.row_of("Today's Top Picks for You", "personal", &top) {
            r.archetype = "top_picks".to_string();
            r.meta = meta("row.todays_top_picks", json!({}), Some(json!({ "kind": if taste.is_some() { "taste" } else { "popular" } })));
            r.items.iter().for_each(|i| { used.insert(i.id); });
            rows.push(r);
        }
        // Your Next Watch: close to the latest watched titles
        if let Some(t) = self.taste_of(&sig.recent.iter().take(3).map(|k| (*k, 1.0)).collect::<Vec<_>>()) {
            let keys = self.neighbour_keys(&t, 600).await;
            let ranked = self.rank_by_taste(&keys, &used, &t, false);
            if let Some(mut r) = self.row_of("Your Next Watch", "personal", &ranked) {
                r.archetype = "next_watch".to_string();
                r.meta = meta("row.your_next_watch", json!({}), Some(json!({ "kind": "recent" })));
                r.items.iter().for_each(|i| { used.insert(i.id); });
                rows.push(r);
            }
        }
        // We Think You'll Love These: near what the profile liked or saved (not what it watched last)
        if on_home {
            let taste_lk = self.taste_of(&sig.liked.iter().take(10).map(|k| (*k, 1.0)).chain(sig.saved.iter().take(10).map(|k| (*k, 0.8))).collect::<Vec<_>>());
            if let Some(t) = taste_lk {
                let keys = self.neighbour_keys(&t, 800).await;
                let ranked = self.rank_by_taste(&keys, &used, &t, false);
                if let Some(mut r) = self.row_of("We Think You'll Love These", "personal", &ranked) {
                    r.archetype = "love_these".to_string();
                    r.meta = meta("row.we_think_youll_love", json!({}), Some(json!({ "kind": "liked" })));
                    r.items.iter().for_each(|i| { used.insert(i.id); });
                    rows.push(r);
                }
            }
            if let Some(r) = self.list_row(&sig.liked, "Familiar Favourites", "favourites", "row.familiar_favourites", "liked") { rows.push(r); }
        }
        // Gems for You: hidden favourites near the taste, or the best of them for a new profile
        let gems = match &taste {
            Some(t) => {
                let keys = self.neighbour_keys(t, 2500).await;
                self.rank_by_taste(&keys, &used, t, true)
            }
            None => blend_types(self.rank_by_session(self.catalog.gem_keys.as_slice(), &used, true, 200_017)),
        };
        if let Some(mut r) = self.row_of("Gems for You", "personal", &gems) {
            r.archetype = "gems".to_string();
            r.meta = meta("row.gems_for_you", json!({}), Some(json!({ "kind": if taste.is_some() { "taste" } else { "popular" } })));
            rows.push(r);
        }
        rows
    }

    /// Titles that came out in the last weeks (the real release dates of fresh.json), best known first; None when fewer than six.
    pub(crate) fn build_new_row(&self, taken: &HashSet<u32>) -> Option<Row> {
        let today = crate::fresh::today_days();
        let mut items: Vec<&CatalogItem> = self
            .fresh
            .items
            .iter()
            .filter(|f| crate::fresh::parse_days(&f.release_date).is_some_and(|d| (0..=NEW_RELEASE_DAYS).contains(&(today - d))))
            .filter_map(|f| self.catalog.get(&crate::catalog::item_key(&f.media_type, f.id)))
            .filter(|i| !taken.contains(&i.id) && self.matches_filters(i) && i.has_poster() && i.has_overview() && self.reliable(i))
            .collect();
        items.sort_by(|a, b| b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then(a.id.cmp(&b.id)));
        let mut r = self.row_of_any("New Releases", "new", &items)?;
        r.archetype = "new".to_string();
        r.meta = meta("row.new_releases", json!({}), None);
        Some(r)
    }

    /// The events whose calendar window is open today and that have title groups to show (Halloween in October, Pride in June).
    pub(crate) fn active_events(&self) -> Vec<&str> {
        events_in_season(self.taxonomy, self.vibe, self.today)
    }

    /// Chips for the top of a page: one for each event in season, with the page it opens (the Halloween chip of the Netflix app).
    pub fn season_chips(&self) -> Vec<serde_json::Value> {
        self.active_events().into_iter().map(|ev| json!({ "id": ev, "label_key": format!("event.{ev}"), "label": prettify(ev), "target": event_target(ev) })).collect()
    }

    /// The collection rows of the page: the themes the engine picked for this profile in this part of the day. A theme is an event in season (its sub groups of
    /// titles) or a mood ("For Every Feeling"); the pick rotates every six hours and differs between profiles, events in season are favoured, and a page shows one
    /// or two of them. Each row carries what a banner needs: label, description, accent hue, target page and the lead titles' pictures.
    pub(crate) fn build_collection_rows(&self, taken: &HashSet<u32>, want: usize) -> Vec<Row> {
        let slot = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600)) as usize;
        let seed = format!("{}|{}", self.session, slot);
        let count = (1 + (entry_noise(&seed, 99_991) > 0.45) as usize).min(want);
        enum Theme<'a> { Event(&'a str), Mood(&'a crate::vibe::Mood) }
        let mut themes: Vec<(f32, Theme)> = Vec::new();
        for ev in self.active_events() { themes.push((0.35 + entry_noise(&seed, ev.len() * 31 + ev.bytes().map(|b| b as usize).sum::<usize>()), Theme::Event(ev))); }
        for m in &self.vibe.moods { themes.push((entry_noise(&seed, 7_000 + m.id.len() * 13 + m.id.bytes().map(|b| b as usize).sum::<usize>()) + if self.late_night && m.late { 0.4 } else { 0.0 }, Theme::Mood(m))); }
        themes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new();
        let mut used = taken.clone();
        for (_, theme) in themes {
            if rows.len() >= count { break; }
            match theme {
                Theme::Event(ev) => {
                    let mut lead: Vec<&CatalogItem> = Vec::new();
                    let mut names: Vec<&str> = Vec::new();
                    for sub in &self.vibe.seasons[ev] {
                        if sub.kw_share < 0.4 || names.len() >= 4 { continue; }
                        let got = self.season_fit(ev, self.eligible(&sub.items, &used, true));
                        if got.len() < 8 { continue; }
                        names.push(sub.name.as_str());
                        lead.extend(got.into_iter().take(5));
                    }
                    let label = prettify(ev);
                    if lead.len() < 8 { continue; }
                    if let Some(mut r) = self.row_of_any(&format!("{label} Collection"), "collection", &lead) {
                        r.archetype = "collection".to_string();
                        r.meta = Some(collection_meta(ev, &format!("event.{ev}"), &label, &names.join(", "), event_target(ev), &r));
                        r.items.iter().for_each(|i| { used.insert(i.id); });
                        rows.push(r);
                    }
                }
                Theme::Mood(m) => {
                    if let Some(mut r) = self.mood_row(m, &used) {
                        let names: Vec<&str> = m.groups.iter().take(3).filter_map(|(g, _)| self.group_names.get(&(5, *g)).map(|s| s.as_str())).collect();
                        r.archetype = "collection".to_string();
                        r.kind = "collection".to_string();
                        r.meta = Some(collection_meta(&m.id, "collection.for_every_feeling", "For Every Feeling", &names.join(", "), json!({ "genre": "Moods" }), &r));
                        r.items.iter().for_each(|i| { used.insert(i.id); });
                        rows.push(r);
                    }
                }
            }
        }
        rows
    }
}

/// Films and series alternate in a list that comes from popularity (series have larger popularity numbers and would fill a row), each kept in its own order.
fn blend_types(items: Vec<&CatalogItem>) -> Vec<&CatalogItem> {
    let (movies, series): (Vec<&CatalogItem>, Vec<&CatalogItem>) = items.into_iter().partition(|i| i.media_type == "movie");
    let (mut m, mut t) = (movies.into_iter(), series.into_iter());
    let mut out = Vec::new();
    loop {
        let (a, b) = (m.next(), t.next());
        if a.is_none() && b.is_none() { break; }
        out.extend(a);
        out.extend(b);
    }
    out
}

pub fn events_in_season<'a>(taxonomy: &'a crate::taxonomy::TaxonomyStore, vibe: &crate::vibe::VibeIndex, today: (u8, u8)) -> Vec<&'a str> {
    let mut events: Vec<&str> = Vec::new();
    for e in &taxonomy.entries {
        let all_year = e.season.is_none_or(|s| s.start == (1, 1) && s.end == (12, 31));
        if e.kind == "calendar" && !all_year && e.season.is_some_and(|s| s.active_on(today)) && vibe.seasons.contains_key(&e.event) && !events.contains(&e.event.as_str()) {
            events.push(e.event.as_str());
        }
    }
    events
}

pub fn prettify(id: &str) -> String {
    id.split('_').map(|w| { let mut c = w.chars(); c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default() }).collect::<Vec<_>>().join(" ")
}

/// The page that shows the whole collection of an event: the picker pages for the events that have one, the event id otherwise.
fn event_target(ev: &str) -> serde_json::Value {
    match ev {
        "halloween" => json!({ "genre": "Halloween" }),
        "pride" => json!({ "genre": "Pride" }),
        _ => json!({ "genre": ev }),
    }
}

fn collection_meta(id: &str, label_key: &str, label: &str, description: &str, target: serde_json::Value, row: &Row) -> serde_json::Value {
    let images: Vec<&str> = row.items.iter().filter_map(|i| i.backdrop_path.as_deref().or(i.poster_path.as_deref())).take(8).collect();
    // a colour of its own for each collection, the same on every visit
    let hue = id.bytes().fold(17u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) % 360;
    json!({
        "title_key": "collection.title",
        "title_args": { "label": label },
        "collection": { "id": id, "label_key": label_key, "label": label, "description": description, "target": target, "images": images, "hue": hue },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(rating: f32, popularity: f32, votes: u32) -> CatalogItem {
        CatalogItem {
            id: 1, tmdb_id: 1, media_type: "movie".into(), title: "t".into(), release_year: Some(2000), genres: vec![],
            overview: String::new(), tagline: None, director: None, poster_path: Some("/p.jpg".into()), backdrop_path: Some("/b.jpg".into()),
            vote_average: rating, popularity, is_anime: false, vote_count: votes, adult: false,
            original_language: "en".into(), tv_type: None, countries: vec![], keywords: vec![], runtime: 0, flags: 0, cert: 0, cert_gb: 0, cert_us: 0,
        }
    }

    #[test]
    fn blend_alternates_films_and_series() {
        let (mut a, mut b, mut c) = (item(8.0, 5.0, 400), item(8.0, 5.0, 400), item(8.0, 5.0, 400));
        a.id = 1; b.id = 2; c.id = 3; b.media_type = "tv".into(); c.media_type = "tv".into();
        let out = blend_types(vec![&b, &c, &a]);
        assert_eq!(out.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn a_gem_is_well_rated_little_known_and_voted_on() {
        assert!(is_gem(&item(8.0, 5.0, 400)));
        assert!(!is_gem(&item(6.5, 5.0, 400)), "rating too low");
        assert!(!is_gem(&item(8.0, 90.0, 400)), "too popular");
        assert!(!is_gem(&item(8.0, 5.0, 20)), "too few votes");
    }

    #[test]
    fn event_ids_become_labels_and_targets() {
        assert_eq!(prettify("halloween"), "Halloween");
        assert_eq!(prettify("black_history_month"), "Black History Month");
        assert_eq!(event_target("pride"), json!({ "genre": "Pride" }));
        assert_eq!(event_target("diwali"), json!({ "genre": "diwali" }));
    }

    #[test]
    fn collection_meta_carries_label_target_and_art() {
        let row = Row { title: "x".into(), archetype: "collection".into(), kind: "collection".into(), items: vec![crate::page::RowItem::from_catalog(&item(8.0, 5.0, 400))], meta: None };
        let m = collection_meta("halloween", "event.halloween", "Halloween", "Spooky, Gothic", event_target("halloween"), &row);
        assert_eq!(m["title_key"], "collection.title");
        assert_eq!(m["collection"]["target"]["genre"], "Halloween");
        assert_eq!(m["collection"]["images"], json!(["/b.jpg"]));
        assert_eq!(m["collection"]["description"], "Spooky, Gothic");
    }

    #[test]
    fn meta_adds_the_reason_only_when_given() {
        assert!(meta("row.gems_for_you", json!({}), None).unwrap().get("reason").is_none());
        let m = meta("row.because_you_watched", json!({ "title": "A" }), Some(json!({ "kind": "recent" }))).unwrap();
        assert_eq!(m["reason"]["kind"], "recent");
    }
}
