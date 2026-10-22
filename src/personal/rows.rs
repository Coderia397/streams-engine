//! Rows made for one profile: Today's Top Picks, Gems, Because You Watched, More Like, Your Next Watch, My List, Continue Watching and New Releases.
use super::*;

const NEW_RELEASE_DAYS: i64 = 45;
const GEM_MIN_RATING: f32 = 7.4;
const GEM_MAX_POPULARITY: f32 = 30.0;

/// A title is a gem when it is rated well by enough people but few have found it yet.
/// Titles in the Everyone's Watching row.
const POPULAR_ROW: usize = 60;
/// The scope's own titles a thin page draws its session rows from (best quality first)
const SCOPE_POOL: usize = 3000;

pub fn is_gem(item: &CatalogItem) -> bool {
    item.vote_average >= GEM_MIN_RATING && item.popularity < GEM_MAX_POPULARITY && item.vote_count >= 150
}

pub(crate) fn meta(key: &str, args: serde_json::Value, reason: Option<serde_json::Value>) -> Option<serde_json::Value> {
    let mut m = json!({ "title_key": key, "title_args": args });
    if let (Some(r), Some(obj)) = (reason, m.as_object_mut()) {
        obj.insert("reason".to_string(), r);
    }
    Some(m)
}

impl<'a> PageContext<'a> {
    /// A title of the profile's own lists (saved, recent): any surface rule and age rule of the page, but not the audience rule (a parent's list may hold children's titles).
    pub(crate) fn personal_ok(&self, item: &CatalogItem) -> bool {
        !item.adult
            && (self.is_kids() || !self.childrens_title(item))
            && (self.surface == "both" || item.media_type == self.surface || item.media_type == "both")
            && self.genre_spec.is_none_or(|g| g.matches(item))
            && self.maturity.is_none_or(|m| self.maturity_allows(m, item))
            && (!crate::page::safety_nets() || self.allowed_by_nets(item))
    }

    pub(crate) fn vector_of(&self, key: u32) -> Option<Vec<f32>> {
        let ix = self.vectors.index_of(key)?;
        if self.vectors.is_zero_row(ix) {
            return None;
        }
        Some(self.vectors.get_f32(ix))
    }

    /// The mean of the vectors of some titles, each with a weight, normalised; None when none has a vector.
    pub(crate) fn taste_of(&self, parts: &[(u32, f32)]) -> Option<Vec<f32>> {
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

    pub(crate) async fn neighbour_keys(&self, vec: &[f32], k: usize) -> Vec<u32> {
        let guard = self.hnsw.read().await;
        self.retrieve_sync(&guard, vec, k).into_iter().map(|(id, _)| id).collect()
    }

    /// The titles from `keys` this page may show, ranked by closeness to `taste` plus a little quality, best first. `gems_only` keeps the hidden favourites.
    pub(crate) fn rank_by_taste(&self, keys: &[u32], taken: &HashSet<u32>, taste: &[f32], gems_only: bool) -> Vec<&CatalogItem> {
        let mut cands: Vec<(f32, &CatalogItem)> = self
            .eligible(keys, taken, false)
            .into_iter()
            .filter(|i| (self.is_kids() || !self.childrens_title(i)) && (!gems_only || is_gem(i)))
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
    pub(crate) fn rank_by_session(&self, keys: &[u32], taken: &HashSet<u32>, gems_only: bool, salt: usize) -> Vec<&CatalogItem> {
        let mut cands: Vec<(f32, &CatalogItem)> = self
            .eligible(keys, taken, false)
            .into_iter()
            .filter(|i| (self.is_kids() || !self.childrens_title(i)) && (!gems_only || is_gem(i)))
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

    pub(crate) fn list_row(&self, keys: &[u32], title: &str, archetype: &str, key: &str, reason: &str) -> Option<Row> {
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
    pub(crate) async fn about_title_row(&self, anchor: &CatalogItem, taken: &HashSet<u32>, key: &str, kind: &str, title: String, archetype: &str) -> Option<Row> {
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

    /// The titles the page may show, best quality first, for a profile without history on a page whose scope the popularity chart leaves nearly empty (Stand-up,
    /// Black Stories, Documentaries): the scope's own titles, up to SCOPE_POOL.
    fn scope_keys(&self) -> Vec<u32> {
        let mut v: Vec<&CatalogItem> = self.catalog.values().filter(|i| self.matches_filters(i) && self.member_ok(i) && self.reliable(i)).collect();
        crate::quality::sort_by_quality(&mut v);
        v.into_iter().take(SCOPE_POOL).map(|i| i.id).collect()
    }

    /// The titles the page's Top 10 rows will hold (the most popular of each type the page may show): a profile without history must not meet them again in its
    /// Top Picks (place_charts strips a Top 10 title out of every other row, which left the Home page's Top Picks under six titles and dropped it).
    fn chart_ids(&self) -> HashSet<u32> {
        let mut out: HashSet<u32> = HashSet::default();
        if self.genre.is_some() {
            return out;
        }
        for t in ["movie", "tv"] {
            out.extend(
                self.catalog
                    .chart_keys
                    .iter()
                    .filter_map(|k| self.catalog.get(k))
                    .filter(|i| i.media_type == t && self.matches_filters(i) && !self.fresh.is_upcoming(&i.media_type, i.tmdb_id, &self.today_ymd))
                    .take(crate::page::TOP_N)
                    .map(|i| i.id),
            );
        }
        out
    }

    /// A row's candidates for a profile without history: the popularity chart's first `chart` titles the page may show (the gems list for `gems_only`), shaken per session;
    /// when the page's scope leaves fewer than a row of them, the scope's own titles instead (its gems: rated 7 or more and little watched).
    fn session_pool(&self, chart: usize, used: &HashSet<u32>, gems_only: bool, salt: usize) -> Vec<&CatalogItem> {
        let from_chart = if gems_only {
            self.rank_by_session(self.catalog.gem_keys.as_slice(), used, true, salt)
        } else {
            self.rank_by_session(&self.catalog.chart_keys[..self.catalog.chart_keys.len().min(chart)], used, false, salt)
        };
        if from_chart.len() >= crate::page::COHERENCE_MIN_ITEMS {
            return blend_types(from_chart);
        }
        let scope = self.scope_keys();
        let ranked = self.rank_by_session(&scope, used, false, salt);
        if gems_only {
            ranked.into_iter().filter(|i| i.vote_average >= 7.0 && i.popularity < GEM_MAX_POPULARITY).collect()
        } else {
            blend_types(ranked)
        }
    }

    /// All personal rows of this page. `taken` holds the titles other rows of the page already use; discovery rows avoid them and avoid the profile's own titles.
    /// A children's genre page carries no Top Picks and no Next Watch (Netflix: none of its 36 children's genre pages, note 97).
    pub(crate) async fn build_personal_rows(&self, taken: &HashSet<u32>) -> Vec<Row> {
        let sig = self.signals;
        let kids_genre_page = self.is_kids() && self.genre.is_some();
        let own: HashSet<u32> = sig.saved.iter().chain(&sig.recent).chain(&sig.liked).copied().collect();
        let mut rows: Vec<Row> = Vec::new();
        let on_home = self.genre.is_none() && self.language.is_none();
        if on_home {
            if let Some(r) = self.list_row(&sig.recent, "Continue Watching", "continue", "row.continue_watching", "continue") { rows.push(r); }
            if let Some(r) = self.list_row(&sig.saved, "My List", "list", "row.my_list", "saved") { rows.push(r); }
        }
        let mut used: HashSet<u32> = taken.clone();
        used.extend(own.iter().copied());
        used.extend(self.chart_ids());
        // titles of the profile in the order of use for anchors: the latest watched first, then liked, then saved
        let anchors = |keys: &[u32], skip: &HashSet<u32>| -> Vec<&CatalogItem> {
            keys.iter().filter(|k| !skip.contains(*k)).filter_map(|k| self.catalog.get(k)).filter(|i| self.personal_ok(i)).collect()
        };
        let mut anchor_used: HashSet<u32> = HashSet::default();
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
            None if kids_genre_page => Vec::new(),
            None => self.session_pool(1500, &used, false, 100_003),
        };
        if let Some(mut r) = self.row_of("Today's Top Picks for You", "personal", &top) {
            r.archetype = "top_picks".to_string();
            r.meta = meta("row.todays_top_picks", json!({}), Some(json!({ "kind": if taste.is_some() { "taste" } else { "popular" } })));
            r.items.iter().for_each(|i| { used.insert(i.id); });
            rows.push(r);
        }
        let composed = self.composed_rows();
        // Your Next Watch: close to the latest watched titles; a profile without history gets well liked titles chosen by its session
        let next = match self.taste_of(&sig.recent.iter().take(3).map(|k| (*k, 1.0)).collect::<Vec<_>>()) {
            Some(t) => {
                let keys = self.neighbour_keys(&t, 600).await;
                self.rank_by_taste(&keys, &used, &t, false)
            }
            None if composed && !kids_genre_page => self.session_pool(3000, &used, false, 300_007),
            None => Vec::new(),
        };
        if let Some(mut r) = self.row_of("Your Next Watch", "personal", &next) {
            r.archetype = "next_watch".to_string();
            r.meta = meta("row.your_next_watch", json!({}), Some(json!({ "kind": if sig.recent.is_empty() { "popular" } else { "recent" } })));
            r.items.iter().for_each(|i| { used.insert(i.id); });
            rows.push(r);
        }
        // We Think You'll Love These: near what the profile liked or saved (not what it watched last); on every composed page, and for a new profile by session
        if on_home || composed {
            let taste_lk = self.taste_of(&sig.liked.iter().take(10).map(|k| (*k, 1.0)).chain(sig.saved.iter().take(10).map(|k| (*k, 0.8))).collect::<Vec<_>>());
            let love = match taste_lk {
                Some(t) => {
                    let keys = self.neighbour_keys(&t, 800).await;
                    self.rank_by_taste(&keys, &used, &t, false)
                }
                None if composed => self.session_pool(4000, &used, false, 400_009),
                None => Vec::new(),
            };
            if let Some(mut r) = self.row_of("We Think You'll Love These", "personal", &love) {
                r.archetype = "love_these".to_string();
                r.meta = meta("row.we_think_youll_love", json!({}), Some(json!({ "kind": if sig.liked.is_empty() && sig.saved.is_empty() { "popular" } else { "liked" } })));
                r.items.iter().for_each(|i| { used.insert(i.id); });
                rows.push(r);
            }
            if let Some(r) = self.list_row(&sig.liked, "Familiar Favourites", "favourites", "row.familiar_favourites", "liked") { rows.push(r); }
        }
        // Gems for You: hidden favourites near the taste, or the best of them for a new profile
        let gems = match &taste {
            Some(t) => {
                let keys = self.neighbour_keys(t, 2500).await;
                self.rank_by_taste(&keys, &used, t, true)
            }
            None => self.session_pool(0, &used, true, 200_017),
        };
        if let Some(mut r) = self.row_of("Gems for You", "personal", &gems) {
            r.archetype = "gems".to_string();
            r.meta = meta("row.gems_for_you", json!({}), Some(json!({ "kind": if taste.is_some() { "taste" } else { "popular" } })));
            rows.push(r);
        }
        rows
    }

    /// Everyone's Watching: the most popular titles this page may show after its Top 10, up to POPULAR_ROW titles (Netflix's row holds about 80).
    pub(crate) fn build_popular_row(&self, taken: &HashSet<u32>) -> Option<Row> {
        let items: Vec<&CatalogItem> = self
            .catalog
            .chart_keys
            .iter()
            .filter_map(|k| self.catalog.get(k))
            .filter(|i| !taken.contains(&i.id) && self.matches_filters(i) && i.has_poster() && i.has_overview() && !self.fresh.is_upcoming(&i.media_type, i.tmdb_id, &self.today_ymd))
            .filter(|i| self.is_kids() || !self.childrens_title(i))
            .skip(crate::page::TOP_N)
            .take(POPULAR_ROW)
            .collect();
        if items.len() < 10 {
            return None;
        }
        let row_items: Vec<crate::page::RowItem> = items.iter().map(|i| crate::page::RowItem::from_catalog(i)).collect();
        Some(Row { title: "Everyone's Watching".to_string(), archetype: "popular".to_string(), kind: "popular".to_string(), items: row_items, meta: meta("row.everyones_watching", json!({}), None) })
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
}
