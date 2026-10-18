#![allow(dead_code)] // Public API surface — functions used at runtime via dynamic dispatch

mod charts;
mod entry;
mod events;
mod genre;
mod kids_collections;

use crate::catalog::{CatalogItem, CatalogMap};
use crate::centroid::CentroidVault;
use crate::cluster::ClusterSeeds;
use crate::hnsw::{HnswIndex, ENGINE_READY};
use crate::mmr::{filter_outliers, greedy_mmr, mean_pairwise_cosine};
use crate::quality::rerank_score;
use crate::taxonomy::TaxonomyStore;
use crate::titles::{byw_title, clamp_words, derive_title};
use crate::vectors::{cosine_sim, l2_normalize, sphere_mean, VectorStore, DIMS};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Rules that keep defects out of default pages: tile art required, no news, talk or soap in default rows,
/// no Christmas or Halloween titles outside their window. `ENGINE_SAFETY_NETS=off` disables them for baselines.
pub static SAFETY_NETS: AtomicBool = AtomicBool::new(true);
pub fn safety_nets() -> bool {
    SAFETY_NETS.load(Ordering::Relaxed)
}

/// Titles in the Top 10 row
const TOP_N: usize = 10;

/// At most this many titles in a row may share a significant title word (a franchise, a character name), so a row is
/// not 13 Doctor Who entries. And at most this many documentary or reality rows on a default page.
const FRANCHISE_CAP: usize = 2;
const NONSCRIPTED_ROW_CAP: u32 = 2;

/// Session seeded exploration: how far a session may reorder seeds, and the share of members a session leaves out of
/// the MMR pool, so two sessions see different rows and titles while one session stays stable.
const SEED_NOISE: f32 = 0.30;
const MEMBER_DROP: f32 = 0.22;

/// Coherence gate thresholds. Set by percentile matching (ai-engine/scripts/calibrate.py): each gate keeps the strictness it had
/// in the first Qwen3-Embedding-4B space, translated to the current space (title free text with tagline, people and cast,
/// seeds trained on the quality pool). Rerun the script after any embedding change; thresholds from another space do not transfer.
/// Measured 2026-10-04: old to new COHERENCE_OK 0.55 to 0.60 (item to medoid cosine), COHERENCE_WEAK 0.50 to 0.56 (mean pairwise
/// cosine of a row), MMD_MAX_INTER_SIM 0.76 to 0.74 and BYW_MAX_CENTROID_SIM 0.68 to 0.63 (centroid to centroid).
const COHERENCE_STRONG: f32 = 0.74;
const COHERENCE_OK:     f32 = 0.60;
const COHERENCE_WEAK:   f32 = 0.56;
const COHERENCE_MIN_ITEMS: usize = 6;

/// Max cosine similarity between any two row anchors (MMD gate)
const MMD_MAX_INTER_SIM: f32 = 0.74;

/// MMR λ: 0 = pure diversity, 1 = pure relevance. 0.72 = production default.
const MMR_LAMBDA: f32 = 0.72;

/// Genre bitmask window: no genre repeated in the last N rows
const GENRE_WINDOW: usize = 4;

/// Entry driven pages need a real vocabulary; below this the cluster path is used.
const MIN_TAXONOMY_FOR_ENTRY_ROWS: usize = 200;

/// Best members (by quality) kept per seed before MMR
const MEMBER_POOL: usize = 120;

/// Items per row (max)
const ROW_SIZE: usize = 15;
/// A genre page shows at least this many rows: when its own pool gives fewer, rows of neighbouring titles fill it up (`fill_thin_genre_page`)
pub const THIN_PAGE_FLOOR: usize = 15;
/// Fixed positions (zero based) of the catalog and personal rows on a page
const SLOT_CONTINUE: usize = 0;
const SLOT_TOP_PICKS: usize = 1;
const SLOT_CHART: [usize; 2] = [2, 7];
const SLOT_LIST: usize = 3;
const SLOT_COLLECTION: [usize; 4] = [4, 11, 19, 28];
const SLOT_NEW: usize = 5;
const SLOT_ABOUT: [usize; 3] = [6, 12, 14];
const SLOT_NEXT: usize = 9;
const SLOT_GEMS: usize = 10;
const SLOT_LOVE: usize = 8;
const SLOT_FAVOURITES: usize = 13;
/// A genre page built from groups: the (level, id) of each cluster and the item indices kept
type GroupMode = (Vec<(u8, i32)>, Vec<usize>);
/// The titles of one group, by group id
type GroupItems = (i32, Vec<usize>);
/// An item index with its score
type ScoredIdx = (usize, f32);
/// A seasonal sub group is shown only when at least this share of its titles carry the event keywords (the rest joined the pool by similarity and drift)
const SEASON_MIN_KW_SHARE: f32 = 0.4;
/// Mean centred cosine of a seasonal sub group's best 15 titles to the event vector below which the sub group is not shown
const SEASON_MIN_FIT: f32 = 0.18;
/// Events whose pool is defined by precise catalog keywords (identity, heritage and family days): their sub groups are ranked by fit but not dropped for a low fit
const KEYWORD_EVENTS: &[&str] = &["pride", "black_history", "womens_history", "awards", "mothers_day", "fathers_day", "cinco_de_mayo", "st_patricks", "july4", "ramadan_eid", "independence_india", "hanukkah", "diwali", "easter", "lunar_new_year"];

/// ANN top-k for HNSW retrieval
const ANN_K: usize = 200;

/// Deeper retrieval for entry rows: only about 5% of the catalog passes the quality gate
const ANN_ENTRY_K: usize = 500;

/// Taxonomy cosine threshold to accept Hungarian assignment (else fallback title)
const TAXONOMY_THRESHOLD: f32 = 0.55;

/// BYW anchor: cosine to any existing row centroid must be below this
const BYW_MAX_CENTROID_SIM: f32 = 0.63;

// ── PageState ─────────────────────────────────────────────────────────────────

pub struct PageState {
    pub used_titles:   HashSet<String>,
    pub recent_genres: VecDeque<String>,
    /// Centroids of placed thematic rows, in placement order
    pub placed_centroids: Vec<Vec<f32>>,
    /// Ids already placed in a row of this page (mixing never reuses a title)
    pub used_ids: HashSet<u32>,
    /// Documentary and reality rows placed so far
    pub nonscripted_rows: u32,
}

impl Default for PageState {
    fn default() -> Self {
        Self::new()
    }
}

impl PageState {
    pub fn new() -> Self {
        Self {
            used_titles:      HashSet::new(),
            recent_genres:    VecDeque::new(),
            placed_centroids: Vec::new(),
            used_ids:         HashSet::new(),
            nonscripted_rows: 0,
        }
    }

    pub fn can_place_genre(&self, genre: &str) -> bool {
        !self.recent_genres.iter().any(|g| g == genre)
    }

    pub fn record_thematic(&mut self, genre: String, centroid: Vec<f32>) {
        self.recent_genres.push_back(genre);
        if self.recent_genres.len() > GENRE_WINDOW {
            self.recent_genres.pop_front();
        }
        self.placed_centroids.push(centroid);
    }

    /// Claim the first unused title from candidates. Final fallback always wins.
    /// Titles are compared in the form clients see (at most 4 words), so two long candidates that shorten to the
    /// same text cannot both be used on one page.
    pub fn claim_title(&mut self, candidates: &[String]) -> String {
        // "Horror & Mystery" and "Mystery & Horror" are the same title
        let canon = |t: &str| -> String {
            match t.split_once(" & ") {
                Some((a, b)) if !b.contains(" & ") => {
                    let (a, b) = (a.trim().to_lowercase(), b.trim().to_lowercase());
                    if a <= b { format!("{a} & {b}") } else { format!("{b} & {a}") }
                }
                _ => t.to_lowercase(),
            }
        };
        let clamped: Vec<String> = candidates.iter().map(|t| clamp_words(t, 6)).collect();
        let chosen = clamped
            .iter()
            .find(|t| !t.is_empty() && !self.used_titles.contains(&canon(t)))
            .cloned()
            .unwrap_or_else(|| {
                let base = clamped.last().cloned().unwrap_or_else(|| "More Picks".to_string());
                format!("{base} {}", self.used_titles.len() + 1)
            });
        self.used_titles.insert(canon(&chosen));
        chosen
    }
}

// ── Row / RowItem ─────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct Row {
    pub title:     String,
    pub archetype: String, // "trending" | "top_picks" | "byw" | "thematic"
    /// Taxonomy row type (vibe, theme, country, calendar, ...) for entry driven rows
    #[serde(skip_serializing_if = "String::is_empty")]
    pub kind:      String,
    pub items:     Vec<RowItem>,
    /// Extra facts of the row for the client (a collection's text and target, the title a personal row refers to); absent for ordinary rows
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta:      Option<serde_json::Value>,
}

#[derive(Serialize, Clone)]
pub struct RowItem {
    /// Internal key (`catalog::item_key`), used for dedup and session state
    pub id:           u32,
    /// TMDB id, what clients get as `id` (only unique together with `media_type`)
    pub tmdb_id:      u32,
    pub title:        String,
    pub media_type:   String,
    pub overview:     String,
    pub poster_path:  Option<String>,
    pub backdrop_path: Option<String>,
    pub vote_average: f32,
    pub popularity:   f32,
    pub release_year: Option<i32>,
    pub genres:       Vec<String>,
    /// TMDB tv type (Scripted, Reality, Talk Show, ...) and original language, for clients and checks
    pub tv_type:      Option<String>,
    pub language:     String,
    /// British age rank of the title (0 unknown), see `catalog::cert_rank`
    pub cert:         u8,
    pub cert_gb:      u8,
    pub cert_us:      u8,
    /// Minutes: movie runtime or median episode runtime, 0 when unknown
    pub runtime:      u32,
}

impl RowItem {
    pub fn from_catalog(item: &CatalogItem) -> Self {
        Self {
            id:            item.id,
            tmdb_id:       item.tmdb_id,
            title:         item.title.clone(),
            media_type:    item.media_type.clone(),
            overview:      item.overview.clone(),
            poster_path:   item.poster_path.clone(),
            backdrop_path: item.backdrop_path.clone(),
            vote_average:  item.vote_average,
            popularity:    item.popularity,
            release_year:  item.release_year,
            genres:        item.genres.clone(),
            tv_type:       item.tv_type.clone(),
            language:      item.original_language.clone(),
            cert:          item.cert,
            cert_gb:       item.cert_gb,
            cert_us:       item.cert_us,
            runtime:       item.runtime,
        }
    }
}

// ── PageContext ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct PageContext<'a> {
    pub catalog:     &'a CatalogMap,
    pub vectors:     &'a VectorStore,
    pub hnsw:        &'a RwLock<Option<HnswIndex>>,
    /// Index over the title profile vectors; row definitions are matched against it when it is built
    pub seeds:       &'a ClusterSeeds,
    /// World, atom and theme group of every title (single database)
    pub groups:      &'a crate::groups::GroupMap,
    pub group_names: &'a crate::groups::GroupNames,
    /// Vibe groups, moods and seasonal sub groups
    pub vibe:        &'a crate::vibe::VibeIndex,
    /// The client's local time is between 22:00 and 04:59: late night rows come first
    pub late_night:  bool,
    pub vault:       &'a CentroidVault,
    pub taxonomy:    &'a TaxonomyStore,
    pub profiles:    &'a crate::profiles::ProfileStore,
    pub topics:      &'a crate::topics::TopicStore,
    /// Release dates of new and upcoming titles, and today's date (YYYY-MM-DD) to compare them with
    pub fresh:       &'a crate::fresh::FreshStore,
    pub today_ymd:   String,
    /// Original language code (a language page); every row holds only titles in that language
    pub language:    Option<&'a str>,
    /// Children's home only: build the page from the animated titles (`Some(true)`) or from the live action ones (`Some(false)`)
    pub animation:   Option<bool>,
    /// The Kids home is built from the Kids row vocabulary (`taxonomy` is then that vocabulary): rows chosen by value and variety, none at a fixed place
    pub kids_entries: bool,
    pub surface:     &'a str,   // "movie" | "tv" | "both"
    /// Optional genre as sent by the client (kept for row titles).
    pub genre:       Option<&'a str>,
    /// The rule that genre resolves to, applied to every row (see genre_spec.rs).
    pub genre_spec:  Option<&'a crate::genre_spec::GenreSpec>,
    pub taste:       Vec<f32>,  // blended query vector on unit sphere
    pub target_rows: usize,
    /// Corpus mean vector for centred scoring
    pub mean:        &'a [f32],
    /// Today's (month, day) for calendar rows
    pub today:       (u8, u8),
    /// Session id: seeds the exploration noise so a session sees a stable page
    pub session:     &'a str,
    /// Maturity of the verified caller's profile; none for anonymous requests (see identity.rs)
    pub maturity:    Option<crate::identity::Maturity>,
    /// Fill pass of a thin genre page: the pool is these titles (neighbours of what the page shows) instead of the whole catalog
    pub fill_pool:   Option<&'a [u32]>,
    /// Titles a fill pass must not use again (the ones the page already shows)
    pub skip:        Option<&'a HashSet<u32>>,
    /// What the client sent about the profile: saved, recent and liked titles (personal rows)
    pub signals:     &'a crate::signals::Signals,
}

/// The page's genre rule. The Pride page trusts the model's profile when the title has one: LGBTQ+ must be central to the story (measured
/// 51 of 53 correct on titles graded by hand, against 18 of 50 for "present"); a title without a profile keeps the keyword rule.
pub fn genre_ok_with(spec: Option<&crate::genre_spec::GenreSpec>, profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    let Some(spec) = spec else { return true };
    if matches!(spec, crate::genre_spec::GenreSpec::Lgbtq) {
        if let Some(p) = profiles.get(item.id) {
            return p.lgbtq == crate::profiles::Lgbtq::Central;
        }
    }
    spec.matches(item)
}

/// May this tier see the title? A real certificate decides (see `Maturity::allows`). A title with no certificate but a profile is judged by
/// the profile (strict rule for children, no sexual content for teenagers); without either the older genre and keyword rules apply.
pub fn maturity_allows_with(m: crate::identity::Maturity, profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    use crate::identity::Maturity;
    if item.cert == 0 && !matches!(m, Maturity::Adult) {
        if let Some(p) = profiles.get(item.id) {
            if item.adult || item.flags & crate::catalog::FLAG_SUGGESTIVE != 0 {
                return false;
            }
            return match m {
                Maturity::LittleKids | Maturity::OlderKids => p.suits_children(),
                Maturity::Twelve => p.suits_teens(),
                _ => !p.content.iter().any(|c| matches!(c.as_str(), "sex" | "nudity")),
            };
        }
    }
    m.allows(item)
}

/// A page of an adult genre (Action, Thriller, Pride...) does not carry children's titles for an adult or an anonymous caller. Kids profiles are
/// limited by their maturity instead, and pages about kids, family, animation or anime keep them.
pub fn audience_fits_with(spec: Option<&crate::genre_spec::GenreSpec>, maturity: Option<crate::identity::Maturity>, profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    let Some(spec) = spec else { return true };
    if spec.keeps_children() || maturity.is_some_and(|m| m.is_kids()) {
        return true;
    }
    if item.cert != 0 {
        return item.cert >= 3; // U and PG are children's and family ratings
    }
    if let Some(p) = profiles.get(item.id) {
        return p.audience & (crate::profiles::TEEN | crate::profiles::ADULT) != 0;
    }
    !item.genres.iter().any(|g| g.eq_ignore_ascii_case("Kids") || g.eq_ignore_ascii_case("Family"))
}

impl<'a> PageContext<'a> {
    pub(crate) fn matches_filters(&self, item: &CatalogItem) -> bool {
        (self.surface == "both" || item.media_type == self.surface || item.media_type == "both")
            && self.language.is_none_or(|l| language_matches(l, &item.original_language))
            && self.genre_ok(item)
            && self.audience_fits(item)
            && self.maturity.is_none_or(|m| self.maturity_allows(m, item))
            && (!safety_nets() || self.allowed_by_nets(item))
    }

    fn genre_ok(&self, item: &CatalogItem) -> bool {
        genre_ok_with(self.genre_spec, self.profiles, item)
    }

    pub(crate) fn maturity_allows(&self, m: crate::identity::Maturity, item: &CatalogItem) -> bool {
        maturity_allows_with(m, self.profiles, item)
    }

    fn audience_fits(&self, item: &CatalogItem) -> bool {
        audience_fits_with(self.genre_spec, self.maturity, self.profiles, item)
    }

    /// Vote count floor: a rating from a handful of votes says nothing. Known only for pulled titles (all of the
    /// quality pool); genre pages use a lower floor so small collections survive.
    pub(crate) fn reliable(&self, item: &CatalogItem) -> bool {
        meets_vote_floor(item, self.genre_spec.is_some() || self.language.is_some())
    }

    /// News, talk and soap never fill default rows (a page that asks for them keeps them); seasonal titles only
    /// appear inside their window (the Halloween page lifts the Halloween rule).
    pub(crate) fn allowed_by_nets(&self, item: &CatalogItem) -> bool {
        use crate::catalog::{FLAG_CHRISTMAS, FLAG_HALLOWEEN, FLAG_NONSCRIPTED, FLAG_SUGGESTIVE, FLAG_WRESTLING_EVENT};
        let f = item.flags & crate::catalog::FLAGS_RULES;
        if f == 0 {
            return true;
        }
        if f & FLAG_SUGGESTIVE != 0 {
            return false;
        }
        if f & FLAG_NONSCRIPTED != 0 && !self.genre_spec.is_some_and(|s| s.includes_nonscripted()) {
            return false;
        }
        // pay per view wrestling events are not default home content (the Sports page keeps them)
        if f & FLAG_WRESTLING_EVENT != 0 && !matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Sports)) {
            return false;
        }
        let (m, d) = self.today;
        let md = (m as u32) * 100 + d as u32;
        if f & FLAG_CHRISTMAS != 0 && !(md >= 1120 || md <= 106) {
            return false;
        }
        let halloween_page = matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Halloween));
        if f & FLAG_HALLOWEEN != 0 && !halloween_page && !(1001..=1101).contains(&md) {
            return false;
        }
        true
    }

    // ── HNSW / brute-force retrieval ─────────────────────────────────────────

    /// Retrieve top-k candidates for a query vector.
    /// HNSW when index is ready; rayon brute-force otherwise.
    async fn retrieve_candidates(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if ENGINE_READY.load(Ordering::Relaxed) {
            let guard = self.hnsw.read().await;
            if let Some(idx) = guard.as_ref() {
                if let Ok(mut results) = idx.search(query, k) {
                    for (id, sim) in &mut results {
                        if let Some(item) = self.catalog.get(id) {
                            *sim = rerank_score(*sim, item.popularity);
                        }
                    }
                    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    return results;
                }
            }
        }
        self.vectors.brute_force_topk(query, k)
    }

    /// Synchronous variant of `retrieve_candidates` for use inside rayon, with the
    /// HNSW read guard already held by the caller.
    pub(crate) fn retrieve_sync(&self, index: &Option<HnswIndex>, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if ENGINE_READY.load(Ordering::Relaxed) {
            if let Some(idx) = index.as_ref() {
                if let Ok(mut results) = idx.search(query, k) {
                    for (id, sim) in &mut results {
                        if let Some(item) = self.catalog.get(id) {
                            *sim = rerank_score(*sim, item.popularity);
                        }
                    }
                    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    return results;
                }
            }
        }
        self.vectors.brute_force_topk(query, k)
    }

    // ── Main page assembly ────────────────────────────────────────────────────

    /// The page for this request: rows from the builder for the page kind, then the Top 10 rows are placed by value among
    /// them (see `place_charts`).
    pub async fn build_page(&self) -> Vec<Row> {
        let rows = self.build_page_rows().await;
        let mut rows = self.place_charts(rows);
        // another version of the same title (a remake, a re-release) is not a second suggestion in one row
        for row in &mut rows {
            let mut seen: HashSet<(String, String)> = HashSet::new();
            row.items.retain(|i| seen.insert((i.media_type.clone(), crate::people::normalise(&i.title))));
        }
        rows
    }

    /// The rows of the page: the topic rows, with up to two seasonal rows (from the row vocabulary, only inside their calendar window) placed near the top
    /// of the home, films and series pages.
    async fn build_page_rows(&self) -> Vec<Row> {
        let mut rows = self.build_page_rows_inner().await;
        let kids = self.maturity.is_some_and(|m| m.is_kids());
        if self.genre.is_none() && self.language.is_none() && self.animation.is_none() && !kids && !self.taxonomy.entries.is_empty() {
            // the collection rows (an event's banner, a mood) are built first so the seasonal rows take what is left
            let used0: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let collections = self.build_collection_rows(&used0, 4);
            rows.extend(collections);
            let used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let titles: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
            // sub groups of the season's pool when they exist, the flat keyword rows otherwise
            let seasonal = { let sub = self.build_seasonal_sub_rows(&used); if sub.is_empty() { self.build_seasonal_rows(&used).await } else { sub } };
            for (n, row) in seasonal.into_iter().filter(|r| !titles.contains(&r.title.to_lowercase())).enumerate() {
                let at = (if n == 0 { 2 } else { 5 + n }).min(rows.len());
                rows.insert(at, row);
            }
        }
        // a children's page gets its banner collections (named ones and the ones in season), placed by slot like the adult banners
        if kids && self.kids_entries && self.genre.is_none() && self.language.is_none() && self.animation.is_none() {
            let used0: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let banners = self.build_kids_collection_rows(&used0, 3).await;
            rows.extend(banners);
        }
        // late night moods lead a night page, vibe rows (how the titles feel) are mixed in on every page that is not a children's or a language page
        if !kids && self.language.is_none() && self.animation.is_none() {
            let used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let mut extra: Vec<Row> = Vec::new();
            if self.late_night && self.genre.is_none() { extra.extend(self.build_mood_rows(&used, 2, true)); }
            let mut taken = used.clone(); extra.iter().for_each(|r| r.items.iter().for_each(|i| { taken.insert(i.id); }));
            // topic rows that carry only a genre name ("Comedies", "Documentary & Drama") are replaced by rows named for how their titles feel
            let before = rows.len();
            if self.genre.is_none() { rows.retain(|r| !(r.kind.is_empty() && r.archetype == "thematic" && is_plain_genre_title(&r.title))); }
            let removed = before - rows.len();
            let titles: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
            let vibes: Vec<Row> = self.build_vibe_rows(&taken, 3 + removed.min(8)).into_iter().filter(|r| !titles.contains(&r.title.to_lowercase())).collect();
            let night = extra.len();
            let mut taken_all = taken.clone(); vibes.iter().for_each(|r| r.items.iter().for_each(|i| { taken_all.insert(i.id); }));
            let attrs: Vec<Row> = self.build_attribute_rows(&taken_all, 2).into_iter().filter(|r| !titles.contains(&r.title.to_lowercase())).collect();
            rows = self.interleave(rows, extra, 1);
            rows = self.interleave(rows, vibes, 3 + night);
            rows = self.interleave(rows, attrs, 5 + night);
            // topic rows (what the titles are about), a few for each page
            let on_page: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
            let taken_topics: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let topic_rows = self.build_topic_rows(&taken_topics, if self.genre.is_none() { 4 } else { 3 }, &on_page);
            rows = self.interleave(rows, topic_rows, 4 + night);
            // one mood row among the topic rows on a page without a genre
            if self.genre.is_none() && !self.late_night {
                let taken_now: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
                let on_page: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
                let moods: Vec<Row> = self.build_mood_rows(&taken_now, 3, false).into_iter().filter(|r| !on_page.contains(&r.title.to_lowercase())).take(1).collect();
                rows = self.interleave(rows, moods, 8);
            }
        }
        // personal rows, New Releases and collections: placed by slot in `place_charts`
        if !kids && self.language.is_none() && self.animation.is_none() {
            let taken_now: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let mut extra = self.build_personal_rows(&taken_now).await;
            if self.genre.is_none() {
                let mut taken_all = taken_now.clone();
                extra.iter().for_each(|r| r.items.iter().for_each(|i| { taken_all.insert(i.id); }));
                if let Some(r) = self.build_new_row(&taken_all) { r.items.iter().for_each(|i| { taken_all.insert(i.id); }); extra.push(r); }
            }
            rows.extend(extra);
        }
        // a row named for adult content does not belong on a general page
        rows.retain(|r| !is_suggestive_title(&r.title));
        // a row of fewer than 8 titles looks unfinished next to full rows
        if self.genre.is_none() { rows.retain(|r| r.meta.is_some() || r.items.len() >= 8); }
        rows
    }



    async fn build_page_rows_inner(&self) -> Vec<Row> {
        // A children's profile gets a page built from the titles it may see (clustered like a genre page) instead of the global topic
        // seeds, which are mostly adult topics with a few allowed titles left in each.
        let kids_home = self.maturity.is_some_and(|m| m.is_kids()) && self.profiles.has_profiles();
        if kids_home && self.language.is_none() && self.animation.is_none() {
            if self.kids_entries {
                // a genre page keeps the rows that still have enough titles of the genre; a page that stays thin falls back to the clustered genre page
                let rows = self.build_entry_page().await;
                if rows.len() > if self.genre.is_none() { 4 } else { 8 } {
                    return rows;
                }
            }
            if self.genre.is_none() {
                return self.build_kids_home();
            }
        }
        if self.genre.is_some() || self.language.is_some() || kids_home {
            let rows = self.build_genre_page();
            return self.fill_thin_genre_page(rows).await;
        }
        // The page built wholly from row definitions is off unless asked for (STREAMS_ENTRY_PAGES=1): measured 2026-10-13, its row names are vaguer and it fills
        // fewer rows than the topic rows; the vocabulary is used for seasonal rows (build_seasonal_rows).
        if std::env::var("STREAMS_ENTRY_PAGES").is_ok() && self.taxonomy.entries.len() >= MIN_TAXONOMY_FOR_ENTRY_ROWS {
            let rows = self.build_entry_page().await;
            if rows.len() > 2 {
                return rows;
            }
        }
        let mut state = PageState::new();
        let mut rows: Vec<Row> = Vec::new();

        // thematic_centroids tracks cluster centroids in row-placement order,
        // parallel to the thematic subset of `rows`. Used for Hungarian alignment.
        let mut thematic_centroids: Vec<Vec<f32>> = Vec::new();
        let mut thematic_row_indices: Vec<usize> = Vec::new(); // indices into `rows`
        let mut thematic_seed_idx: Vec<usize> = Vec::new(); // the seed each thematic row came from, parallel to thematic_row_indices

        // ── Fixed rows ────────────────────────────────────────────────────────
        if let Some(row) = self.build_trending_row() {
            rows.push(row);
        }

        // ── Taste-biased seed scoring ─────────────────────────────────────────
        // Score each cluster seed by cosine similarity to taste vector.
        // Cold start (zero taste vector) → all scores equal → pure MMD order.
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let mut scored_seeds: Vec<(f32, usize)> = self
            .seeds
            .seeds
            .iter()
            .enumerate()
            .map(|(i, seed)| {
                let base = if has_taste {
                    cosine_sim(&self.taste, &seed.centroid)
                } else {
                    1.0
                };
                // exploration: a session reorders the seeds a little (stable for the session)
                let noise = if self.session.is_empty() || !safety_nets() { 0.0 } else { SEED_NOISE * entry_noise(self.session, i) };
                (base + noise, i)
            })
            .collect();

        // Sort by relevance descending (preserves MMD order for equal scores)
        scored_seeds.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // ── Thematic rows from cluster seeds ──────────────────────────────────
        let t_seed_loop = std::time::Instant::now();
        let (mut n_visit, mut n_similar, mut n_empty, mut n_genre, mut n_mmr, mut n_items, mut n_row) = (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
        for (_, seed_idx) in &scored_seeds {
            if rows.len() > self.target_rows {
                break; // +1 for trending row
            }

            let seed = &self.seeds.seeds[*seed_idx];
            n_visit += 1;

            // MMD inter-anchor gate: reject if too similar to any placed centroid
            let too_similar = thematic_centroids
                .iter()
                .any(|c| cosine_sim(&seed.centroid, c) > MMD_MAX_INTER_SIM);
            if too_similar {
                n_similar += 1;
                continue;
            }

            // Members that pass surface/genre and quality filters (no vectors yet)
            let mut cand_items: Vec<&CatalogItem> = seed
                .member_ids
                .iter()
                .filter_map(|id| self.catalog.get(id))
                .filter(|item| self.matches_filters(item) && self.member_ok(item) && self.reliable(item))
                .collect();

            if cand_items.is_empty() {
                n_empty += 1;
                continue;
            }

            // Dominant genre (majority vote)
            let dominant_genre =
                majority_genre(cand_items.iter().map(|item| item.genres.as_slice()));

            // Documentary and reality rows are capped on default pages
            let nonscripted_row = dominant_genre == "Documentary" || dominant_genre == "Reality";
            if nonscripted_row && self.genre.is_none() && safety_nets() && state.nonscripted_rows >= NONSCRIPTED_ROW_CAP {
                n_genre += 1;
                continue;
            }

            // Genre bitmask: no same genre in last GENRE_WINDOW rows.
            // Checked before any vector work so rejected seeds stay cheap.
            // A genre filtered page is one genre by design, so the rule does not apply.
            if self.genre.is_none() && !state.can_place_genre(&dominant_genre) {
                n_genre += 1;
                continue;
            }

            // Best quality members only, then fetch their vectors
            crate::quality::sort_by_quality(&mut cand_items);
            cand_items.truncate(MEMBER_POOL);
            if !self.session.is_empty() && safety_nets() && cand_items.len() > 60 {
                let kept: Vec<&CatalogItem> = cand_items
                    .iter()
                    .copied()
                    .filter(|it| entry_noise(self.session, it.id as usize) >= MEMBER_DROP)
                    .collect();
                if kept.len() >= 45 {
                    cand_items = kept;
                }
            }
            let members: Vec<(&CatalogItem, Vec<f32>)> = cand_items
                .into_iter()
                .filter_map(|item| {
                    let idx = self.vectors.index_of(item.id)?;
                    Some((item, self.vectors.get_f32(idx)))
                })
                .collect();

            if members.is_empty() {
                continue;
            }

            // MMR: intra-row diversity
            let candidates: Vec<(u32, Vec<f32>)> = members
                .iter()
                .map(|(item, vec)| (item.id, vec.clone()))
                .collect();
            n_mmr += 1;
            let selected_ids =
                greedy_mmr(&seed.centroid, &candidates, ROW_SIZE * 2, MMR_LAMBDA);

            let (sel_ids, sel_vecs): (Vec<u32>, Vec<Vec<f32>>) = selected_ids
                .iter()
                .filter_map(|id| {
                    let idx = self.vectors.index_of(*id)?;
                    Some((*id, self.vectors.get_f32(idx)))
                })
                .unzip();

            // Coherence filter (outliers by medoid distance)
            let (fil_ids, fil_vecs) = filter_outliers(&sel_ids, &sel_vecs, COHERENCE_OK);

            // Coherence gate: weak clusters need more members to pass
            let coherence = mean_pairwise_cosine(&fil_vecs);
            let min_items = if coherence < COHERENCE_WEAK {
                COHERENCE_MIN_ITEMS + 4
            } else {
                COHERENCE_MIN_ITEMS
            };
            if fil_ids.len() < min_items {
                n_items += 1;
                continue;
            }

            // Cluster centroid from filtered items
            let centroid = sphere_mean(&fil_vecs);

            // Build row items (up to ROW_SIZE)
            let row_items: Vec<RowItem> = self.diverse_titles(&fil_ids, ROW_SIZE);

            if row_items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }

            // Mix movies and series where the theme has both (surfaces that show both types)
            let mut row_items = row_items;
            if self.surface == "both" {
                let majority_tv = row_items.iter().filter(|r| r.media_type == "tv").count() * 2 >= row_items.len();
                let other = if majority_tv { "movie" } else { "tv" };
                let mut exclude = state.used_ids.clone();
                exclude.extend(row_items.iter().map(|r| r.id));
                let cands = self.other_type_neighbours(&centroid, other, &exclude).await;
                self.blend_other_type(&mut row_items, &centroid, other, cands);
            }
            state.used_ids.extend(row_items.iter().map(|r| r.id));
            if nonscripted_row {
                state.nonscripted_rows += 1;
            }

            // Record state
            n_row += 1;
            let row_index = rows.len();
            thematic_row_indices.push(row_index);
            thematic_seed_idx.push(*seed_idx);
            thematic_centroids.push(centroid.clone());
            state.record_thematic(dominant_genre, centroid);

            rows.push(Row {
                title:     "__PENDING__".to_string(), // replaced after Hungarian below
                archetype: "thematic".to_string(),
                kind:      String::new(),
                items:     row_items, meta: None,
            });
        }

        tracing::debug!(
            "seed loop: {:.0} ms, visited {n_visit}, too similar {n_similar}, no members {n_empty}, genre window {n_genre}, mmr runs {n_mmr}, too few items {n_items}, rows {n_row}",
            t_seed_loop.elapsed().as_secs_f64() * 1000.0
        );

        // ── Optional BYW row ──────────────────────────────────────────────────
        // Only added if there's a taste vector. Anchor must be thematically
        // distinct from all placed thematic rows (cosine < BYW_MAX_CENTROID_SIM).
        if has_taste {
            if let Some(byw) = self.build_byw_row(&state).await {
                rows.push(byw);
            }
        }

        // ── Hungarian shelf alignment ─────────────────────────────────────────
        // Assign unique taxonomy titles to all thematic rows simultaneously.
        // thematic_centroids is PARALLEL to thematic_row_indices — no index arithmetic.
        let taxonomy_titles = self.taxonomy.hungarian_assign(
            &thematic_centroids,
            self.surface,
            TAXONOMY_THRESHOLD,
        );

        for (ti, &row_idx) in thematic_row_indices.iter().enumerate() {
            let dominant_genre = majority_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
            );
            let taxonomy_title = taxonomy_titles.get(ti).and_then(|t| t.clone());
            let era = dominant_era(&rows[row_idx].items);
            let second = second_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
                &dominant_genre,
            );
            let has_taxonomy = taxonomy_title.is_some();
            let mut derived = derive_title(
                &dominant_genre, second.as_deref(), era.as_deref(),
                rows[row_idx].items.first().map(|i| i.title.as_str()),
                taxonomy_title,
            );
            let row_catalog: Vec<&CatalogItem> = rows[row_idx].items.iter().filter_map(|i| self.catalog.get(&i.id)).collect();
            let mut candidates = if has_taxonomy && !derived.is_empty() { vec![derived.remove(0)] } else { Vec::new() };
            // the name written for the group comes first; the titles derived from the displayed items follow as fallbacks
            if let Some(seed) = thematic_seed_idx.get(ti).and_then(|i| self.seeds.seeds.get(*i)) {
                if !seed.name.is_empty() && crate::row_label::name_fits(&seed.name, &row_catalog) && !entry_names_other_kind(&seed.name, self.surface) { candidates.insert(0, seed.name.clone()); }
            }
            candidates.extend(crate::row_label::candidates(&row_catalog));
            candidates.extend(derived);
            let candidates = crate::row_label::mark_animated(candidates, &row_catalog);
            let title = state.claim_title(&candidates);
            rows[row_idx].title = clamp_words(&title, 6);
        }

        rows
    }

    // ── Entry driven pages ────────────────────────────────────────────────────


    // ── Vibe, mood and seasonal rows (notes/52) ───────────────────────────────

    /// Filters of the page without its genre rule (a seasonal page holds titles that are near the season without carrying its keyword).
    pub(crate) fn matches_filters_any_genre(&self, item: &CatalogItem) -> bool {
        (self.surface == "both" || item.media_type == self.surface || item.media_type == "both")
            && self.language.is_none_or(|l| language_matches(l, &item.original_language))
            && self.audience_fits(item)
            && self.maturity.is_none_or(|m| self.maturity_allows(m, item))
            && (!safety_nets() || self.allowed_by_nets(item))
    }

    /// The titles of a list of keys that this page may show, best quality first; `any_genre` skips the page's genre rule.
    /// The quality gate of this page (see `member_quality_for`).
    pub(crate) fn member_ok(&self, item: &CatalogItem) -> bool {
        member_quality_for(self.genre_spec, item)
    }

    pub(crate) fn eligible(&self, keys: &[u32], taken: &HashSet<u32>, any_genre: bool) -> Vec<&CatalogItem> {
        let mut v: Vec<&CatalogItem> = keys
            .iter()
            .filter_map(|k| self.catalog.get(k))
            .filter(|i| !taken.contains(&i.id) && (if any_genre { self.matches_filters_any_genre(i) } else { self.matches_filters(i) }) && self.member_ok(i) && self.reliable(i))
            .collect();
        crate::quality::sort_by_quality(&mut v);
        v
    }

    /// A row of these titles in the order given (no re-ordering for variety): the profile's own lists, the new releases, the lead titles of a collection.
    pub(crate) fn row_of_any(&self, title: &str, kind: &str, items: &[&CatalogItem]) -> Option<Row> {
        let row_items: Vec<RowItem> = items.iter().take(ROW_SIZE).map(|i| RowItem::from_catalog(i)).collect();
        if row_items.len() < 3 { return None; }
        Some(Row { title: title.to_string(), archetype: "thematic".to_string(), kind: kind.to_string(), items: row_items, meta: None })
    }

    pub(crate) fn row_of(&self, title: &str, kind: &str, items: &[&CatalogItem]) -> Option<Row> {
        let ids: Vec<u32> = items.iter().take(60).map(|i| i.id).collect();
        let row_items = self.diverse_titles(&ids, ROW_SIZE);
        if row_items.len() < COHERENCE_MIN_ITEMS { return None; }
        Some(Row { title: title.to_string(), archetype: "thematic".to_string(), kind: kind.to_string(), items: row_items, meta: None })
    }

    /// A vibe name is shown only when it keeps its promise for the titles of the row and does not name the other kind of title (Series on a films page).
    pub(crate) fn vibe_name_ok(&self, name: &str, items: &[&CatalogItem]) -> bool {
        crate::row_label::name_fits(name, items) && !entry_names_other_kind(name, self.surface)
    }



    /// The row of one mood: titles of its vibe groups ranked by their fit to the mood phrase; None when fewer than 12 titles are free.
    pub(crate) fn mood_row(&self, m: &crate::vibe::Mood, used: &HashSet<u32>) -> Option<Row> {
        let mut cand: Vec<(f32, &CatalogItem)> = Vec::new(); let mut seen: HashSet<u32> = HashSet::new();
        // the phrase vector, centred like the title vectors; a title is ranked by its own fit to the phrase as well as by the review match of its vibe group
        let mv: Option<Vec<f32>> = (m.vector.len() == DIMS).then(|| { let mut c: Vec<f32> = m.vector.iter().zip(self.mean).map(|(x, mu)| x - mu).collect(); l2_normalize(&mut c); c });
        for (g, score) in m.groups.iter().take(6) {
            if let Some(keys) = self.vibe.members.get(g) {
                for it in self.eligible(keys, used, false).into_iter().take(60) {
                    if !seen.insert(it.id) { continue; }
                    let fit = match &mv {
                        Some(v) => {
                            let Some(ix) = self.vectors.index_of(it.id) else { continue };
                            if self.vectors.is_zero_row(ix) { continue; }
                            let mut c: Vec<f32> = self.vectors.get_f32(ix).iter().zip(self.mean).map(|(x, mu)| x - mu).collect();
                            l2_normalize(&mut c);
                            cosine_sim(&c, v)
                        }
                        None => 0.0,
                    };
                    cand.push((fit + 0.5 * *score + 0.1 * (crate::quality::item_quality(it) / 35.0).min(1.5), it));
                }
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.id.cmp(&b.1.id)));
        let items: Vec<&CatalogItem> = cand.into_iter().map(|c| c.1).collect();
        if items.len() < 12 { return None; }
        self.row_of(&m.title, "mood", &items)
    }

    /// Mood rows (late night, cozy, mind-bending ...): the titles of the vibe groups that fit the mood best. On a late night page the late night moods come first.
    pub(crate) fn build_mood_rows(&self, taken: &HashSet<u32>, want: usize, late_only: bool) -> Vec<Row> {
        if want == 0 || self.vibe.moods.is_empty() { return Vec::new(); }
        let mut moods: Vec<&crate::vibe::Mood> = self.vibe.moods.iter().filter(|m| !late_only || m.late).collect();
        moods.sort_by(|a, b| entry_noise(self.session, b.id.len() + b.title.len()).partial_cmp(&entry_noise(self.session, a.id.len() + a.title.len())).unwrap_or(std::cmp::Ordering::Equal));
        if self.late_night && !late_only { moods.sort_by_key(|m| !m.late); }
        let mut rows: Vec<Row> = Vec::new(); let mut used: HashSet<u32> = taken.clone();
        for m in moods {
            if rows.len() >= want { break; }
            if let Some(r) = self.mood_row(m, &used) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        rows
    }


    /// Items of a seasonal sub group ordered by how well they fit the event (centred cosine to the event vector, plus a small quality term), the ones that do not fit left out.
    /// A sub group whose best titles fit weakly is dropped (an empty list).
    pub(crate) fn season_fit<'b>(&self, event: &str, items: Vec<&'b CatalogItem>) -> Vec<&'b CatalogItem> {
        let Some(ev) = self.event_vector(event) else { return items };
        let mut scored: Vec<(f32, &CatalogItem)> = items
            .into_iter()
            .filter_map(|it| {
                let ix = self.vectors.index_of(it.id)?;
                if self.vectors.is_zero_row(ix) { return None; }
                let mut c: Vec<f32> = self.vectors.get_f32(ix).iter().zip(self.mean).map(|(x, m)| x - m).collect();
                l2_normalize(&mut c);
                Some((cosine_sim(&c, &ev), it))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.id.cmp(&b.1.id)));
        let top: Vec<f32> = scored.iter().take(15).map(|s| s.0).collect();
        if top.len() < 8 { return Vec::new(); }
        let mean_top = top.iter().sum::<f32>() / top.len() as f32;
        tracing::debug!("season fit {event}: top {mean_top:.3}, best {:.3}", top[0]);
        let floor = if KEYWORD_EVENTS.contains(&event) { f32::MIN } else { SEASON_MIN_FIT };
        if mean_top < floor { return Vec::new(); }
        scored.into_iter().filter(|(f, _)| *f >= floor * 0.8).map(|(_, it)| it).collect()
    }




    /// Inserts extra rows at positions 3, 6, 9 ... (or at the end when the page is shorter), keeping the first rows of the page.
    fn interleave(&self, mut rows: Vec<Row>, extra: Vec<Row>, first: usize) -> Vec<Row> {
        for (n, r) in extra.into_iter().enumerate() {
            let at = (first + 3 * n).min(rows.len());
            rows.insert(at, r);
        }
        rows
    }


    // ── Genre pages ───────────────────────────────────────────────────────────





    /// Build row items from ranked ids; no significant title word is shared by more than FRANCHISE_CAP titles.
    pub(crate) fn diverse_titles(&self, ids: &[u32], n: usize) -> Vec<RowItem> {
        let cap = if safety_nets() { FRANCHISE_CAP } else { usize::MAX };
        let mut per: HashMap<String, usize> = HashMap::new();
        let mut out: Vec<RowItem> = Vec::with_capacity(n);
        for id in ids {
            let Some(item) = self.catalog.get(id) else { continue };
            let words = title_words(&item.title);
            if words.iter().any(|w| per.get(w).copied().unwrap_or(0) >= cap) {
                continue;
            }
            for w in words {
                *per.entry(w).or_insert(0) += 1;
            }
            out.push(RowItem::from_catalog(item));
            if out.len() >= n {
                break;
            }
        }
        out
    }

    // ── Mixing movies and series ──────────────────────────────────────────────



    // ── Placing the Top 10 rows ───────────────────────────────────────────────


    // ── Row builders ──────────────────────────────────────────────────────────




}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Prior weight per row type, mirroring the page policy type mix in
/// ai-engine/config/row_architecture.json (vibe and theme rows dominate).
fn kind_weight(kind: &str) -> f32 {
    match kind {
        "vibe" => 1.0,
        "theme" => 0.85,
        "combo" => 0.7,
        "country" => 0.5,
        "setting" => 0.4,
        "animal" | "job" | "hobby" | "place" | "learning" => 0.6,
        "variant" => 0.5,
        "decade" | "character" | "audience" => 0.35,
        "animation" | "style" | "editorial" => 0.3,
        "identity" => 0.2,
        _ => 0.3,
    }
}

/// Stable value in (0, 1) per (session, entry): exploration without flicker.
/// Significant words of a title (3 or more letters, not a number, not a filler word), lower case, without repeats.
fn title_words(title: &str) -> Vec<String> {
    const FILLER: &[&str] = &[
        "the", "and", "for", "you", "your", "with", "from", "that", "this", "are", "was", "not", "but", "his", "her", "its", "our",
        "their", "who", "why", "how", "what", "when", "where", "into", "over", "under", "off", "out", "one", "two", "new", "old",
        "man", "men", "day", "night", "life", "love", "last", "first", "les", "der", "die", "das", "los", "las", "part", "chapter",
        "season", "episode", "movie", "film", "story", "special",
    ];
    let mut out: Vec<String> = Vec::new();
    for w in title.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if w.chars().count() >= 3 && !w.chars().all(|c| c.is_ascii_digit()) && !FILLER.contains(&w) && !out.iter().any(|o| o == w) {
            out.push(w.to_string());
        }
    }
    out
}

pub(crate) fn entry_noise(session: &str, idx: usize) -> f32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    session.hash(&mut h);
    idx.hash(&mut h);
    ((h.finish() >> 40) as f32) / ((1u64 << 24) as f32)
}

/// A row title that names adult content ("Adult Film Anthologies", "Erotic Thrillers"); "Young Adult" and "Adult Animation" are fine.
fn is_suggestive_title(title: &str) -> bool {
    let t = title.to_lowercase();
    ["adult film", "adult video", "adult entertainment", "adult content"].iter().any(|p| t.contains(p))
        || t.split(|c: char| !c.is_alphanumeric()).any(|w| matches!(w, "erotic" | "erotica" | "sexy" | "porn" | "nudity" | "nude" | "softcore" | "hentai" | "ecchi" | "harem" | "stripper"))
}

/// The genre most of a row's titles carry (at least 40 percent of them), empty when no genre leads.
fn dominant_genre(row: &Row) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for i in &row.items {
        for g in i.genres.iter().take(3) { *counts.entry(g.as_str()).or_insert(0) += 1; }
    }
    counts.into_iter().filter(|(_, n)| *n * 5 >= row.items.len() * 2).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0))).map(|(g, _)| g.to_string()).unwrap_or_default()
}

/// Reorders rows (looking ahead at most eight rows) so that a row's leading genre differs from that of the three rows before it whenever the following rows allow; the order is otherwise kept.
fn spread_by_genre(rows: Vec<Row>) -> Vec<Row> {
    let dom: Vec<String> = rows.iter().map(dominant_genre).collect();
    let mut slots: Vec<Option<Row>> = rows.into_iter().map(Some).collect();
    let mut left: Vec<usize> = (0..slots.len()).collect();
    let mut out: Vec<Row> = Vec::with_capacity(left.len());
    let mut recent: Vec<usize> = Vec::new();
    while !left.is_empty() {
        let pos = left.iter().take(8).position(|&i| dom[i].is_empty() || !recent.iter().rev().take(3).any(|&r| dom[r] == dom[i])).unwrap_or(0);
        let i = left.remove(pos);
        recent.push(i);
        out.extend(slots[i].take());
    }
    out
}

/// A topic row title made only of genre and format words ("Comedies", "Documentary & Drama", "Thriller Picks"), which says nothing about how the titles feel.
fn is_plain_genre_title(title: &str) -> bool {
    const WORDS: &[&str] = &[
        "action", "adventure", "animation", "animated", "anime", "comedy", "comedies", "crime", "documentary", "documentaries", "drama", "dramas", "family", "fantasy", "history", "horror", "music",
        "musical", "mystery", "mysteries", "romance", "romantic", "sci-fi", "science", "fiction", "thriller", "thrillers", "war", "western", "westerns", "reality", "kids", "films", "movies", "series",
        "shows", "tv", "picks", "more", "&", "and", "of", "the",
    ];
    let mut any = false;
    for w in title.split_whitespace() {
        let lw = w.to_lowercase();
        if !WORDS.contains(&lw.as_str()) { return false; }
        any = true;
    }
    any
}

/// Vote count floor (see `PageContext::reliable`): 10 votes on default pages, 5 on genre pages.
/// Whether a title is made for children, for a children's home page: the title's profile says kids or family, or (no profile) it is a Kids, Family
/// or Animation title. A U or PG certificate alone is not enough: an old U-rated Western is safe for a child and still not a child's title.
pub fn kid_interest(profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    match profiles.get(item.id) {
        Some(p) => p.audience & (crate::profiles::KIDS | crate::profiles::FAMILY) != 0,
        None => item.genres.iter().any(|g| matches!(g.as_str(), "Kids" | "Family" | "Animation")),
    }
}

/// Whether a title's original language belongs to a language page: TMDB writes Chinese as `zh` or `cn` and Norwegian as `no` or `nb`.
pub fn language_matches(page: &str, original: &str) -> bool {
    page == original || matches!((page, original), ("zh", "cn") | ("no", "nb"))
}

/// True when a row title names films on a series page or series on a films page.
fn entry_names_other_kind(title: &str, surface: &str) -> bool {
    let t = title.to_lowercase();
    let films = t.split(' ').any(|w| matches!(w, "film" | "films" | "movie" | "movies" | "cinema"));
    let series = t.split(' ').any(|w| matches!(w, "series" | "show" | "shows" | "sitcom" | "sitcoms" | "episodes"));
    (surface == "tv" && films) || (surface == "movie" && series)
}

/// A Top 10 needs real titles: art, a decent rating, scripted or documentary TV.
pub fn chart_ok(item: &CatalogItem) -> bool {
    item.vote_average >= 6.0
        && item.has_poster()
        && item.has_backdrop()
        && item.has_overview()
        && !matches!(item.tv_type.as_deref(), Some("Reality") | Some("Video") | Some("News") | Some("Talk Show") | Some("Soap"))
}

pub fn meets_vote_floor(item: &CatalogItem, genre_page: bool) -> bool {
    !safety_nets() || item.vote_count == 0 || item.vote_count >= if genre_page { 5 } else { 10 }
}

/// Cheap quality gate for cluster members: drops obscure, unrated and stub entries.
pub fn passes_member_quality(item: &CatalogItem) -> bool {
    // The catalog has no vote_count, so popularity is the proxy for a reliable rating.
    // Measured pool: popularity >= 2 with rating >= 6.0 keeps about 38k titles of 700k.
    item.vote_average >= 6.0 && item.popularity >= 2.0 && item.has_overview() && item.has_poster() && (!safety_nets() || item.has_backdrop())
}

/// The quality gate of a page: a page whose members the Colab judge vetted (Stand-up, Cult, Independent ...) accepts less known titles (rating 5.5, popularity 0.5), since
/// the niche is made of them; every other page uses `passes_member_quality`.
pub fn member_quality_for(spec: Option<&crate::genre_spec::GenreSpec>, item: &CatalogItem) -> bool {
    if spec.is_some_and(|s| s.is_judged()) {
        item.vote_average >= 5.5 && item.popularity >= 0.5 && item.has_overview() && item.has_poster() && (!safety_nets() || item.has_backdrop())
    } else {
        passes_member_quality(item)
    }
}

/// Second most common genre, if any.
fn second_genre<'a>(genres_iter: impl Iterator<Item = &'a [String]>, first: &str) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for genres in genres_iter {
        for g in genres {
            if g != first && !g.contains(first) && !first.contains(g.as_str()) {
                *counts.entry(g.as_str()).or_insert(0) += 1;
            }
        }
    }
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(g, _)| g.to_string())
}

/// Decade label ("90s", "2010s") when at least half of the items share a decade.
fn dominant_era(items: &[RowItem]) -> Option<String> {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    let mut total = 0usize;
    for it in items {
        if let Some(y) = it.release_year {
            *counts.entry(y / 10 * 10).or_insert(0) += 1;
            total += 1;
        }
    }
    let (decade, n) = counts.into_iter().max_by_key(|(_, c)| *c)?;
    if total == 0 || n * 10 < total * 7 {
        return None;
    }
    Some(if decade >= 2000 { format!("{decade}s") } else { format!("{}s", decade % 100) })
}


/// Dominant genre from an iterator of genre slices (majority vote).
fn majority_genre<'a>(genres_iter: impl Iterator<Item = &'a [String]>) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for genres in genres_iter {
        for genre in genres {
            *counts.entry(genre.clone()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(g, _)| g)
        .unwrap_or_else(|| "Drama".to_string())
}

#[cfg(test)]
mod suggestive_tests {
    use super::is_suggestive_title;

    #[test]
    fn adult_content_names_are_found_and_genre_names_are_kept() {
        assert!(is_suggestive_title("Adult Film Anthologies"));
        assert!(is_suggestive_title("Erotic Thrillers"));
        assert!(!is_suggestive_title("Young Adult Dramas"));
        assert!(!is_suggestive_title("Adult Animation"));
        assert!(!is_suggestive_title("Cozy Comfort Watches"));
    }
}
