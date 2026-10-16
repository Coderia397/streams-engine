#![allow(dead_code)] // Public API surface — functions used at runtime via dynamic dispatch

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
const SLOT_COLLECTION: [usize; 2] = [4, 11];
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
    /// Release dates of new and upcoming titles, and today's date (YYYY-MM-DD) to compare them with
    pub fresh:       &'a crate::fresh::FreshStore,
    pub today_ymd:   String,
    /// Original language code (a language page); every row holds only titles in that language
    pub language:    Option<&'a str>,
    /// Children's home only: build the page from the animated titles (`Some(true)`) or from the live action ones (`Some(false)`)
    pub animation:   Option<bool>,
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
            let collections = self.build_collection_rows(&used0, 2);
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

    /// The vector a seasonal row ranks a title by: its catalog vector.
    fn season_vector(&self, id: u32) -> Option<Vec<f32>> {
        let idx = self.vectors.index_of(id)?;
        if self.vectors.is_zero_row(idx) { return None; }
        Some(self.vectors.get_f32(idx))
    }

    /// Up to two rows for what is in season (taxonomy entries of type calendar whose window is active today): the entry's description is matched against the title
    /// profiles, the titles are filtered like any page (age limit, quality) and a row needs at least MIN titles that no other row has.
    async fn build_seasonal_rows(&self, used: &HashSet<u32>) -> Vec<Row> {
        const MIN: usize = 8;
        let mut cands: Vec<usize> = self
            .taxonomy
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.kind == "calendar"
                    && e.season.is_some_and(|s| s.active_on(self.today))
                    && (e.media_type == "both" || e.media_type == self.surface || self.surface == "both")
                    && !entry_names_other_kind(&e.title, self.surface)
                    && !e.centered.is_empty()
            })
            .map(|(i, _)| i)
            .collect();
        // a session sees a stable choice among the active entries
        cands.sort_by(|a, b| entry_noise(self.session, *b).partial_cmp(&entry_noise(self.session, *a)).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new();
        let mut taken: HashSet<u32> = used.clone();
        let mut chosen_centred: Vec<&Vec<f32>> = Vec::new();
        for idx in cands.into_iter().take(8) {
            if rows.len() >= 2 { break; }
            let e = &self.taxonomy.entries[idx];
            if chosen_centred.iter().any(|c| cosine_sim(c, &e.centered) > 0.85) { continue; }
            // the titles of the season: those that carry one of the event's keywords (or, for Pride, whose profile says LGBTQ+ is central), ranked by how close
            // their profile is to the row's description and by quality; an event with too few such titles gets no row
            let mut pool: Vec<(f32, u32)> = Vec::new();
            for item in self.catalog.values() {
                let in_season = if e.lgbtq_central {
                    self.profiles.get(item.id).is_some_and(|p| p.lgbtq == crate::profiles::Lgbtq::Central)
                } else {
                    !e.keywords.is_empty() && item.keywords.iter().any(|k| e.keywords.iter().any(|w| w == k))
                };
                if !in_season || taken.contains(&item.id) || !passes_member_quality(item) || !self.reliable(item) || !self.matches_filters(item) {
                    continue;
                }
                let sim = self.season_vector(item.id).map(|v| cosine_sim(&v, &e.embedding)).unwrap_or(0.0);
                pool.push((sim + 0.15 * (crate::quality::item_quality(item) / 35.0).min(1.5), item.id));
            }
            pool.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            let ids: Vec<u32> = pool.into_iter().take(ROW_SIZE * 4).map(|(_, id)| id).collect();
            if ids.len() < MIN { continue; }
            let cand_vecs: Vec<(u32, Vec<f32>)> = ids.iter().filter_map(|id| self.season_vector(*id).map(|v| (*id, v))).collect();
            let picked = greedy_mmr(&e.embedding, &cand_vecs, ROW_SIZE, MMR_LAMBDA);
            let items: Vec<RowItem> = picked.iter().filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog)).collect();
            if items.len() < MIN { continue; }
            items.iter().for_each(|i| { taken.insert(i.id); });
            chosen_centred.push(&e.centered);
            rows.push(Row { title: clamp_words(&e.title, 5), archetype: "thematic".to_string(), kind: e.kind.clone(), items, meta: None });
        }
        rows
    }

    async fn build_page_rows_inner(&self) -> Vec<Row> {
        // A children's profile gets a page built from the titles it may see (clustered like a genre page) instead of the global topic
        // seeds, which are mostly adult topics with a few allowed titles left in each.
        let kids_home = self.maturity.is_some_and(|m| m.is_kids()) && self.profiles.has_profiles();
        if kids_home && self.genre.is_none() && self.language.is_none() && self.animation.is_none() {
            return self.build_kids_home();
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
                .filter(|item| self.matches_filters(item) && passes_member_quality(item) && self.reliable(item))
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

    /// Netflix style page: rank row definitions (taxonomy entries) for this
    /// session first, then fill each chosen row with titles retrieved around the
    /// entry's own vector. Each entry is a row query, so the vocabulary can be
    /// far larger than the number of clusters and every row title is a real
    /// promise about what is inside.
    async fn build_entry_page(&self) -> Vec<Row> {
        const MIN_ROW: usize = 8;
        /// Entries known to fill fewer titles than this are not considered.
        const MIN_SUPPORT: u16 = 12;
        const POOL_PER_ROW: usize = 60;
        /// Hard ceiling on centred cosine between two chosen entries (near duplicates).
        const ENTRY_HARD_SIM: f32 = 0.85;
        /// Soft diversification weight: value = score - MU * (max centred cosine to chosen rows).
        const MU: f32 = 1.1;
        const EXPLORE: f32 = 0.6;
        /// Share of rows chosen by taste. The rest explore, so a concentrated taste
        /// still sees new territory (Spotify BaRT style explore/exploit).
        const EXPLOIT_SHARE: f32 = 0.7;

        let t0 = std::time::Instant::now();
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let taste_c: Vec<f32> = if has_taste {
            let mut c: Vec<f32> = self.taste.iter().zip(self.mean).map(|(x, m)| x - m).collect();
            l2_normalize(&mut c);
            c
        } else {
            Vec::new()
        };

        let mut rows: Vec<Row> = Vec::new();
        let mut used: HashSet<u32> = HashSet::new();
        if let Some(r) = self.build_trending_row() {
            r.items.iter().for_each(|i| { used.insert(i.id); });
            rows.push(r);
        }

        // 1. Score row definitions two ways: for taste (exploit) and for exploration
        struct Scored { idx: usize, exploit: f32, explore: f32 }
        let mut scored: Vec<Scored> = self
            .taxonomy
            .entries
            .par_iter()
            .enumerate()
            .filter(|(i, e)| {
                (e.media_type == "both" || e.media_type == self.surface || self.surface == "both")
                    // a row for "both" that names the other kind (a films row on the series page) stays off this page
                    && !entry_names_other_kind(&e.title, self.surface)
                    && e.season.is_none_or(|s| s.active_on(self.today))
                    && !e.centered.is_empty()
                    && self.taxonomy.support_for(*i, self.surface).is_none_or(|n| n >= MIN_SUPPORT)
            })
            .map(|(i, e)| {
                let prior = 0.25 * kind_weight(&e.kind);
                let season = if e.season.is_some() { 0.6 } else { 0.0 }; // active window: near the top
                let noise = entry_noise(self.session, i);
                let taste_cos = if has_taste { cosine_sim(&taste_c, &e.centered) } else { 0.0 };
                Scored {
                    idx: i,
                    exploit: prior + season + 1.5 * taste_cos + 0.15 * noise,
                    explore: prior + season + EXPLORE * noise,
                }
            })
            .collect();
        let n_eligible = scored.len();

        let want = self.target_rows;
        let exploit_rows = if has_taste { ((want as f32) * EXPLOIT_SHARE).ceil() as usize } else { 0 };
        // Shortlist both rankings so each phase has enough to satisfy the page rules
        let per_list = (want * 4).clamp(60, 200);
        let mut pick_idx: Vec<usize> = Vec::new();
        if has_taste {
            scored.sort_unstable_by(|a, b| b.exploit.partial_cmp(&a.exploit).unwrap_or(std::cmp::Ordering::Equal));
            pick_idx.extend(scored.iter().take(per_list).map(|s| s.idx));
        }
        scored.sort_unstable_by(|a, b| b.explore.partial_cmp(&a.explore).unwrap_or(std::cmp::Ordering::Equal));
        pick_idx.extend(scored.iter().take(per_list).map(|s| s.idx));
        pick_idx.sort_unstable();
        pick_idx.dedup();
        let score_of: HashMap<usize, (f32, f32)> = scored.iter().map(|s| (s.idx, (s.exploit, s.explore))).collect();
        let t_score = t0.elapsed();

        // 2. Retrieve titles per shortlisted entry in parallel
        struct Cand { idx: usize, exploit: f32, explore: f32, hits: Vec<u32> }
        let guard = self.hnsw.read().await;
        let cands: Vec<Cand> = pick_idx
            .par_iter()
            .filter_map(|&idx| {
                let e = &self.taxonomy.entries[idx];
                let hits = self.retrieve_sync(&guard, &e.embedding, ANN_ENTRY_K);
                let good: Vec<(&CatalogItem, f32)> = hits
                    .iter()
                    .filter_map(|(id, sim)| self.catalog.get(id).map(|i| (i, *sim)))
                    .filter(|(i, _)| self.matches_filters(i) && passes_member_quality(i) && self.reliable(i))
                    .collect();
                if good.len() < MIN_ROW {
                    return None;
                }
                let strength = 0.2
                    * good.iter().take(10)
                        .map(|(i, _)| crate::quality::item_quality(i) / 35.0)
                        .sum::<f32>()
                    / good.len().min(10) as f32;
                let (ex, xp) = score_of[&idx];
                Some(Cand {
                    idx,
                    exploit: ex + strength,
                    explore: xp + strength,
                    hits: good.into_iter().take(POOL_PER_ROW).map(|(i, _)| i.id).collect(),
                })
            })
            .collect();
        drop(guard);
        let t_retrieve = t0.elapsed();
        if std::env::var("STREAMS_DEBUG_ENTRIES").is_ok() {
            eprintln!("entries: eligible {n_eligible}, shortlisted {}, with enough titles {} (surface {})", pick_idx.len(), cands.len(), self.surface);
        }

        // 3. Selection: score based diversification (MMR over row definitions) under page rules
        let max_per_type = (((want as f32) * 0.16).ceil() as usize).max(2);
        let max_calendar = 2usize;
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut chosen: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut max_sim: Vec<f32> = vec![0.0; cands.len()];
        let mut alive: Vec<bool> = vec![true; cands.len()];
        let mut last_kind = String::new();

        while chosen.len() < want {
            let exploiting = chosen.len() < exploit_rows;
            let mut best: Option<(usize, f32)> = None;
            let mut relaxed: Option<(usize, f32)> = None; // ignores the adjacency rule
            for (ci, c) in cands.iter().enumerate() {
                if !alive[ci] || max_sim[ci] >= ENTRY_HARD_SIM {
                    continue;
                }
                let e = &self.taxonomy.entries[c.idx];
                let n = *type_counts.get(&e.kind).unwrap_or(&0);
                let cap = if e.kind == "calendar" { max_calendar } else { max_per_type };
                if n >= cap || c.hits.iter().filter(|id| !used.contains(id)).count() < MIN_ROW {
                    continue;
                }
                let base = if exploiting { c.exploit } else { c.explore };
                let value = base - MU * max_sim[ci].max(0.0);
                if relaxed.is_none_or(|(_, v)| value > v) {
                    relaxed = Some((ci, value));
                }
                if (e.kind != last_kind || e.kind.is_empty()) && best.is_none_or(|(_, v)| value > v) {
                    best = Some((ci, value));
                }
            }
            let Some((ci, _)) = best.or(relaxed) else { break };
            alive[ci] = false;
            let e = &self.taxonomy.entries[cands[ci].idx];
            *type_counts.entry(e.kind.clone()).or_insert(0) += 1;
            last_kind = e.kind.clone();
            for (cj, other) in cands.iter().enumerate() {
                if alive[cj] {
                    let s = cosine_sim(&e.centered, &self.taxonomy.entries[other.idx].centered);
                    if s > max_sim[cj] {
                        max_sim[cj] = s;
                    }
                }
            }
            let fresh: Vec<u32> = cands[ci].hits.iter().copied().filter(|id| !used.contains(id)).collect();
            // Reserve ids now so later rows cannot claim them
            fresh.iter().take(ROW_SIZE * 2).for_each(|id| { used.insert(*id); });
            chosen.push((cands[ci].idx, fresh));
        }
        let t_select = t0.elapsed();
        if std::env::var("STREAMS_DEBUG_ENTRIES").is_ok() {
            eprintln!("entries: chosen {} of wanted {want}", chosen.len());
        }

        // 4. Fill rows: MMR over each row's own candidates
        let mut state = PageState::new();
        for (idx, fresh) in chosen.iter() {
            let e = &self.taxonomy.entries[*idx];
            let cand_vecs: Vec<(u32, Vec<f32>)> = fresh
                .iter()
                .take(ROW_SIZE * 2)
                .filter_map(|id| {
                    let i = self.vectors.index_of(*id)?;
                    Some((*id, self.vectors.get_f32(i)))
                })
                .collect();
            let picked = greedy_mmr(&e.embedding, &cand_vecs, ROW_SIZE, MMR_LAMBDA);
            let items: Vec<RowItem> = picked
                .iter()
                .filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog))
                .collect();
            if items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }
            state.record_thematic(e.kind.clone(), e.embedding.clone());
            rows.push(Row {
                title: clamp_words(&e.title, 5),
                archetype: "thematic".to_string(),
                kind: e.kind.clone(),
                items, meta: None,
            });
        }
        let t_fill = t0.elapsed();

        // 5. Taste rows near the top: Top Picks and Because You Watched
        if has_taste {
            let mut extra: Vec<Row> = Vec::new();
            if let Some(byw) = self.build_byw_row(&state).await {
                extra.push(byw);
            }
            let hits = self.retrieve_candidates(&self.taste, ANN_K).await;
            let cand_vecs: Vec<(u32, Vec<f32>)> = hits
                .iter()
                .filter_map(|(id, _)| self.catalog.get(id))
                .filter(|i| self.matches_filters(i) && passes_member_quality(i) && self.reliable(i))
                .take(ROW_SIZE * 3)
                .filter_map(|i| {
                    let idx = self.vectors.index_of(i.id)?;
                    Some((i.id, self.vectors.get_f32(idx)))
                })
                .collect();
            let picked = greedy_mmr(&self.taste, &cand_vecs, ROW_SIZE, MMR_LAMBDA);
            let items: Vec<RowItem> = picked
                .iter()
                .filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog))
                .collect();
            if items.len() >= COHERENCE_MIN_ITEMS {
                extra.insert(0, Row {
                    title: "Top Picks for You".to_string(),
                    archetype: "top_picks".to_string(),
                    kind: String::new(),
                    items, meta: None,
                });
            }
            // After the trending row, before the thematic rows
            let at = rows.len().min(1);
            for (k, r) in extra.into_iter().enumerate() {
                rows.insert(at + k, r);
            }
        }
        tracing::debug!(
            "entry page: {} eligible, {} retrieved, {} rows chosen of {} | score {:?} retrieve {:?} select {:?} fill {:?} taste-rows {:?}",
            n_eligible, cands.len(), chosen.len(), want,
            t_score, t_retrieve - t_score, t_select - t_retrieve, t_fill - t_select, t0.elapsed() - t_fill
        );

        rows
    }

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
    pub(crate) fn eligible(&self, keys: &[u32], taken: &HashSet<u32>, any_genre: bool) -> Vec<&CatalogItem> {
        let mut v: Vec<&CatalogItem> = keys
            .iter()
            .filter_map(|k| self.catalog.get(k))
            .filter(|i| !taken.contains(&i.id) && (if any_genre { self.matches_filters_any_genre(i) } else { self.matches_filters(i) }) && passes_member_quality(i) && self.reliable(i))
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

    /// Up to `want` rows defined by a catalog fact (runtime, format, source), chosen per session among the kinds that suit the page's surface; a title the page already shows is not used again.
    fn build_attribute_rows(&self, taken: &HashSet<u32>, want: usize) -> Vec<Row> {
        use crate::attributes::SPECS;
        let mut order: Vec<(f32, &crate::attributes::AttrSpec)> = SPECS
            .iter()
            .filter(|s| s.media == "both" || self.surface == "both" || s.media == self.surface)
            .map(|s| (entry_noise(self.session, 7919 + s.id.bytes().map(|b| b as usize * 31).sum::<usize>()), s))
            .collect();
        order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut used: HashSet<u32> = taken.clone();
        let mut rows: Vec<Row> = Vec::new();
        for (_, spec) in order {
            if rows.len() >= want { break; }
            let Some(keys) = self.catalog.attr_keys.get(spec.id) else { continue };
            let items = self.eligible(keys, &used, false);
            if items.len() < 12 { continue; }
            // quality order, shaken a little per session so the row is not the same twelve titles every time
            let mut scored: Vec<(f32, &CatalogItem)> = items.into_iter().take(150).enumerate().map(|(rank, i)| (40.0 * entry_noise(self.session, i.id as usize) - rank as f32, i)).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.id.cmp(&b.1.id)));
            let list: Vec<&CatalogItem> = scored.into_iter().map(|s| s.1).collect();
            let title = match self.surface { "movie" => spec.title_movie, "tv" => spec.title_tv, _ => spec.title_both };
            if let Some(r) = self.row_of(title, "attribute", &list) {
                r.items.iter().for_each(|i| { used.insert(i.id); });
                rows.push(r);
            }
        }
        rows
    }

    /// Up to `want` rows of titles that reviewers describe alike (vibe groups), chosen per session, in the page's genre when it has one.
    fn build_vibe_rows(&self, taken: &HashSet<u32>, want: usize) -> Vec<Row> {
        if want == 0 || self.vibe.members.is_empty() { return Vec::new(); }
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let mut groups: Vec<i32> = self.vibe.members.keys().copied().collect();
        groups.sort_unstable();
        // a session sees a stable choice: groups are ordered by a session dependent noise and the first ones that qualify are scored
        groups.sort_by(|a, b| entry_noise(self.session, *b as usize).partial_cmp(&entry_noise(self.session, *a as usize)).unwrap_or(std::cmp::Ordering::Equal));
        let mut scored: Vec<(f32, String, Vec<&CatalogItem>)> = Vec::new();
        for g in groups.into_iter().take(500) {
            let Some(name) = self.group_names.get(&(5, g)) else { continue };
            let items = self.eligible(&self.vibe.members[&g], taken, false);
            if items.len() < 10 || !self.vibe_name_ok(name, &items) { continue; }
            let mass: f32 = items.iter().take(15).map(|i| crate::quality::item_quality(i)).sum::<f32>() / 15.0;
            let score = if has_taste {
                let mut c = vec![0.0f32; DIMS];
                for it in items.iter().take(10) { if let Some(ix) = self.vectors.index_of(it.id) { for (a, x) in c.iter_mut().zip(self.vectors.get_f32(ix)) { *a += x; } } }
                l2_normalize(&mut c);
                cosine_sim(&self.taste, &c)
            } else { mass * (0.7 + 0.6 * entry_noise(self.session, g as usize + 7)) };
            scored.push((score, name.clone(), items));
            if scored.len() >= want * 6 { break; }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new(); let mut used: HashSet<u32> = taken.clone(); let mut names: HashSet<String> = HashSet::new();
        for (_, name, items) in scored {
            if rows.len() >= want { break; }
            if !names.insert(name.to_lowercase()) { continue; }
            let fresh: Vec<&CatalogItem> = items.into_iter().filter(|i| !used.contains(&i.id)).collect();
            if let Some(r) = self.row_of(&name, "vibe", &fresh) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        rows
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

    /// The centred vector of an event: the mean of the centred vectors of its calendar entries (Halloween: Spooky Mystery Films, Gothic Thrillers ...); None when the vocabulary has none.
    fn event_vector(&self, event: &str) -> Option<Vec<f32>> {
        let mut sum = vec![0.0f32; DIMS]; let mut n = 0;
        for e in self.taxonomy.entries.iter().filter(|e| e.kind == "calendar" && e.event == event && e.centered.len() == DIMS) {
            for (a, x) in sum.iter_mut().zip(&e.centered) { *a += x; }
            n += 1;
        }
        if n == 0 { return None; }
        l2_normalize(&mut sum);
        Some(sum)
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

    /// Rows of the calendar events that are in season: each event's pool is split into named sub groups, the page shows two of the first event and one of the second.
    fn build_seasonal_sub_rows(&self, taken: &HashSet<u32>) -> Vec<Row> {
        let mut events: Vec<(&str, f32)> = Vec::new();
        for e in &self.taxonomy.entries {
            let all_year = e.season.is_none_or(|s| s.start == (1, 1) && s.end == (12, 31));
            if e.kind == "calendar" && !all_year && e.season.is_some_and(|s| s.active_on(self.today)) && self.vibe.seasons.contains_key(&e.event) && !events.iter().any(|(x, _)| *x == e.event.as_str()) {
                events.push((e.event.as_str(), entry_noise(self.session, e.event.len() * 31 + e.event.bytes().map(|b| b as usize).sum::<usize>())));
            }
        }
        events.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new(); let mut used: HashSet<u32> = taken.clone();
        for (ei, (ev, _)) in events.into_iter().take(2).enumerate() {
            let want = if ei == 0 { 2 } else { 1 };
            let mut subs: Vec<&crate::vibe::SeasonSub> = self.vibe.seasons[ev].iter().collect();
            subs.sort_by(|a, b| entry_noise(self.session, b.name.len() + b.items.len()).partial_cmp(&entry_noise(self.session, a.name.len() + a.items.len())).unwrap_or(std::cmp::Ordering::Equal));
            let mut got = 0;
            for sub in subs {
                if got >= want { break; }
                if sub.kw_share < SEASON_MIN_KW_SHARE { continue; }
                let items = self.season_fit(ev, self.eligible(&sub.items, &used, true));
                if items.len() < 8 { continue; }
                let title = if self.vibe_name_ok(&sub.name, &items) { sub.name.clone() } else { continue };
                if let Some(r) = self.row_of(&title, "calendar", &items) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); got += 1; }
            }
        }
        rows
    }

    /// The page of a calendar event (Halloween, Pride): its sub groups as rows, in the order of their size.
    fn build_event_page(&self, event: &str) -> Vec<Row> {
        let Some(subs) = self.vibe.seasons.get(event) else { return Vec::new() };
        let mut rows: Vec<Row> = Vec::new();
        if let Some(r) = self.build_trending_row() { rows.push(r); }
        let mut used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
        let mut titles: HashSet<String> = HashSet::new();
        for sub in subs {
            if rows.len() > self.target_rows { break; }
            if sub.kw_share < SEASON_MIN_KW_SHARE { continue; }
            let items = self.season_fit(event, self.eligible(&sub.items, &used, true));
            if items.len() < 8 || !self.vibe_name_ok(&sub.name, &items) || !titles.insert(sub.name.to_lowercase()) { continue; }
            if let Some(r) = self.row_of(&sub.name, "calendar", &items) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        rows
    }

    /// The Moods page: mood rows first (late night moods first at night), vibe rows after them.
    fn build_moods_page(&self) -> Vec<Row> {
        let mut rows: Vec<Row> = Vec::new();
        if let Some(r) = self.build_trending_row() { rows.push(r); }
        let mut used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
        let moods = self.build_mood_rows(&used, self.target_rows, false);
        moods.iter().for_each(|r| r.items.iter().for_each(|i| { used.insert(i.id); }));
        rows.extend(moods);
        if rows.len() < self.target_rows { rows.extend(self.build_vibe_rows(&used, self.target_rows - rows.len())); }
        rows
    }

    /// Inserts extra rows at positions 3, 6, 9 ... (or at the end when the page is shorter), keeping the first rows of the page.
    fn interleave(&self, mut rows: Vec<Row>, extra: Vec<Row>, first: usize) -> Vec<Row> {
        for (n, r) in extra.into_iter().enumerate() {
            let at = (first + 3 * n).min(rows.len());
            rows.insert(at, r);
        }
        rows
    }

    /// A children's home: half the rows from live action titles, half from animated ones, alternating, so the page is not all cartoons. Each half is
    /// a page of its own (clustered from its own titles); the charts come from the live action half, which is built over the whole pool.
    fn build_kids_home(&self) -> Vec<Row> {
        let half = (self.target_rows / 2).max(4) + 1;
        let live = PageContext { animation: Some(false), target_rows: half, ..self.clone() }.build_genre_page();
        let anim = PageContext { animation: Some(true), target_rows: half, ..self.clone() }.build_genre_page();
        let (charts, live_rows): (Vec<Row>, Vec<Row>) = live.into_iter().partition(|r| r.archetype == "trending");
        let anim_rows: Vec<Row> = anim.into_iter().filter(|r| r.archetype != "trending").collect();
        let mut out: Vec<Row> = charts;
        let (mut a, mut l) = (anim_rows.into_iter(), live_rows.into_iter());
        loop {
            match (l.next(), a.next()) {
                (None, None) => break,
                (lr, ar) => {
                    out.extend(lr);
                    out.extend(ar);
                }
            }
        }
        // a row name used twice by the two halves keeps its first use
        let mut seen: HashSet<String> = HashSet::new();
        out.retain(|r| r.archetype == "trending" || seen.insert(r.title.clone()));
        out
    }

    // ── Genre pages ───────────────────────────────────────────────────────────

    /// A genre page with fewer rows than the floor is topped up with rows of neighbouring titles: the titles closest to the mean of what the page shows (any genre, same age and
    /// quality rules, nothing already shown), grouped into rows by the same builder and named by their own groups. Netflix fills thin genre pages the same way.
    async fn fill_thin_genre_page(&self, mut rows: Vec<Row>) -> Vec<Row> {
        let count = |rows: &[Row]| rows.iter().filter(|r| r.archetype != "trending").count();
        if self.genre.is_none() || self.language.is_some() || self.animation.is_some() || count(&rows) == 0 || count(&rows) >= THIN_PAGE_FLOOR {
            return rows;
        }
        let mut titles: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
        // up to three passes, each looking further out around the mean of everything shown so far
        for reach in [3000usize, 9000, 25000] {
            let own = count(&rows);
            if own >= THIN_PAGE_FLOOR { break; }
            let mut mean = vec![0f32; DIMS];
            let mut n = 0usize;
            for it in rows.iter().flat_map(|r| &r.items) {
                if let Some(ix) = self.vectors.index_of(it.id) {
                    for (a, x) in mean.iter_mut().zip(self.vectors.row(ix)) { *a += x.to_f32(); }
                    n += 1;
                }
            }
            if n == 0 { break; }
            l2_normalize(&mut mean);
            let keys: Vec<u32> = {
                let guard = self.hnsw.read().await;
                self.retrieve_sync(&guard, &mean, reach).into_iter().map(|(k, _)| k).collect()
            };
            let used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let ctx = PageContext { genre: None, genre_spec: None, target_rows: THIN_PAGE_FLOOR - own + 4, fill_pool: Some(&keys), skip: Some(&used), ..self.clone() };
            let mut added = 0;
            for r in ctx.build_genre_page_plain() {
                if count(&rows) >= THIN_PAGE_FLOOR { break; }
                if r.archetype == "trending" || !titles.insert(r.title.to_lowercase()) { continue; }
                rows.push(Row { kind: "related".to_string(), ..r });
                added += 1;
            }
            if added == 0 && reach >= 25000 { break; }
        }
        rows
    }

    /// Genre pages: the Moods page and the event pages (Halloween, Pride) are made of their own rows, topped up with the regular genre rows when the event has fewer sub groups than the page has rows.
    fn build_genre_page(&self) -> Vec<Row> {
        if let Some(g) = self.genre {
            let g = g.to_lowercase();
            if g == "moods" && !self.vibe.moods.is_empty() && self.language.is_none() {
                let r = self.build_moods_page();
                if r.len() >= 4 { return r; }
            }
            // any calendar event that has sub groups of titles has a page of its own (halloween, pride, christmas ...)
            let event: Option<String> = match g.as_str() { "halloween" => Some("halloween".to_string()), "pride" | "lgbtq" => Some("pride".to_string()), other if self.vibe.seasons.contains_key(other) => Some(other.to_string()), _ => None };
            if let Some(ev) = event {
                if self.language.is_none() && self.animation.is_none() {
                    let mut rows = self.build_event_page(&ev);
                    if rows.len() >= 5 {
                        if rows.len() <= self.target_rows {
                            let mut used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
                            let mut titles: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
                            for r in self.build_genre_page_plain() {
                                if rows.len() > self.target_rows { break; }
                                if r.archetype == "trending" || !titles.insert(r.title.to_lowercase()) { continue; }
                                let items: Vec<RowItem> = r.items.into_iter().filter(|i| !used.contains(&i.id)).collect();
                                if items.len() >= 8 { items.iter().for_each(|i| { used.insert(i.id); }); rows.push(Row { items, ..r }); }
                            }
                        }
                        return rows;
                    }
                }
            }
        }
        self.build_genre_page_plain()
    }

    /// Rows for a genre filtered page. The global seeds hold too few titles of any
    /// single genre, so cluster the genre's own best titles on the fly instead
    /// (farthest first init, a few Lloyd rounds) to get sub-topics inside the genre.
    fn build_genre_page_plain(&self) -> Vec<Row> {
        const POOL: usize = 3000;
        const MIN_CLUSTER: usize = 8;

        let mut rows: Vec<Row> = Vec::new();
        if let Some(r) = self.build_trending_row() {
            rows.push(r);
        }

        // a children's home holds titles made for children; a genre page chosen by a children's profile (Comedies, Adventure) keeps every title it may see
        let kids_home = self.genre.is_none() && self.language.is_none() && self.maturity.is_some_and(|m| m.is_kids());
        let candidates: Box<dyn Iterator<Item = &CatalogItem> + '_> = match self.fill_pool {
            Some(keys) => Box::new(keys.iter().filter_map(|k| self.catalog.get(k))),
            None => Box::new(self.catalog.values()),
        };
        let mut pool: Vec<&CatalogItem> = candidates
            .filter(|i| passes_member_quality(i) && self.reliable(i) && self.matches_filters(i) && self.skip.is_none_or(|s| !s.contains(&i.id)) && (!kids_home || kid_interest(self.profiles, i)) && self.animation.is_none_or(|a| a == i.genres.iter().any(|g| g == "Animation")))
            .collect();
        // ties are broken by id so a page does not depend on the hash map order of the catalog (which changes at every reload)
        crate::quality::sort_by_quality(&mut pool);
        // Variety: on a page that is not about anime, Japanese language titles fill at most a third of the pool (they are many among the best
        // rated series and would otherwise fill most rows).
        if !matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Anime)) && self.language != Some("ja") {
            let mut japanese = 0usize;
            pool.retain(|i| {
                if i.original_language == "ja" {
                    japanese += 1;
                    japanese <= POOL / 3
                } else {
                    true
                }
            });
        }
        pool.truncate(POOL);

        let items: Vec<(&CatalogItem, Vec<f32>)> = pool
            .into_iter()
            .filter_map(|i| {
                let idx = self.vectors.index_of(i.id)?;
                if self.vectors.is_zero_row(idx) {
                    return None;
                }
                Some((i, self.vectors.get_f32(idx)))
            })
            .collect();
        if items.len() < COHERENCE_MIN_ITEMS {
            return rows;
        }

        // Rows from the groups of the database. Every title of the page's pool belongs to a group at each level of the hierarchy (world level 2 down to level 1, themes in between);
        // the finest level that gives enough groups of MIN_CLUSTER titles or more is used, each such group is a row. A pool too small for any level to give four rows is clustered by
        // vectors instead, and its rows are then named after the group most of their titles share.
        let items_all = items;
        let want_rows = self.target_rows.min(items_all.len() / 20).max(6);
        let mut group_mode: Option<GroupMode> = None; // (level, id) per cluster, and the item indices kept
        if !self.groups.is_empty() {
            let mut best: Option<(u8, Vec<GroupItems>)> = None;
            for level in 0..crate::groups::N_LEVELS {
                let mut by: HashMap<i32, Vec<usize>> = HashMap::new();
                for (i, (it, _)) in items_all.iter().enumerate() {
                    if let Some(g) = self.groups.get(&it.id) { let id = g.at(level); if id >= 0 { by.entry(id).or_default().push(i); } }
                }
                let mut kept: Vec<(i32, Vec<usize>)> = by.into_iter().filter(|(_, ix)| ix.len() >= MIN_CLUSTER).collect();
                kept.sort_by(|x, y| y.1.len().cmp(&x.1.len()).then(x.0.cmp(&y.0)));
                let enough = kept.len() >= want_rows;
                if best.as_ref().is_none_or(|(_, b)| kept.len() > b.len()) || enough { best = Some((level, kept)); }
                if enough { break; }
            }
            if let Some((level, kept)) = best {
                if kept.len() >= 4 {
                    let ids: Vec<(u8, i32)> = kept.iter().map(|(g, _)| (level, *g)).collect();
                    let idx: Vec<usize> = kept.into_iter().flat_map(|(_, ix)| ix).collect();
                    group_mode = Some((ids, idx));
                }
            }
        }
        let cluster_group: Vec<(u8, i32)>;
        let items: Vec<(&CatalogItem, Vec<f32>)>;
        let centers: Vec<Vec<f32>>;
        let assign: Vec<(usize, f32)>;
        let k: usize;
        if let Some((ids, idx)) = group_mode {
            // items of the kept groups, in group order; a group's centre is the mean of its members
            let mut sel: Vec<(&CatalogItem, Vec<f32>)> = Vec::with_capacity(idx.len());
            let mut cid: Vec<usize> = Vec::with_capacity(idx.len());
            let mut taken: HashMap<(u8, i32), usize> = HashMap::new();
            for (c, key) in ids.iter().enumerate() { taken.insert(*key, c); }
            for i in idx {
                let it = items_all[i].0;
                if let Some(g) = self.groups.get(&it.id) {
                    if let Some(&c) = taken.get(&(ids[0].0, g.at(ids[0].0))) { sel.push((it, items_all[i].1.clone())); cid.push(c); }
                }
            }
            let mut cs: Vec<Vec<f32>> = vec![vec![0.0f32; DIMS]; ids.len()];
            for (n, (_, v)) in sel.iter().enumerate() { for (a, x) in cs[cid[n]].iter_mut().zip(v) { *a += x; } }
            for c in cs.iter_mut() { l2_normalize(c); }
            assign = sel.iter().enumerate().map(|(n, (_, v))| (cid[n], cosine_sim(v, &cs[cid[n]]))).collect();
            k = ids.len(); centers = cs; cluster_group = ids; items = sel;
        } else {
            let items_k = items_all;
            let k_means = self.target_rows.min(items_k.len() / MIN_CLUSTER).max(1);
            let (c_, a_) = {
                let items = &items_k;
                let k = k_means;

            // Farthest first initialisation
            let mut centers: Vec<Vec<f32>> = vec![items[0].1.clone()];
            let mut best_sim: Vec<f32> = items.iter().map(|(_, v)| cosine_sim(v, &centers[0])).collect();
            while centers.len() < k {
                let far = best_sim
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let c = items[far].1.clone();
                for (i, (_, v)) in items.iter().enumerate() {
                    let s = cosine_sim(v, &c);
                    if s > best_sim[i] {
                        best_sim[i] = s;
                    }
                }
                centers.push(c);
            }

            let assign_all = |centers: &[Vec<f32>]| -> Vec<(usize, f32)> {
                items
                    .par_iter()
                    .map(|(_, v)| {
                        centers
                            .iter()
                            .enumerate()
                            .map(|(c, cv)| (c, cosine_sim(v, cv)))
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or((0, 0.0))
                    })
                    .collect()
            };

            let mut assign = assign_all(&centers);
            for _ in 0..6 {
                let mut sums = vec![vec![0.0f32; DIMS]; k];
                for (i, &(c, _)) in assign.iter().enumerate() {
                    for (a, x) in sums[c].iter_mut().zip(items[i].1.iter()) {
                        *a += x;
                    }
                }
                for (c, sum) in sums.into_iter().enumerate() {
                    let mut sum = sum;
                    if sum.iter().any(|x| x.abs() > 1e-9) {
                        l2_normalize(&mut sum);
                        centers[c] = sum;
                    }
                }
                assign = assign_all(&centers);
            }

                (centers, assign)
            };
            k = k_means; centers = c_; assign = a_; cluster_group = Vec::new(); items = items_k;
        }

        // Order clusters: taste alignment when there is a taste vector, else size
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let anime_page = matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Anime));
        let mut anime_count = vec![0usize; k];
        // Japanese language titles count with anime for the variety cap below (Japanese superhero and mecha shows are not all flagged as anime)
        let mut japanese_count = vec![0usize; k];
        for (i, &(c, _)) in assign.iter().enumerate() {
            if items[i].0.is_anime {
                anime_count[c] += 1;
            }
            if items[i].0.original_language == "ja" {
                japanese_count[c] += 1;
            }
        }
        let mut order: Vec<(usize, f32)> = (0..k)
            .map(|c| {
                let size = assign.iter().filter(|(a, _)| *a == c).count();
                let key = if has_taste {
                    cosine_sim(&self.taste, &centers[c])
                } else {
                    // largest clusters first, with a session jitter, and an anime cluster does not lead a page that is not about anime
                    let jitter = if self.session.is_empty() || !safety_nets() { 1.0 } else { 0.7 + 0.6 * entry_noise(self.session, c) };
                    let anime_penalty = if !anime_page && size > 0 && anime_count[c] * 10 > size * 6 { 0.5 } else { 1.0 };
                    size as f32 * jitter * anime_penalty
                };
                (c, key)
            })
            .collect();
        order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Variety: on a page that is not about anime, at most two anime clusters sit among the first rows; the rest follow the other clusters.
        if !anime_page {
            let is_anime = |c: usize| {
                let size = assign.iter().filter(|(a, _)| *a == c).count();
                size > 0 && anime_count[c].max(japanese_count[c]) * 10 > size * 4
            };
            let mut kept = 0;
            let (mut early, mut late): (Vec<ScoredIdx>, Vec<ScoredIdx>) = (Vec::new(), Vec::new());
            for entry in order.drain(..) {
                if is_anime(entry.0) {
                    kept += 1;
                    if kept <= 2 { early.push(entry) } else { late.push(entry) }
                } else {
                    early.push(entry)
                }
            }
            early.extend(late);
            order = early;
        }

        let first_thematic = rows.len();
        let mut state = PageState::new();
        let mut placed: Vec<usize> = Vec::new();
        for (c, _) in order {
            let mut members: Vec<(usize, f32)> = assign
                .iter()
                .enumerate()
                .filter(|(_, (a, _))| *a == c)
                .map(|(i, (_, s))| (i, *s))
                .collect();
            if members.len() < MIN_CLUSTER {
                continue;
            }
            members.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            members.truncate(40);

            let candidates: Vec<(u32, Vec<f32>)> = members
                .iter()
                .map(|&(i, _)| (items[i].0.id, items[i].1.clone()))
                .collect();
            let picked = greedy_mmr(&centers[c], &candidates, ROW_SIZE * 2, MMR_LAMBDA);
            let row_items: Vec<RowItem> = self.diverse_titles(&picked, ROW_SIZE);
            if row_items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }
            let mut row_items = row_items;
            if self.surface == "both" {
                let majority_tv = row_items.iter().filter(|r| r.media_type == "tv").count() * 2 >= row_items.len();
                let other = if majority_tv { "movie" } else { "tv" };
                let in_row: HashSet<u32> = row_items.iter().map(|r| r.id).collect();
                // rank the other type by closeness to the cluster centre without copying vectors, then copy only the best 120
                let mut scored: Vec<(f32, usize)> = items
                    .iter()
                    .enumerate()
                    .filter(|(_, (it, _))| it.media_type == other && !in_row.contains(&it.id) && !state.used_ids.contains(&it.id))
                    .map(|(i, (_, v))| (cosine_sim(v, &centers[c]), i))
                    .collect();
                scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(120);
                let cands: Vec<(u32, Vec<f32>)> = scored.into_iter().map(|(_, i)| (items[i].0.id, items[i].1.clone())).collect();
                self.blend_other_type(&mut row_items, &centers[c], other, cands);
            }
            state.used_ids.extend(row_items.iter().map(|r| r.id));
            state.record_thematic(self.genre.unwrap_or("").to_string(), centers[c].clone());
            rows.push(Row { title: "__PENDING__".to_string(), archetype: "thematic".to_string(), kind: String::new(), items: row_items, meta: None });
            placed.push(c);
        }

        let _genre = self.genre.unwrap_or("").to_string();
        for (ri, &c) in placed.iter().enumerate() {
            let row_idx = first_thematic + ri;
            if row_idx >= rows.len() {
                break;
            }
            let era = dominant_era(&rows[row_idx].items);
            // name the row after what is in it, not after the page genre
            let dom = majority_genre(rows[row_idx].items.iter().map(|item| item.genres.as_slice()));
            let second = second_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
                &dom,
            );
            let row_catalog: Vec<&CatalogItem> = rows[row_idx].items.iter().filter_map(|i| self.catalog.get(&i.id)).collect();
            // concrete names from the catalog's own keywords come first; then the theme most of the row's titles share (from their profiles, when at
            // least half of them have one and half of those carry it); then the older generic names
            let mut candidates = crate::row_label::candidates(&row_catalog);
            let row_profiles: Vec<&crate::profiles::TitleProfile> = rows[row_idx].items.iter().filter_map(|i| self.profiles.get(i.id)).collect();
            if row_profiles.len() * 2 >= rows[row_idx].items.len() {
                if let Some((theme, _)) = crate::profiles::row_theme(&row_profiles, 0.5).or_else(|| self.profiles.row_theme_lift(&row_profiles, 0.3, 8.0)) {
                    candidates.extend(crate::row_label::theme_title(&theme, &row_catalog));
                }
                if let Some(tone) = self.profiles.row_tone(&row_profiles, 0.35, 2.5) {
                    candidates.extend(crate::row_label::tone_title(tone, &row_catalog));
                }
            }
            candidates.extend(derive_title(
                &dom, second.as_deref(), era.as_deref(),
                rows[row_idx].items.first().map(|i| i.title.as_str()),
                None,
            ));
            // the name of the group the row is built from, or of the group most of its titles belong to, comes first when it keeps its promise
            let group_name = cluster_group.get(c).and_then(|key| self.group_names.get(key).cloned()).or_else(|| self.majority_group_name(&row_catalog));
            if let Some(n) = group_name {
                if crate::row_label::name_fits(&n, &row_catalog) && !entry_names_other_kind(&n, self.surface) { candidates.insert(0, n); }
            }
            let candidates = crate::row_label::mark_animated(candidates, &row_catalog);
            let title = state.claim_title(&candidates);
            rows[row_idx].title = clamp_words(&title, 6);
        }

        rows
    }

    /// The name of the group (finest level first) that at least half of the titles belong to, if it has one.
    fn majority_group_name(&self, items: &[&CatalogItem]) -> Option<String> {
        if items.is_empty() { return None; }
        for level in 0..crate::groups::N_LEVELS {
            let mut count: HashMap<i32, usize> = HashMap::new();
            for it in items { if let Some(g) = self.groups.get(&it.id) { let id = g.at(level); if id >= 0 { *count.entry(id).or_insert(0) += 1; } } }
            if let Some((id, n)) = count.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))) {
                if n * 2 >= items.len() {
                    if let Some(name) = self.group_names.get(&(level, id)) { return Some(name.clone()); }
                }
            }
        }
        None
    }

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

    /// Rows on a surface that shows both types (home) should mix movies and series when the theme has both.
    /// Clusters tend to be one type, so a row that is mostly one type gets about 40 percent of its nearest
    /// neighbours of the other type, picked by relevance to the row centroid (with a floor relative to the
    /// row's own items) and diversified with MMR. A row with fewer than MIN_BLEND on theme items of the other
    /// type stays as it is: not every row has to mix.
    fn blend_other_type(&self, row: &mut Vec<RowItem>, centroid: &[f32], other: &str, cands: Vec<(u32, Vec<f32>)>) {
        const TARGET_SHARE: f32 = 0.4;
        const MIN_BLEND: usize = 3;
        const FLOOR_DROP: f32 = 0.12;
        if row.is_empty() || cands.is_empty() {
            return;
        }
        let have_other = row.iter().filter(|r| r.media_type == other).count();
        let target = ((ROW_SIZE as f32) * TARGET_SHARE).round() as usize;
        if have_other >= target {
            return;
        }
        // relevance of the items the row already has of its own type
        let mut own: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .filter(|(_, r)| r.media_type != other)
            .filter_map(|(i, r)| {
                let idx = self.vectors.index_of(r.id)?;
                Some((i, cosine_sim(centroid, &self.vectors.get_f32(idx))))
            })
            .collect();
        if own.is_empty() {
            return;
        }
        let mut vals: Vec<f32> = own.iter().map(|x| x.1).collect();
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let floor = vals[vals.len() / 2] - FLOOR_DROP;
        let ok: Vec<(u32, Vec<f32>)> = cands.into_iter().filter(|(_, v)| cosine_sim(centroid, v) >= floor).collect();
        if ok.is_empty() {
            return;
        }
        let picks = greedy_mmr(centroid, &ok, target - have_other, MMR_LAMBDA);
        if have_other + picks.len() < MIN_BLEND {
            return;
        }

        // compose: own items by relevance, then the other type, spread evenly through the row
        own.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let other_existing: Vec<RowItem> = row.iter().filter(|r| r.media_type == other).cloned().collect();
        let other_new: Vec<RowItem> = picks.iter().filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog)).collect();
        let n_other = other_existing.len() + other_new.len();
        let new_len = ROW_SIZE.min(row.len() + other_new.len());
        let keep_own = new_len.saturating_sub(n_other).min(own.len());
        let own_items: Vec<RowItem> = own.iter().take(keep_own).map(|&(i, _)| row[i].clone()).collect();
        let other_items: Vec<RowItem> = other_existing.into_iter().chain(other_new).collect();
        let mut keyed: Vec<(f32, RowItem)> = Vec::with_capacity(own_items.len() + other_items.len());
        for (rank, it) in own_items.iter().enumerate() {
            keyed.push(((rank as f32 + 0.5) / own_items.len() as f32, it.clone()));
        }
        for (rank, it) in other_items.iter().enumerate() {
            keyed.push(((rank as f32 + 0.25) / other_items.len() as f32, it.clone()));
        }
        keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        *row = keyed.into_iter().map(|(_, it)| it).collect();
    }

    /// Other type neighbours of a centroid from the HNSW index, for the seed path.
    async fn other_type_neighbours(&self, centroid: &[f32], other: &str, exclude: &HashSet<u32>) -> Vec<(u32, Vec<f32>)> {
        let found = self.retrieve_candidates(centroid, ANN_ENTRY_K).await;
        found
            .into_iter()
            .filter_map(|(id, _)| {
                let item = self.catalog.get(&id)?;
                if item.media_type != other || exclude.contains(&id) || !self.matches_filters(item) || !passes_member_quality(item) || !self.reliable(item) {
                    return None;
                }
                let idx = self.vectors.index_of(id)?;
                Some((id, self.vectors.get_f32(idx)))
            })
            .take(120)
            .collect()
    }

    // ── Placing the Top 10 rows ───────────────────────────────────────────────

    /// Puts the Top 10 rows among the rows of the page by value, the way rows are chosen for a page: every row has a value,
    /// the page is filled greedily from the highest, and a row loses value for being like a row already placed (two charts
    /// are alike, so the second waits). Nothing says "second" or "third": a profile with a clear taste values a chart less
    /// than rows made for it, so the chart sinks; a new profile values a chart more, so it rises. A session's noise moves a
    /// chart a little so pages differ between sessions.
    fn place_charts(&self, rows: Vec<Row>) -> Vec<Row> {
        /// Value of a chart for a profile without taste, and how much taste takes away.
        const CHART_BASE: f32 = 0.80;
        const CHART_TASTE_PENALTY: f32 = 0.30;
        const CHART_NOISE: f32 = 0.35;
        /// How much resemblance to a row already placed costs.
        const SIMILAR_COST: f32 = 0.35;

        let mut charts = self.build_chart_rows();
        let (old_charts, mut body): (Vec<Row>, Vec<Row>) = rows.into_iter().partition(|r| r.archetype == "trending");
        if charts.is_empty() {
            charts = old_charts;
        }
        if charts.is_empty() {
            return body;
        }
        // a title that is in a Top 10 is not repeated in the rows around it (the Top 10 must keep its ten)
        let chart_ids: HashSet<u32> = charts.iter().flat_map(|c| c.items.iter().map(|i| i.id)).collect();
        for row in body.iter_mut() {
            if row.items.iter().any(|i| chart_ids.contains(&i.id)) {
                row.items.retain(|i| !chart_ids.contains(&i.id));
            }
        }
        body.retain(|r| r.items.len() >= COHERENCE_MIN_ITEMS);

        // Fixed slots: the rows of the catalog layer (Top 10, New Releases, collections) and of the personal layer sit at the same positions in every session and on every profile
        // that has them; the topic rows keep their order around them. Positions are zero based and apply once the rows exist.
        let mut slotted: Vec<(usize, Row)> = Vec::new();
        let mut plain: Vec<Row> = Vec::new();
        let (mut byw_n, mut coll_n) = (0usize, 0usize);
        for r in body {
            match r.archetype.as_str() {
                "continue" => slotted.push((SLOT_CONTINUE, r)),
                "top_picks" => slotted.push((SLOT_TOP_PICKS, r)),
                "list" => slotted.push((SLOT_LIST, r)),
                "new" => slotted.push((SLOT_NEW, r)),
                "next_watch" => slotted.push((SLOT_NEXT, r)),
                "gems" => slotted.push((SLOT_GEMS, r)),
                "love_these" => slotted.push((SLOT_LOVE, r)),
                "favourites" => slotted.push((SLOT_FAVOURITES, r)),
                "byw" | "more_like" => { slotted.push((SLOT_ABOUT[byw_n.min(SLOT_ABOUT.len() - 1)], r)); byw_n += 1; }
                "collection" => { slotted.push((SLOT_COLLECTION[coll_n.min(SLOT_COLLECTION.len() - 1)], r)); coll_n += 1; }
                _ => plain.push(r),
            }
        }
        for (k, r) in charts.into_iter().enumerate() {
            slotted.push((SLOT_CHART[k.min(SLOT_CHART.len() - 1)], r));
        }
        // equal slots (more rows than slots of a kind) go one after the other
        slotted.sort_by_key(|(slot, _)| *slot);
        let mut next_free = 0usize;
        for (slot, row) in slotted {
            let at = slot.max(next_free).min(plain.len());
            plain.insert(at, row);
            next_free = at + 1;
        }
        plain
    }

    // ── Row builders ──────────────────────────────────────────────────────────

    fn build_trending_row(&self) -> Option<Row> {
        self.build_chart_row(None)
    }

    /// The Top 10 rows of this page: Movies and Series separately on the home page, one row on the other pages.
    fn build_chart_rows(&self) -> Vec<Row> {
        if self.genre.is_none() && self.surface == "both" && safety_nets() {
            ["movie", "tv"].into_iter().filter_map(|t| self.build_chart_row(Some(t))).collect()
        } else {
            self.build_chart_row(None).into_iter().collect()
        }
    }

    fn build_chart_row(&self, only: Option<&str>) -> Option<Row> {
        let nets = safety_nets();
        let size = if nets { TOP_N } else { ROW_SIZE };
        let kids_chart = self.genre.is_none() && self.language.is_none() && self.maturity.is_some_and(|m| m.is_kids()) && self.profiles.has_profiles();
        let keep = |item: &CatalogItem| {
            only.is_none_or(|t| item.media_type == t)
                && self.matches_filters(item)
                // a children's Top 10 holds titles made for children, as the rest of their home does
                && (!kids_chart || kid_interest(self.profiles, item))
                // a title that has not come out yet is not in a Top 10 of what people watch
                && !self.fresh.is_upcoming(&item.media_type, item.tmdb_id, &self.today_ymd)
        };
        let by_pop_desc = |a: &&CatalogItem, b: &&CatalogItem| {
            b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id))
        };
        let mut items: Vec<&CatalogItem> = if nets {
            // the catalog keeps the titles that can be on a chart, most popular first: the first `size` that pass the page's filters are the chart
            self.catalog.chart_keys.iter().filter_map(|k| self.catalog.get(k)).filter(|i| keep(i)).take(size).collect()
        } else {
            let mut all: Vec<&CatalogItem> = self.catalog.values().filter(|i| keep(i)).collect();
            if all.len() > size {
                all.select_nth_unstable_by(size - 1, by_pop_desc);
                all.truncate(size);
            }
            all
        };
        items.sort_unstable_by(by_pop_desc);
        let row_items: Vec<RowItem> = items
            .into_iter()
            .map(RowItem::from_catalog)
            .collect();

        if row_items.is_empty() {
            return None;
        }

        let title = match (only, self.surface, nets) {
            (Some("tv"), _, _) | (None, "tv", _) => "Top 10 Series Today",
            (Some("movie"), _, _) | (None, "movie", true) => "Top 10 Movies Today",
            _ => "Top 10 Today",
        };

        Some(Row {
            title:     title.to_string(),
            archetype: "trending".to_string(),
            kind:      String::new(),
            items:     row_items, meta: None,
        })
    }

    /// Build a "Because You Watched X" row using HNSW retrieval.
    /// Anchor must pass quality gate AND be thematically distinct from all placed rows.
    async fn build_byw_row(&self, state: &PageState) -> Option<Row> {
        // Find a high-quality anchor that is distinct from placed row centroids
        let candidates = self.retrieve_candidates(&self.taste, ANN_K).await;

        let anchor_id = candidates.iter().find_map(|(id, _)| {
            let item = self.catalog.get(id)?;
            if !self.matches_filters(item) || item.vote_average < 7.2 || item.popularity < 40.0 || item.overview.chars().count() < 80 {
                return None;
            }
            let idx = self.vectors.index_of(*id)?;
            let vec = self.vectors.get_f32(idx);
            // Must be distinct from all placed thematic row centroids
            let too_close = state.placed_centroids.iter().any(|c| cosine_sim(&vec, c) > BYW_MAX_CENTROID_SIM);
            if too_close { return None; }
            Some(*id)
        })?;

        let anchor_item = self.catalog.get(&anchor_id)?;
        let anchor_idx = self.vectors.index_of(anchor_id)?;
        let anchor_vec = self.vectors.get_f32(anchor_idx);

        // Retrieve similar items for the row
        let similar = self.retrieve_candidates(&anchor_vec, ANN_K / 2).await;
        let row_items: Vec<RowItem> = similar
            .iter()
            .filter(|(id, _)| *id != anchor_id)
            .filter_map(|(id, _)| self.catalog.get(id).filter(|i| self.matches_filters(i)))
            .take(ROW_SIZE)
            .map(RowItem::from_catalog)
            .collect();

        if row_items.len() < COHERENCE_MIN_ITEMS {
            return None;
        }

        Some(Row {
            title:     byw_title(&anchor_item.title),
            archetype: "byw".to_string(),
            kind:      String::new(),
            items:     row_items, meta: None,
        })
    }
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
