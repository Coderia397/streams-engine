//! Tuning constants of the page builders: sizes, gates, slots. Thresholds that depend on the embedding space are calibrated, see the comments.
use super::*;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Rules that keep defects out of default pages: tile art required, no news, talk or soap in default rows,
/// no Christmas or Halloween titles outside their window. `ENGINE_SAFETY_NETS=off` disables them for baselines.
pub static SAFETY_NETS: AtomicBool = AtomicBool::new(true);
pub fn safety_nets() -> bool {
    SAFETY_NETS.load(Ordering::Relaxed)
}

/// Titles in the Top 10 row
pub(crate) const TOP_N: usize = 10;

/// At most this many titles in a row may share a significant title word (a franchise, a character name), so a row is
/// not 13 Doctor Who entries. And at most this many documentary or reality rows on a default page.
pub(crate) const FRANCHISE_CAP: usize = 2;
pub(crate) const NONSCRIPTED_ROW_CAP: u32 = 2;

/// Session seeded exploration: how far a session may reorder seeds, and the share of members a session leaves out of
/// the MMR pool, so two sessions see different rows and titles while one session stays stable.
pub(crate) const SEED_NOISE: f32 = 0.30;
pub(crate) const MEMBER_DROP: f32 = 0.22;

/// Coherence gate thresholds. Set by percentile matching (ai-engine/scripts/calibrate.py): each gate keeps the strictness it had
/// in the first Qwen3-Embedding-4B space, translated to the current space (title free text with tagline, people and cast,
/// seeds trained on the quality pool). Rerun the script after any embedding change; thresholds from another space do not transfer.
/// Measured 2026-10-04: old to new COHERENCE_OK 0.55 to 0.60 (item to medoid cosine), COHERENCE_WEAK 0.50 to 0.56 (mean pairwise
/// cosine of a row), MMD_MAX_INTER_SIM 0.76 to 0.74 and BYW_MAX_CENTROID_SIM 0.68 to 0.63 (centroid to centroid).
pub(crate) const COHERENCE_STRONG: f32 = 0.74;
pub(crate) const COHERENCE_OK:     f32 = 0.60;
pub(crate) const COHERENCE_WEAK:   f32 = 0.56;
pub(crate) const COHERENCE_MIN_ITEMS: usize = 6;

/// Max cosine similarity between any two row anchors (MMD gate)
pub(crate) const MMD_MAX_INTER_SIM: f32 = 0.74;

/// MMR λ: 0 = pure diversity, 1 = pure relevance. 0.72 = production default.
pub(crate) const MMR_LAMBDA: f32 = 0.72;

/// Genre bitmask window: no genre repeated in the last N rows
pub(crate) const GENRE_WINDOW: usize = 4;

/// Entry driven pages need a real vocabulary; below this the cluster path is used.
pub(crate) const MIN_TAXONOMY_FOR_ENTRY_ROWS: usize = 200;

/// Best members (by quality) kept per seed before MMR
pub(crate) const MEMBER_POOL: usize = 120;

/// Items per row (max)
pub(crate) const ROW_SIZE: usize = 15;
/// A genre page shows at least this many rows: when its own pool gives fewer, rows of neighbouring titles fill it up (`fill_thin_genre_page`)
pub const THIN_PAGE_FLOOR: usize = 15;
/// Fixed positions (zero based) of the catalog and personal rows on a page
pub(crate) const SLOT_CONTINUE: usize = 0;
pub(crate) const SLOT_TOP_PICKS: usize = 1;
pub(crate) const SLOT_CHART: [usize; 2] = [2, 7];
pub(crate) const SLOT_LIST: usize = 3;
pub(crate) const SLOT_COLLECTION: [usize; 4] = [4, 11, 19, 28];
pub(crate) const SLOT_NEW: usize = 5;
pub(crate) const SLOT_ABOUT: [usize; 3] = [6, 12, 14];
pub(crate) const SLOT_NEXT: usize = 9;
pub(crate) const SLOT_GEMS: usize = 10;
pub(crate) const SLOT_LOVE: usize = 8;
pub(crate) const SLOT_FAVOURITES: usize = 13;
/// A genre page built from groups: the (level, id) of each cluster and the item indices kept
pub(crate) type GroupMode = (Vec<(u8, i32)>, Vec<usize>);
/// The titles of one group, by group id
pub(crate) type GroupItems = (i32, Vec<usize>);
/// An item index with its score
pub(crate) type ScoredIdx = (usize, f32);
/// A seasonal sub group is shown only when at least this share of its titles carry the event keywords (the rest joined the pool by similarity and drift)
pub(crate) const SEASON_MIN_KW_SHARE: f32 = 0.4;
/// Mean centred cosine of a seasonal sub group's best 15 titles to the event vector below which the sub group is not shown
pub(crate) const SEASON_MIN_FIT: f32 = 0.18;
/// Events whose pool is defined by precise catalog keywords (identity, heritage and family days): their sub groups are ranked by fit but not dropped for a low fit
pub(crate) const KEYWORD_EVENTS: &[&str] = &["pride", "black_history", "womens_history", "awards", "mothers_day", "fathers_day", "cinco_de_mayo", "st_patricks", "july4", "ramadan_eid", "independence_india", "hanukkah", "diwali", "easter", "lunar_new_year"];

/// ANN top-k for HNSW retrieval
pub(crate) const ANN_K: usize = 200;

/// Deeper retrieval for entry rows: only about 5% of the catalog passes the quality gate
pub(crate) const ANN_ENTRY_K: usize = 500;
/// Depth of a row's search on a genre page.
pub(crate) const ANN_GENRE_K: usize = 2500;

/// Taxonomy cosine threshold to accept Hungarian assignment (else fallback title)
pub(crate) const TAXONOMY_THRESHOLD: f32 = 0.55;

/// BYW anchor: cosine to any existing row centroid must be below this
pub(crate) const BYW_MAX_CENTROID_SIM: f32 = 0.63;
