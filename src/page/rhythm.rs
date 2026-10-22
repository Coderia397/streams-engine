//! The page rhythm learned from the Netflix captures (`page_rhythm_v1.json` next to the database, written by ai-engine/pipeline/netflix_rhythm.py over every capture
//! round and profile): how many rows each captured page has, where each personal and chart row sits (position share, 0 = first row, 1 = last), where each family of
//! topic row sits, and how often neighbouring rows repeat a genre word. The page plan takes the row count from it, `place_charts` the positions of the personal and
//! chart rows, and `build_request_page` the order of the topic rows. Without the file the engine keeps its earlier constants (tuning.rs SLOT_*).
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Quart {
    pub median: f32,
    pub p25: f32,
    pub p75: f32,
    #[serde(default)]
    pub n: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PageMatch {
    pub kids: bool,
    pub surface: String,
    pub genre: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PageRows {
    pub rows_median: f32,
    #[serde(default)]
    pub instances: usize,
    #[serde(default)]
    pub archetypes: Vec<String>,
    #[serde(rename = "match")]
    pub page_match: PageMatch,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Archetype {
    pub present: f32,
    pub position: Quart,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Family {
    pub position: Option<Quart>,
    #[serde(default)]
    pub count: Option<Quart>,
    #[serde(default)]
    pub positions: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Transitions {
    #[serde(default)]
    pub same_genre_adjacent: f32,
    #[serde(default)]
    pub same_genre_expected_random: Option<f32>,
    #[serde(default)]
    pub same_family_adjacent: f32,
}

/// What two visits of the same page say (the captures of one profile and page in two rounds): the move of a returning row (position share) is the width of the
/// engine's session noise around a row's learned position
#[derive(Debug, Clone, Deserialize)]
pub struct Visits {
    #[serde(default)]
    pub pairs: usize,
    pub move_share: Option<Quart>,
    #[serde(default)]
    pub names_kept: Option<f32>,
    #[serde(default)]
    pub first_screen_kept: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Kind {
    pub instances: usize,
    pub rows: Quart,
    #[serde(default)]
    pub visits: Option<Visits>,
    #[serde(default)]
    pub pages: HashMap<String, PageRows>,
    #[serde(default)]
    pub families: HashMap<String, Family>,
    #[serde(default)]
    pub archetypes: HashMap<String, Archetype>,
    #[serde(default)]
    pub transitions: Transitions,
}

#[derive(Debug, Deserialize)]
pub struct Rhythm {
    pub kinds: HashMap<String, Kind>,
}

static RHYTHM: OnceLock<Option<Rhythm>> = OnceLock::new();

/// Reads `page_rhythm_v1.json` next to `db_path` once; a missing file leaves the engine on its constants.
pub fn load(db_path: &str) {
    let r = std::path::Path::new(db_path)
        .parent()
        .and_then(|d| std::fs::read_to_string(d.join("page_rhythm_v1.json")).ok())
        .and_then(|t| serde_json::from_str::<Rhythm>(&t).map_err(|e| tracing::warn!("page_rhythm_v1.json is not readable: {e}")).ok());
    match &r {
        Some(r) => tracing::info!("Page rhythm: {} page kinds from the captures", r.kinds.len()),
        None => tracing::info!("Page rhythm: no file, the engine keeps its slot constants"),
    }
    let _ = RHYTHM.set(r);
}

pub fn get() -> Option<&'static Rhythm> {
    RHYTHM.get().and_then(|r| r.as_ref())
}

/// The captured page kind a page of ours is: the profile kind, the surface and whether it has a genre.
pub fn kind_for(kids: bool, surface: &str, has_genre: bool) -> &'static str {
    match (kids, surface, has_genre) {
        (true, _, true) => "kids_genre",
        (true, "tv", false) => "kids_series",
        (true, "movie", false) => "kids_films",
        (true, _, false) => "kids_home",
        (false, "tv", true) => "genre_series",
        (false, "movie", true) => "genre_films",
        (false, _, true) => "genre_films",
        (false, "tv", false) => "series",
        (false, "movie", false) => "films",
        (false, _, false) => "home",
    }
}

pub fn kind(kids: bool, surface: &str, has_genre: bool) -> Option<&'static Kind> {
    get()?.kinds.get(kind_for(kids, surface, has_genre))
}

/// Rows Netflix gives this page: the captured page itself (the same profile kind, surface and genre name, case insensitive), else the median of its page kind.
pub fn page_rows(kids: bool, surface: &str, genre: Option<&str>) -> Option<f32> {
    let r = get()?;
    if let Some(g) = genre {
        let hit = r.kinds.values().flat_map(|k| k.pages.values()).find(|p| {
            p.page_match.kids == kids && p.page_match.genre.as_deref().is_some_and(|x| x.eq_ignore_ascii_case(g)) && (p.page_match.surface == surface || p.page_match.surface == "both" || surface == "both")
        });
        if let Some(p) = hit {
            return Some(p.rows_median);
        }
    }
    kind(kids, surface, genre.is_some()).map(|k| k.rows.median)
}

/// Where a personal or chart row of this archetype sits on pages of this kind (position share), when Netflix carries it there.
pub fn archetype_position(kids: bool, surface: &str, has_genre: bool, archetype: &str) -> Option<Quart> {
    kind(kids, surface, has_genre)?.archetypes.get(archetype).map(|a| a.position)
}

/// Where rows of this family (topic, mood, place, award, people) sit on pages of this kind.
pub fn family_position(kids: bool, surface: &str, has_genre: bool, family: &str) -> Option<Quart> {
    kind(kids, surface, has_genre)?.families.get(family).and_then(|f| f.position)
}

/// The personal and chart rows the captures show on this very page (by profile kind, surface and genre name), when the page was captured: a thin page (Cult, Dinosaurs)
/// carries only those.
pub fn page_archetypes(kids: bool, surface: &str, genre: Option<&str>) -> Option<(f32, Vec<String>)> {
    let r = get()?;
    let g = genre?;
    r.kinds.values().flat_map(|k| k.pages.values()).find(|p| {
        p.page_match.kids == kids && p.page_match.genre.as_deref().is_some_and(|x| x.eq_ignore_ascii_case(g)) && (p.page_match.surface == surface || p.page_match.surface == "both" || surface == "both")
    }).map(|p| (p.rows_median, p.archetypes.clone()))
}

/// How many personal and chart rows Netflix carries on pages of this kind (medians), so a page's learned row count leaves room for them beside the topic rows.
pub fn personal_chart_rows(kids: bool, surface: &str, has_genre: bool) -> Option<f32> {
    let k = kind(kids, surface, has_genre)?;
    let count = |fam: &str| k.families.get(fam).and_then(|f| f.count.as_ref()).map_or(0.0, |q| q.median);
    Some(count("personal") + count("chart"))
}

/// How many award or acclaim rows Netflix carries on pages of this kind (median: one on a genre page, 4 to 8 on Home, Films and Series).
pub fn award_rows(kids: bool, surface: &str, has_genre: bool) -> Option<usize> {
    kind(kids, surface, has_genre)?.families.get("award").and_then(|f| f.count.as_ref()).map(|q| q.median.round() as usize)
}

/// Whether Netflix keeps rows of one genre apart on pages of this kind: neighbouring rows share a genre word less often than a random order would give.
pub fn spreads_genres(kids: bool, surface: &str, has_genre: bool) -> Option<bool> {
    let t = &kind(kids, surface, has_genre)?.transitions;
    Some(t.same_genre_adjacent < t.same_genre_expected_random.unwrap_or(t.same_genre_adjacent))
}

/// A position share placed inside its quartile window by a session's noise (0 to 1): the median for 0.5, the quartiles at the ends.
pub fn jittered(q: Quart, noise: f32) -> f32 {
    let x = if noise < 0.5 { q.p25 + (q.median - q.p25) * (noise / 0.5) } else { q.median + (q.p75 - q.median) * ((noise - 0.5) / 0.5) };
    x.clamp(0.0, 1.0)
}

/// How far a row moves between two visits of a page of this kind (position share, the captures' median), when the captures hold two visits.
pub fn visit_move(kids: bool, surface: &str, has_genre: bool) -> Option<f32> {
    kind(kids, surface, has_genre)?.visits.as_ref()?.move_share.map(|q| q.median)
}

/// A position share moved by a session's noise the way a row moves between two visits: the median plus or minus the measured move (the quartile window when the
/// captures hold no second visit of this kind).
pub fn placed(kids: bool, surface: &str, has_genre: bool, q: Quart, noise: f32) -> f32 {
    match visit_move(kids, surface, has_genre) {
        Some(m) if m > 0.0 => (q.median + (noise - 0.5) * 2.0 * m).clamp(0.0, 1.0),
        _ => jittered(q, noise),
    }
}
