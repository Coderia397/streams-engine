//! The rows a page is made of, and the state a page builder carries while it places them.
use super::*;

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
            used_titles:      HashSet::default(),
            recent_genres:    VecDeque::new(),
            placed_centroids: Vec::new(),
            used_ids:         HashSet::default(),
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
