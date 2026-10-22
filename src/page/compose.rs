//! The page: which kinds of rows are built, in which order, and where they are placed among each other.
//!
//! A page is built in layers. The base layer (`build_page_rows_inner`, see clusters.rs and genre.rs) makes the topic rows of the page kind; each layer after it adds a kind
//! of row to the rows so far and decides for itself whether it applies to this page (a children's page, a language page and a genre page each skip some). To add a
//! source of rows, write a layer and put one line in `build_page_rows`; to remove one, delete its line.
use super::*;

/// A page made from requests needs this many request rows, else it is built the older way.
const REQUEST_PAGE_MIN_ROWS: usize = 8;
/// A page of composed rows needs this many; below it the page is built the older way.
const COMPOSED_PAGE_MIN_ROWS: usize = 1;   // Netflix's Dinosaurs page has 2 rows: a thin page shows what it has

/// The ids of the titles already in a row of the page.
fn taken_ids(rows: &[Row]) -> HashSet<u32> {
    rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect()
}

/// The ids in the first screen of every row of the page (composed pages keep only these unique across rows).
fn first_screen_ids(rows: &[Row]) -> HashSet<u32> {
    rows.iter().flat_map(|r| r.items.iter().take(FIRST_SCREEN).map(|i| i.id)).collect()
}

/// The titles of the rows already on the page, lower case.
fn lower_titles(rows: &[Row]) -> HashSet<String> {
    rows.iter().map(|r| r.title.to_lowercase()).collect()
}

/// Inserts extra rows at positions `first`, `first + 3`, `first + 6` ... (or at the end when the page is shorter), keeping the first rows of the page.
fn interleave(rows: &mut Vec<Row>, extra: Vec<Row>, first: usize) {
    for (n, r) in extra.into_iter().enumerate() {
        let at = (first + 3 * n).min(rows.len());
        rows.insert(at, r);
    }
}

impl<'a> PageContext<'a> {
    /// A children's profile (age limited to children).
    pub(crate) fn is_kids(&self) -> bool {
        self.maturity.is_some_and(|m| m.is_kids())
    }

    /// A page of no genre, no language and no animation split: Home, Films or Series.
    pub(crate) fn is_plain_page(&self) -> bool {
        self.genre.is_none() && self.language.is_none() && self.animation.is_none()
    }

    /// The page for this request: rows from the builder for the page kind, then the Top 10 rows are placed by value among
    /// them (see `place_charts`).
    pub async fn build_page(&self) -> Vec<Row> {
        let rows = self.build_page_rows().await;
        let mut rows = self.place_charts(rows);
        // another version of the same title (a remake, a re-release) is not a second suggestion in one row
        for row in &mut rows {
            let mut seen: HashSet<(String, String)> = HashSet::default();
            row.items.retain(|i| seen.insert((i.media_type.clone(), crate::people::normalise(&i.title))));
        }
        rows
    }

    /// The rows of the page, layer by layer.
    pub(crate) async fn build_page_rows(&self) -> Vec<Row> {
        // with request rows on, the topic rows of the page come from requests (page/requests.rs); a genre page that gets too few falls back to its own builder
        let by_request = self.request_page();
        let mut rows = if by_request { self.build_request_page() } else { Vec::new() };
        // a page of composed rows is what its supply gives (Netflix's Cult page has 3 rows); the older request page needs a full first screen
        let floor = if self.composed_rows() { COMPOSED_PAGE_MIN_ROWS } else { REQUEST_PAGE_MIN_ROWS };
        let by_request = by_request && rows.iter().filter(|r| r.archetype != "trending").count() >= floor;
        // a page of composed rows never falls back to the older builders: what its supply gives is the page
        if !by_request && !(self.composed_rows() && self.request_page()) {
            rows = self.build_page_rows_inner().await;
        }
        self.layer_season(&mut rows, by_request).await;
        self.layer_kids_banners(&mut rows).await;
        if by_request { self.layer_night(&mut rows) } else { self.layer_feel(&mut rows) }
        self.layer_personal(&mut rows).await;
        self.layer_hygiene(&mut rows);
        rows
    }

    /// Collection banners (an event, a mood) and up to two seasonal rows (from the row vocabulary, only inside their calendar window) near the top of the home, films and series pages.
    async fn layer_season(&self, rows: &mut Vec<Row>, by_request: bool) {
        if !(self.is_plain_page() && !self.is_kids() && !self.taxonomy.entries.is_empty()) {
            return;
        }
        // the collection rows (an event's banner, a mood) are built first so the seasonal rows take what is left
        let collections = self.build_collection_rows(&taken_ids(rows), 4);
        rows.extend(collections);
        // a page made from requests has its seasonal rows already (calendar requests)
        if by_request {
            return;
        }
        let used = taken_ids(rows);
        let titles = lower_titles(rows);
        // sub groups of the season's pool when they exist, the flat keyword rows otherwise
        let seasonal = {
            let sub = self.build_seasonal_sub_rows(&used);
            if sub.is_empty() { self.build_seasonal_rows(&used).await } else { sub }
        };
        for (n, row) in seasonal.into_iter().filter(|r| !titles.contains(&r.title.to_lowercase())).enumerate() {
            let at = (if n == 0 { 2 } else { 5 + n }).min(rows.len());
            rows.insert(at, row);
        }
    }

    /// A children's page gets its banner collections (named ones and the ones in season), placed by slot like the adult banners.
    async fn layer_kids_banners(&self, rows: &mut Vec<Row>) {
        if !(self.is_kids() && self.kids_entries && self.is_plain_page()) {
            return;
        }
        let banners = self.build_kids_collection_rows(&taken_ids(rows), 3).await;
        rows.extend(banners);
    }

    /// Rows about how the titles feel and what they are about: late night moods lead a night page, vibe rows, attribute rows, topic rows and one mood row are mixed in.
    /// Every page that is not a children's or a language page gets them.
    fn layer_feel(&self, rows: &mut Vec<Row>) {
        if self.is_kids() || self.language.is_some() || self.animation.is_some() {
            return;
        }
        let used = taken_ids(rows);
        let mut night_rows: Vec<Row> = Vec::new();
        if self.late_night && self.genre.is_none() {
            night_rows.extend(self.build_mood_rows(&used, 2, true));
        }
        let mut taken = used;
        night_rows.iter().for_each(|r| r.items.iter().for_each(|i| { taken.insert(i.id); }));
        // topic rows that carry only a genre name ("Comedies", "Documentary & Drama") are replaced by rows named for how their titles feel
        let before = rows.len();
        if self.genre.is_none() {
            rows.retain(|r| !(r.kind.is_empty() && r.archetype == "thematic" && is_plain_genre_title(&r.title)));
        }
        let removed = before - rows.len();
        let titles = lower_titles(rows);
        let vibes: Vec<Row> = self.build_vibe_rows(&taken, 3 + removed.min(8)).into_iter().filter(|r| !titles.contains(&r.title.to_lowercase())).collect();
        let night = night_rows.len();
        let mut taken_all = taken;
        vibes.iter().for_each(|r| r.items.iter().for_each(|i| { taken_all.insert(i.id); }));
        let attrs: Vec<Row> = self.build_attribute_rows(&taken_all, 2).into_iter().filter(|r| !titles.contains(&r.title.to_lowercase())).collect();
        interleave(rows, night_rows, 1);
        interleave(rows, vibes, 3 + night);
        interleave(rows, attrs, 5 + night);
        // topic rows (what the titles are about), a few for each page
        let topic_rows = self.build_topic_rows(&taken_ids(rows), if self.genre.is_none() { 4 } else { 3 }, &lower_titles(rows));
        interleave(rows, topic_rows, 4 + night);
        // one mood row among the topic rows on a page without a genre
        if self.genre.is_none() && !self.late_night {
            let on_page = lower_titles(rows);
            let moods: Vec<Row> = self.build_mood_rows(&taken_ids(rows), 3, false).into_iter().filter(|r| !on_page.contains(&r.title.to_lowercase())).take(1).collect();
            interleave(rows, moods, 8);
        }
    }

    /// A night page made from requests leads with two late night mood rows.
    fn layer_night(&self, rows: &mut Vec<Row>) {
        if !(self.late_night && self.genre.is_none()) {
            return;
        }
        let moods = self.build_mood_rows(&taken_ids(rows), 2, true);
        interleave(rows, moods, 1);
    }

    /// Personal rows, New Releases and collections: added here, placed by slot in `place_charts`.
    /// On a page of composed rows every page gets them, children's pages included, and every page gets New Releases and Everyone's Watching filtered to its scope
    /// (Netflix carries New on Netflix on 79 of 92 pages and Everyone's Watching on 31, note 97).
    async fn layer_personal(&self, rows: &mut Vec<Row>) {
        let composed = self.composed_rows();
        if (self.is_kids() && !composed) || self.language.is_some() || self.animation.is_some() {
            return;
        }
        // composed rows share titles past their first screens, so personal rows avoid only what the first screens show
        let taken_now = if composed { first_screen_ids(rows) } else { taken_ids(rows) };
        let mut extra = self.build_personal_rows(&taken_now).await;
        if self.genre.is_none() || composed {
            let mut taken_all = taken_now;
            extra.iter().for_each(|r| r.items.iter().for_each(|i| { taken_all.insert(i.id); }));
            if let Some(r) = self.build_new_row(&taken_all) {
                extra.push(r);
            }
            // Everyone's Watching: the plain pages and every children's page (Netflix carries it on no adult genre page and on 78% of children's genre pages, note 97)
            if composed && (self.genre.is_none() || self.is_kids()) {
                if let Some(r) = self.build_popular_row(&taken_all) {
                    extra.push(r);
                }
            }
        }
        rows.extend(extra);
    }

    /// Rows that do not belong on a page: a row named for adult content, and on a page without a genre a row of fewer than 8 titles (it looks unfinished next to full rows).
    fn layer_hygiene(&self, rows: &mut Vec<Row>) {
        rows.retain(|r| !is_suggestive_title(&r.title));
        if self.genre.is_none() {
            rows.retain(|r| r.meta.is_some() || r.items.len() >= 8);
        }
    }
}
