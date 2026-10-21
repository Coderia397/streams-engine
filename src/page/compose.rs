//! The page: which kinds of rows are built, in which order, and where they are placed among each other.
use super::*;

impl<'a> PageContext<'a> {
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

    /// The rows of the page: the topic rows, with up to two seasonal rows (from the row vocabulary, only inside their calendar window) placed near the top
    /// of the home, films and series pages.
    pub(crate) async fn build_page_rows(&self) -> Vec<Row> {
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

    /// Inserts extra rows at positions 3, 6, 9 ... (or at the end when the page is shorter), keeping the first rows of the page.
    pub(crate) fn interleave(&self, mut rows: Vec<Row>, extra: Vec<Row>, first: usize) -> Vec<Row> {
        for (n, r) in extra.into_iter().enumerate() {
            let at = (first + 3 * n).min(rows.len());
            rows.insert(at, r);
        }
        rows
    }
}
