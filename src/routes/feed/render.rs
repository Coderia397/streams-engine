//! The rows of a page in the shape the client reads (FeedRow): titles deduplicated across rows, short rows dropped.
use crate::det::HashSet;
use crate::page::{Row, RowItem};
use serde_json::{json, Value};

/// A row with fewer titles than this (after the titles of earlier rows are taken out) is not shown.
const MIN_ROW_ITEMS: usize = 4;

/// Convert engine rows to the client FeedRow shape. Titles are deduplicated across rows (earlier rows win) and rows left with too few items are dropped. With
/// `first_screen_only` (composed rows, note 101) only the first FIRST_SCREEN tiles of each row are kept free of titles shown in the first screen of an earlier row,
/// as Netflix does (note 97); a title moves back in the row instead of leaving it. A row whose name no longer keeps its promise once its shared titles are gone
/// (a "Korean Thrillers" row left with too few Korean titles) is dropped too: the name was checked on the titles before they were shared out.
pub fn to_client_rows(rows: Vec<Row>, already_seen: &HashSet<u32>, profiles: &crate::profiles::ProfileStore, fresh: &crate::fresh::FreshStore, catalog: &crate::catalog::CatalogMap, first_screen_only: bool) -> Vec<Value> {
    use crate::page::FIRST_SCREEN;
    let mut seen: HashSet<u32> = HashSet::default();
    let mut out = Vec::new();
    let today = crate::fresh::today_days();

    for (i, row) in rows.into_iter().enumerate() {
        let chart = row.archetype == "trending";
        let kept: Vec<&RowItem> = if first_screen_only && !chart {
            // the first screen takes the first titles not yet in an earlier first screen; the rest of the row follows in its order, the moved titles after it
            let mut head: Vec<&RowItem> = Vec::new();
            let mut tail: Vec<&RowItem> = Vec::new();
            let mut moved: Vec<&RowItem> = Vec::new();
            for it in row.items.iter().filter(|it| !already_seen.contains(&it.id)) {
                if head.len() < FIRST_SCREEN {
                    if seen.contains(&it.id) { moved.push(it) } else { head.push(it) }
                } else {
                    tail.push(it);
                }
            }
            head.into_iter().chain(tail).chain(moved).collect()
        } else {
            row.items.iter().filter(|it| !already_seen.contains(&it.id) && !seen.contains(&it.id)).collect()
        };
        if kept.len() < MIN_ROW_ITEMS {
            continue;
        }
        let shown: Vec<&crate::catalog::CatalogItem> = kept.iter().filter_map(|it| catalog.get(&it.id)).collect();
        if !crate::row_label::name_fits(&row.title, &shown) {
            continue;
        }
        seen.extend(kept.iter().take(if first_screen_only { FIRST_SCREEN } else { usize::MAX }).map(|it| it.id));
        let items: Vec<Value> = kept.iter().enumerate().map(|(n, it)| item_json(it, chart.then_some(n + 1), profiles, fresh, today)).collect();
        out.push(row_json(&row, i, items));
    }
    out
}

/// One title as the client reads it; `rank` is its place in a Top 10 row.
fn item_json(it: &RowItem, rank: Option<usize>, profiles: &crate::profiles::ProfileStore, fresh: &crate::fresh::FreshStore, today: i64) -> Value {
    let release = fresh.get(&it.media_type, it.tmdb_id).map(|f| f.release_date.as_str());
    json!({
        "id":            it.tmdb_id,
        "media_type":    it.media_type,
        "title":         it.title,
        "release_year":  it.release_year.map(|y| y.to_string()),
        "genres":        it.genres.join(", "),
        "overview":      it.overview,
        "poster_path":   it.poster_path,
        "backdrop_path": it.backdrop_path,
        "vote_average":  it.vote_average,
        "tv_type":       it.tv_type,
        "language":      it.language,
        "rating":        crate::catalog::cert_label(it.cert),
        "rating_gb":     crate::catalog::gb_label(it.cert_gb),
        "rating_us":     crate::catalog::us_label(it.cert_us),
        "runtime":       it.runtime,
        "tags":          crate::contract::item_tags(profiles.get(it.id), &it.genres, 3),
        "badges":        crate::contract::item_badges(rank, release, today),
    })
}

/// One row: its id, kind and title, with the facts of a personal or collection row (translation key and arguments of the title, the reason, a collection's text and target) merged in.
fn row_json(row: &Row, position: usize, items: Vec<Value>) -> Value {
    let row_type = match row.archetype.as_str() { "trending" => "top_ten", "collection" => "collection", _ => "vibe" };
    let mut obj = json!({
        "id":        format!("{}-{}", row.archetype, position),
        "row_type":  row_type,
        "type":      if row.kind.is_empty() && row.archetype != "thematic" && row.archetype != "trending" { row.archetype.clone() } else { row.kind.clone() },
        "title":     row.title,
        "items":     items,
    });
    if let (Some(Value::Object(m)), Some(o)) = (&row.meta, obj.as_object_mut()) {
        for (k, v) in m {
            o.insert(k.clone(), v.clone());
        }
    }
    obj
}
