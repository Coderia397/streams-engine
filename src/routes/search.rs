use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;
use rusqlite::Connection;
use crate::app::AppState;
use crate::catalog::CatalogItem;
use crate::identity::{maturity_of, MaybeCaller, Maturity};
use crate::people::{normalise, within_edits};

/// Normalised titles with popularity, rebuilt when the catalog is replaced (every five minutes), so a title search is a scan of memory rather than
/// a ranking of every full text match (which took seconds for words such as "love").
struct TitleIndex {
    catalog_ptr: usize,
    rows: Vec<(u32, f32, String)>,
}

fn title_index(catalog: &Arc<crate::catalog::CatalogMap>) -> Arc<TitleIndex> {
    use std::sync::{Mutex, OnceLock};
    static INDEX: OnceLock<Mutex<Option<Arc<TitleIndex>>>> = OnceLock::new();
    let slot = INDEX.get_or_init(|| Mutex::new(None));
    let ptr = Arc::as_ptr(catalog) as usize;
    if let Some(i) = slot.lock().unwrap().as_ref().filter(|i| i.catalog_ptr == ptr) {
        return i.clone();
    }
    let rows = catalog.values().map(|i| (i.id, i.popularity, format!(" {} ", normalise(&i.title)))).collect();
    let built = Arc::new(TitleIndex { catalog_ptr: ptr, rows });
    *slot.lock().unwrap() = Some(built.clone());
    built
}

/// Keys of titles whose words all match the query (the last word may be unfinished), exact title first, then titles starting with the query,
/// then the most popular.
fn title_matches(index: &TitleIndex, qn: &str, limit: usize) -> Vec<u32> {
    let words: Vec<&str> = qn.split(' ').filter(|w| !w.is_empty()).collect();
    let Some((last, head)) = words.split_last() else { return Vec::new() };
    let exact = format!(" {qn} ");
    let starts = format!(" {qn}");
    let head_pats: Vec<String> = head.iter().map(|w| format!(" {w} ")).collect();
    let last_pat = format!(" {last}");
    let mut hits: Vec<(u8, f32, u32)> = index
        .rows
        .iter()
        .filter(|(_, _, t)| t.contains(&last_pat) && head_pats.iter().all(|p| t.contains(p.as_str())))
        .map(|(k, pop, t)| (if *t == exact { 0 } else if t.starts_with(&starts) { 1 } else { 2 }, *pop, *k))
        .collect();
    hits.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)).then(a.2.cmp(&b.2)));
    hits.into_iter().take(limit).map(|h| h.2).collect()
}

#[derive(Deserialize)]
pub struct SearchQuery {
    /// Age rating limit of a local profile (U, PG, 12, 15 or 18). Used only when no verified profile is sent; a verified profile's own setting wins.
    pub rating: Option<String>,
    pub q: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}
fn default_limit() -> usize { 20 }

fn result_json(item: &CatalogItem) -> Value {
    json!({
        "id": item.tmdb_id, "title": item.title, "media_type": item.media_type, "poster_path": item.poster_path, "backdrop_path": item.backdrop_path,
        "vote_average": item.vote_average, "genres": item.genres, "release_year": item.release_year, "overview": item.overview,
        "rating": crate::catalog::cert_label(item.cert), "rating_gb": crate::catalog::gb_label(item.cert_gb), "rating_us": crate::catalog::us_label(item.cert_us),
    })
}

/// GET /api/search?q=<query>&limit=<n>
/// Titles first (whole words, the last word may be unfinished), then the titles of a person the query names, then titles within a letter
/// or two of the query (typos). A query that is exactly a person's name lists that person's titles first. Every result carries its real
/// certificates, and a limited profile only gets titles it may see.
pub async fn search(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| params.rating.as_deref().and_then(Maturity::from_rating));
    let q = params.q.trim().to_string();
    let wanted = params.limit.clamp(1, 100);
    if q.is_empty() {
        return Json(json!({"results": [], "query": q}));
    }
    // a limited profile loses some matches to the filter, so look further and trim after
    let limit = if maturity.is_some_and(|m| m != Maturity::Adult) { (wanted * 4).min(200) } else { wanted };
    let catalog = state.catalog.load();

    // 1. titles whose words match, from memory; matches elsewhere in the text (overview, genres, director) only when titles are few
    let qn = normalise(&q);
    let index = title_index(&catalog);
    let title_keys: Vec<u32> = title_matches(&index, &qn, limit);
    let other_keys: Vec<u32> = if title_keys.len() < 5 {
        let db_path = state.config.db_path.clone();
        let q_clone = q.clone();
        match tokio::task::spawn_blocking(move || fts_elsewhere(&db_path, &q_clone, limit)).await {
            Ok(Ok(keys)) => keys,
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    // 2. titles of a person the query names
    let (exact_person, person_keys) = state.people.titles_for(&q, 24);

    // 3. near misses, only when the titles found are few
    let mut fuzzy_keys: Vec<u32> = Vec::new();
    if title_keys.len() < 4 && person_keys.is_empty() && qn.chars().count() >= 4 {
        let max = if qn.chars().count() <= 7 { 1 } else { 2 };
        let mut hits: Vec<&CatalogItem> = catalog
            .values()
            .filter(|i| i.popularity >= 1.0 && i.title.chars().count().abs_diff(qn.chars().count()) <= max)
            .filter(|i| within_edits(&qn, &normalise(&i.title), max))
            .collect();
        hits.sort_unstable_by(|a, b| b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then(a.id.cmp(&b.id)));
        fuzzy_keys = hits.into_iter().take(12).map(|i| i.id).collect();
    }
    // with few title matches, near misses compete with them on popularity (a popular title one letter away beats an obscure exact word hit)
    let (title_keys, fuzzy_keys) = if fuzzy_keys.is_empty() {
        (title_keys, fuzzy_keys)
    } else {
        let mut all: Vec<u32> = title_keys.iter().chain(fuzzy_keys.iter()).copied().collect::<HashSet<u32>>().into_iter().collect();
        all.sort_by(|a, b| {
            let (pa, pb) = (catalog.get(a).map_or(0.0, |i| i.popularity), catalog.get(b).map_or(0.0, |i| i.popularity));
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
        });
        (all, Vec::new())
    };

    // a name typed in full lists that person's titles first, unless it is also the name of a title (one word such as "Prince")
    let named_title = title_keys.iter().filter_map(|k| catalog.get(k)).any(|i| normalise(&i.title) == qn);
    let person_first = exact_person && (qn.contains(' ') || !named_title);
    let ordered: Vec<u32> = if person_first {
        person_keys.iter().take(12).chain(title_keys.iter()).chain(person_keys.iter().skip(12)).chain(other_keys.iter()).copied().collect()
    } else {
        title_keys.iter().chain(person_keys.iter()).chain(fuzzy_keys.iter()).chain(other_keys.iter()).copied().collect()
    };
    let mut seen: HashSet<u32> = HashSet::new();
    let results: Vec<Value> = ordered
        .into_iter()
        .filter(|k| seen.insert(*k))
        .filter_map(|k| catalog.get(&k))
        .filter(|item| maturity.is_none_or(|m| crate::page::maturity_allows_with(m, &state.profiles, item)))
        .take(wanted)
        .map(result_json)
        .collect();

    Json(json!({"results": results, "query": q}))
}

/// Internal item keys of titles whose overview, genres or director hold the query as a phrase, best text match first.
fn fts_elsewhere(db_path: &str, q: &str, limit: usize) -> anyhow::Result<Vec<u32>> {
    let conn = Connection::open(db_path)?;
    let words = normalise(q);
    if words.is_empty() {
        return Ok(Vec::new());
    }
    // only letters and digits reach the query, so nothing in it can be read as full text syntax
    let phrase_query = format!("\"{words}\"");
    let mut stmt = conn.prepare(
        "SELECT c.id FROM catalog_fts f JOIN catalog c ON c.id = f.id WHERE catalog_fts MATCH ?1 ORDER BY rank, c.popularity DESC LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![phrase_query, limit as i64], |r| r.get::<_, i64>(0))?;
    Ok(rows.filter_map(|r| r.ok()).map(|k| k as u32).collect())
}
