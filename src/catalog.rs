use anyhow::Result;
use arc_swap::ArcSwap;
use rusqlite::{Connection, OpenFlags};
use std::sync::atomic::{AtomicU32, Ordering};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// A single item in the streaming catalog.
#[derive(Debug, Clone)]
pub struct CatalogItem {
    /// Internal key, unique across movies and TV (see `item_key`). Use `tmdb_id` for anything shown to clients.
    pub id: u32,
    /// TMDB id. Movies and TV series share the same id space, so it is only unique together with `media_type`.
    pub tmdb_id: u32,
    pub media_type: String,
    pub title: String,
    pub release_year: Option<i32>,
    /// TMDB genre names, e.g. ["Action", "Science Fiction"]
    pub genres: Vec<String>,
    pub overview: String,
    pub tagline: Option<String>,
    pub director: Option<String>,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
    pub vote_average: f32,
    pub popularity: f32,
    pub is_anime: bool,
    /// Present only in a catalog built from the TMDB pull (catalog_v2); defaults otherwise.
    pub vote_count: u32,
    pub adult: bool,
    pub original_language: String,
    /// TMDB tv type: Scripted, Reality, Talk Show, News, Documentary, Miniseries, Video
    pub tv_type: Option<String>,
    /// ISO 3166 origin or production countries (catalog_v2 only)
    pub countries: Vec<String>,
    /// TMDB keywords, lower case (catalog_v2 only, pulled titles)
    pub keywords: Vec<String>,
    /// British age rating of the title as a rank: 0 unknown, 1 U, 2 PG, 3 12, 4 15, 5 18. Taken from the GB certificate, else the US one.
    pub cert: u8,
    /// The real certificates, kept so a client can show the system its user reads: GB rank 1 to 5 (U, PG, 12, 15, 18), and the US code (see `us_label`). 0 when absent.
    pub cert_gb: u8,
    pub cert_us: u8,
    /// Minutes: movie runtime or median episode runtime, 0 when unknown
    pub runtime: u32,
    /// Bit flags computed once at load (see FLAG_*), so page builders never scan titles or keywords per request
    pub flags: u8,
}

pub const FLAG_CHRISTMAS: u8 = 1;
pub const FLAG_HALLOWEEN: u8 = 2;
pub const FLAG_WRESTLING_EVENT: u8 = 4;
pub const FLAG_NONSCRIPTED: u8 = 8; // news, talk, soap
pub const FLAG_SUGGESTIVE: u8 = 16; // ecchi, harem, softcore, erotic: not for billboards, Top 10 or default rows
/// The flags above are the data defect rules; a page asks `flags & FLAGS_RULES` to know whether any applies
pub const FLAGS_RULES: u8 = 31;
pub const FLAG_OVERVIEW: u8 = 32; // overview of 60 characters or more
pub const FLAG_POSTER: u8 = 64; // has a poster
pub const FLAG_BACKDROP: u8 = 128; // has a backdrop

fn compute_flags(title: &str, keywords: &[String], tv_type: Option<&str>, overview: &str, poster: Option<&str>, backdrop: Option<&str>) -> u8 {
    let t = title.to_lowercase();
    let mut f = 0u8;
    if overview.chars().take(60).count() >= 60 {
        f |= FLAG_OVERVIEW;
    }
    if poster.is_some_and(|p| !p.is_empty()) {
        f |= FLAG_POSTER;
    }
    if backdrop.is_some_and(|b| !b.is_empty()) {
        f |= FLAG_BACKDROP;
    }
    if keywords.iter().any(|k| k.contains("christmas")) || t.contains("christmas") || t.contains("xmas") || t.contains("santa claus") {
        f |= FLAG_CHRISTMAS;
    }
    if keywords.iter().any(|k| k.contains("halloween")) || t.contains("halloween") {
        f |= FLAG_HALLOWEEN;
    }
    if keywords.iter().any(|k| k == "pro wrestling" || k == "professional wrestling")
        || ["wwe ", "wwf ", "wcw ", "wrestlemania", "summerslam", "royal rumble", "survivor series"].iter().any(|w| t.contains(w))
    {
        f |= FLAG_WRESTLING_EVENT;
    }
    if matches!(tv_type, Some("News") | Some("Talk Show") | Some("Soap")) {
        f |= FLAG_NONSCRIPTED;
    }
    const SUGGESTIVE: &[&str] = &[
        "ecchi", "harem", "reverse harem", "softcore", "eroticism", "erotic", "erotic movie", "erotic thriller", "sex", "sex comedy",
        "nudity", "sexy", "pornography", "stripper", "brothel", "hentai", "sexual humor",
    ];
    if keywords.iter().any(|k| SUGGESTIVE.contains(&k.as_str())) {
        f |= FLAG_SUGGESTIVE;
    }
    f
}

/// TMDB numbers movies and TV series separately, so the same id exists in both. The engine keys every
/// title by `item_key`: movies keep their TMDB id, TV series get the top bit set.
pub const TV_BIT: u32 = 1 << 31;

#[inline]
pub fn item_key(media_type: &str, tmdb_id: u32) -> u32 {
    if media_type == "tv" { tmdb_id | TV_BIT } else { tmdb_id }
}

/// Resolve a client supplied (id, media_type) to an internal key that exists in the catalog.
/// Without a media type the movie is tried first, then the series.
pub fn resolve_key(catalog: &CatalogMap, tmdb_id: u32, media_type: Option<&str>) -> Option<u32> {
    match media_type {
        Some(mt) => Some(item_key(mt, tmdb_id)).filter(|k| catalog.contains_key(k)),
        None => [tmdb_id, tmdb_id | TV_BIT].into_iter().find(|k| catalog.contains_key(k)),
    }
}

impl CatalogItem {
    /// Overview of 60 characters or more, poster and backdrop present: flags set once at load, so a scan of the catalog never counts characters or compares strings
    #[inline]
    pub fn has_overview(&self) -> bool {
        self.flags & FLAG_OVERVIEW != 0
    }

    #[inline]
    pub fn has_poster(&self) -> bool {
        self.flags & FLAG_POSTER != 0
    }

    #[inline]
    pub fn has_backdrop(&self) -> bool {
        self.flags & FLAG_BACKDROP != 0
    }

}

/// The titles of the catalog by key, plus the keys of the titles that can be on a Top 10 chart (`page::chart_ok`), most popular first (made once at load,
/// after the popularity overlay, so a chart reads its first ten matching titles instead of scanning the catalog).
pub struct Catalog {
    items: HashMap<u32, CatalogItem>,
    pub chart_keys: Vec<u32>,
    /// Per attribute row kind (`attributes::SPECS`) the quality titles that have the attribute, best quality first
    pub attr_keys: HashMap<&'static str, Vec<u32>>,
}

impl Catalog {
    pub fn new(items: HashMap<u32, CatalogItem>) -> Self {
        let mut chart: Vec<&CatalogItem> = items.values().filter(|i| crate::page::chart_ok(i)).collect();
        chart.sort_unstable_by(|a, b| b.popularity.total_cmp(&a.popularity).then_with(|| a.id.cmp(&b.id)));
        let chart_keys = chart.into_iter().map(|i| i.id).collect();
        let mut attr_keys: HashMap<&'static str, Vec<u32>> = HashMap::new();
        for spec in crate::attributes::SPECS {
            let mut v: Vec<&CatalogItem> = items.values().filter(|i| crate::page::passes_member_quality(i) && (spec.test)(i)).collect();
            crate::quality::sort_by_quality(&mut v);
            attr_keys.insert(spec.id, v.into_iter().map(|i| i.id).collect());
        }
        Self { items, chart_keys, attr_keys }
    }
}

impl std::ops::Deref for Catalog {
    type Target = HashMap<u32, CatalogItem>;
    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

pub type CatalogMap = Catalog;
pub type SharedCatalog = Arc<ArcSwap<CatalogMap>>;

/// Date (yyyymmdd) of the popularity overlay in use, 0 when there is none. Shown by /api/health.
pub static POPULARITY_ASOF: AtomicU32 = AtomicU32::new(0);

/// Daily freshness: `popularity_refresh.db` next to the catalog (written by ai-engine/scripts/refresh_popularity.py from
/// TMDB's daily export) replaces the stored popularity, so the Top 10, trending and ranking follow TMDB day by day.
/// A missing or unreadable file leaves the catalog as it is. Re-read on every catalog reload (every 5 minutes).
fn apply_popularity_overlay(map: &mut HashMap<u32, CatalogItem>, db_path: &str) {
    let Some(dir) = std::path::Path::new(db_path).parent() else { return };
    let path = dir.join("popularity_refresh.db");
    if !path.exists() {
        return;
    }
    let result = (|| -> anyhow::Result<(usize, u32)> {
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let asof: String = conn.query_row("SELECT v FROM meta WHERE k='asof'", [], |r| r.get(0))?;
        let digits: u32 = asof.replace('-', "").parse().unwrap_or(0);
        let mut stmt = conn.prepare("SELECT media_type, id, popularity FROM pop")?;
        let mut rows = stmt.query([])?;
        let mut n = 0usize;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(1)?;
            let pop: f64 = r.get(2)?;
            if let Some(item) = map.get_mut(&item_key(r.get_ref(0)?.as_str()?, id as u32)) {
                item.popularity = pop as f32;
                n += 1;
            }
        }
        Ok((n, digits))
    })();
    match result {
        Ok((n, digits)) => {
            POPULARITY_ASOF.store(digits, Ordering::Relaxed);
            info!("Popularity overlay applied to {n} titles (as of {digits})");
        }
        Err(e) => warn!("Popularity overlay not applied: {e}"),
    }
}

/// Parse a JSON array of strings (catalog_v2 stores lists that way). Empty on null or bad input.
fn json_list(raw: Option<String>, lower: bool) -> Vec<String> {
    raw.and_then(|r| serde_json::from_str::<Vec<String>>(&r).ok())
        .map(|v| if lower { v.into_iter().map(|s| s.to_lowercase()).collect() } else { v })
        .unwrap_or_default()
}

/// Load the entire catalog from SQLite into a HashMap keyed by `item_key` (type aware).
/// This is the ONLY place SQLite is accessed — never on the request hot path.
/// The real GB certificate as a label, None when the title has no GB rating.
pub fn gb_label(rank: u8) -> Option<&'static str> {
    Some(match rank { 1 => "U", 2 => "PG", 3 => "12", 4 => "15", 5 => "18", _ => return None })
}

/// The label a client shows for a certificate rank: the British rating, or NR (not rated) when there is none.
pub fn cert_label(cert: u8) -> &'static str {
    match cert {
        1 => "U",
        2 => "PG",
        3 => "12",
        4 => "15",
        5 => "18",
        _ => "NR",
    }
}

/// US certificate codes: 1 G, 2 PG, 3 PG-13, 4 R, 5 NC-17 (films), 6 TV-Y, 7 TV-Y7, 8 TV-G, 9 TV-PG, 10 TV-14, 11 TV-MA.
pub fn us_label(code: u8) -> Option<&'static str> {
    Some(match code {
        1 => "G", 2 => "PG", 3 => "PG-13", 4 => "R", 5 => "NC-17",
        6 => "TV-Y", 7 => "TV-Y7", 8 => "TV-G", 9 => "TV-PG", 10 => "TV-14", 11 => "TV-MA",
        _ => return None,
    })
}

/// The real GB rank (1 to 5) and US code of a certificates column, each 0 when the country has none that is a rating.
pub fn cert_codes(certs: Option<&str>) -> (u8, u8) {
    let Some(text) = certs else { return (0, 0) };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else { return (0, 0) };
    let get = |c: &str| v.get(c).and_then(|x| x.as_str()).map(|s| s.trim().to_ascii_uppercase());
    let gb = match get("GB").as_deref() {
        Some("U") | Some("UC") => 1,
        Some("PG") => 2,
        Some("12") | Some("12A") => 3,
        Some("15") => 4,
        Some("18") | Some("R18") => 5,
        _ => 0,
    };
    let us = match get("US").as_deref() {
        Some("G") => 1, Some("PG") => 2, Some("PG-13") => 3, Some("R") => 4, Some("NC-17") => 5,
        Some("TV-Y") => 6, Some("TV-Y7") | Some("TV-Y7-FV") => 7, Some("TV-G") => 8, Some("TV-PG") => 9, Some("TV-14") => 10, Some("TV-MA") => 11,
        _ => 0,
    };
    (gb, us)
}

/// British age rank from the certificates column (a JSON object of country to certificate). The GB certificate wins; otherwise the US one is
/// mapped to the nearest British rating. 0 when there is none or it is not a rating (for example "NR").
pub fn cert_rank(certs: Option<&str>) -> u8 {
    let Some(text) = certs else { return 0 };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else { return 0 };
    let get = |c: &str| v.get(c).and_then(|x| x.as_str()).map(|s| s.trim().to_ascii_uppercase());
    if let Some(gb) = get("GB") {
        match gb.as_str() {
            "U" | "UC" => return 1,
            "PG" => return 2,
            "12" | "12A" => return 3,
            "15" => return 4,
            "18" | "R18" => return 5,
            _ => {}
        }
    }
    match get("US").as_deref() {
        Some("G") | Some("TV-G") | Some("TV-Y") => 1,
        Some("PG") | Some("TV-PG") | Some("TV-Y7") | Some("TV-Y7-FV") => 2,
        Some("PG-13") => 3,
        Some("TV-14") => 4,
        Some("R") | Some("TV-MA") | Some("NC-17") => 5,
        _ => 0,
    }
}

/// One row of the catalog query as an item.
fn row_to_item(row: &rusqlite::Row) -> rusqlite::Result<CatalogItem> {
    // genres stored as comma-separated string: "Action, Drama, Sci-Fi"
    let genres_str: String = row.get(4).unwrap_or_default();
    let genres: Vec<String> = genres_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // release_year stored as TEXT — parse to i32
    let release_year: Option<i32> = row
        .get::<_, Option<String>>(3)
        .unwrap_or(None)
        .and_then(|s| s.parse().ok());

    // is_anime is INTEGER 0/1
    let is_anime: bool = row.get::<_, i64>(10).unwrap_or(0) != 0;

    let media_type: String = row.get(1).unwrap_or_default();
    let tmdb_id = row.get::<_, i64>(0)? as u32;

    let title: String = row.get(2).unwrap_or_default();
    let keywords = json_list(row.get::<_, Option<String>>(18).unwrap_or(None), true);
    let tv_type = row.get::<_, Option<String>>(16).unwrap_or(None).filter(|s| !s.is_empty());
    let overview: String = row.get(5).unwrap_or_default();
    let poster_path: Option<String> = row.get(6)?;
    let backdrop_path: Option<String> = row.get(7)?;
    let certs: Option<String> = row.get::<_, Option<String>>(20).unwrap_or(None);
    let (gb, us) = cert_codes(certs.as_deref());
    let flags = compute_flags(&title, &keywords, tv_type.as_deref(), &overview, poster_path.as_deref(), backdrop_path.as_deref());
    Ok(CatalogItem {
        id:            item_key(&media_type, tmdb_id),
        tmdb_id,
        media_type,
        title,
        release_year,
        genres,
        overview,
        poster_path,
        backdrop_path,
        vote_average:  row.get::<_, f64>(8).unwrap_or(0.0) as f32,
        popularity:    row.get::<_, f64>(9).unwrap_or(0.0) as f32,
        is_anime,
        tagline:       row.get(11)?,
        director:      row.get(12)?,
        vote_count:    row.get::<_, i64>(13).unwrap_or(0).max(0) as u32,
        adult:         row.get::<_, i64>(14).unwrap_or(0) != 0,
        original_language: row.get::<_, Option<String>>(15).unwrap_or(None).unwrap_or_default(),
        tv_type,
        countries:     json_list(row.get::<_, Option<String>>(17).unwrap_or(None), false),
        keywords,
        runtime:       row.get::<_, Option<i64>>(19).unwrap_or(None).unwrap_or(0).max(0) as u32,
        cert:          cert_rank(certs.as_deref()),
        cert_gb:       gb,
        cert_us:       us,
        flags,
    })
}

/// The catalog query: the column list of `catalog` (older files lack some columns, which then take defaults) and the name of the id column.
fn catalog_sql(conn: &Connection) -> Result<(String, &'static str)> {
    // catalog_v2 has `tmdb_id` (the primary key `id` there is the internal key) plus optional extra columns.
    let cols: std::collections::HashSet<String> = conn
        .prepare("PRAGMA table_info(catalog)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .collect();
    let id_col = if cols.contains("tmdb_id") { "tmdb_id" } else { "id" };
    let opt = |name: &str, default: &str| if cols.contains(name) { name.to_string() } else { format!("{default} AS {name}") };
    let sql = format!(
        "SELECT {id_col}, media_type, title, release_year, genres,
                overview, poster_path, backdrop_path,
                vote_average, popularity, is_anime, tagline, director,
                {vc}, {ad}, {ol}, {tt}, {oc}, {kw}, {rt}, {ce}
         FROM catalog",
        vc = opt("vote_count", "0"),
        ad = opt("adult", "0"),
        ol = opt("original_language", "''"),
        tt = opt("tv_type", "NULL"),
        oc = opt("origin_country", "NULL"),
        kw = opt("keywords", "NULL"),
        rt = opt("runtime", "0"),
        ce = opt("certs", "NULL"),
    );
    Ok((sql, id_col))
}

/// Loads the whole catalog from SQLite. The rows are decoded (lists parsed from JSON, flags computed) by several threads, each reading the rows whose id has its remainder.
pub fn load_catalog(db_path: &str) -> Result<CatalogMap> {
    let (sql, id_col) = catalog_sql(&Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?)?;
    let parts = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8);
    let results: Vec<Result<Vec<CatalogItem>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..parts)
            .map(|k| {
                let sql = format!("{sql} WHERE {id_col} % {parts} = {k}");
                s.spawn(move || -> Result<Vec<CatalogItem>> {
                    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                    let mut stmt = conn.prepare(&sql)?;
                    let items = stmt.query_map([], row_to_item)?.collect::<rusqlite::Result<Vec<_>>>()?;
                    Ok(items)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("catalog loader panicked")).collect()
    });
    let mut map = HashMap::with_capacity(900_000);
    for part in results {
        for item in part? {
            map.insert(item.id, item);
        }
    }

    apply_popularity_overlay(&mut map, db_path);
    let catalog = Catalog::new(map);
    info!("Catalog loaded: {} items, {} on the chart list", catalog.len(), catalog.chart_keys.len());
    Ok(catalog)
}

/// Create a shared, atomically-swappable catalog pointer.
pub fn make_shared_catalog(db_path: &str) -> Result<SharedCatalog> {
    let map = load_catalog(db_path)?;
    Ok(Arc::new(ArcSwap::from_pointee(map)))
}

/// Size and modification time of the files the catalog is made from (the database and the daily popularity overlay): a reload is needed only when one changed.
fn source_stamp(db_path: &str) -> [Option<(u64, std::time::SystemTime)>; 2] {
    let stamp = |p: &std::path::Path| std::fs::metadata(p).ok().and_then(|m| Some((m.len(), m.modified().ok()?)));
    let db = std::path::Path::new(db_path);
    [stamp(db), db.parent().and_then(|d| stamp(&d.join("popularity_refresh.db")))]
}

/// Background task: every 5 minutes checks whether the catalog database or the popularity overlay changed and, only then, re-loads the catalog from SQLite
/// and atomically swaps the shared pointer. All readers see a consistent snapshot with zero locking.
pub fn spawn_catalog_refresh(catalog: SharedCatalog, db_path: String) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300));
        interval.tick().await; // skip immediate first tick
        let mut loaded = source_stamp(&db_path);
        loop {
            interval.tick().await;
            let now = source_stamp(&db_path);
            if now == loaded {
                continue;
            }
            match tokio::task::spawn_blocking({
                let p = db_path.clone();
                move || load_catalog(&p)
            })
            .await
            {
                Ok(Ok(new_map)) => {
                    catalog.store(Arc::new(new_map));
                    loaded = now;
                    info!("Catalog refreshed");
                }
                Ok(Err(e)) => warn!("Catalog refresh error: {e}"),
                Err(e)     => warn!("Catalog refresh join error: {e}"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("streams_cat_{}_{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("c.db").to_string_lossy().to_string()
    }

    #[test]
    fn v2_catalog_keeps_movie_and_tv_with_the_same_tmdb_id() {
        let path = temp_db("v2");
        let c = Connection::open(&path).unwrap();
        c.execute_batch(
            "CREATE TABLE catalog (id INTEGER PRIMARY KEY, tmdb_id INTEGER, media_type TEXT, title TEXT, release_year TEXT,
               genres TEXT, overview TEXT, poster_path TEXT, backdrop_path TEXT, vote_average REAL, popularity REAL,
               is_anime INTEGER, tagline TEXT, director TEXT, vote_count INTEGER, adult INTEGER, original_language TEXT, tv_type TEXT);
             INSERT INTO catalog VALUES (1396, 1396, 'movie', 'Mirror', '1975', 'Drama', 'o', '/p', NULL, 7.9, 4.2, 0, '', 'Andrei Tarkovsky', 3000, 0, 'ru', NULL);
             INSERT INTO catalog VALUES (2147485044, 1396, 'tv', 'Breaking Bad', '2008', 'Drama, Crime', 'o', '/p', '/b', 8.9, 300.0, 0, 'Change the equation.', 'Vince Gilligan', 15000, 0, 'en', 'Scripted');",
        ).unwrap();
        drop(c);
        let map = load_catalog(&path).unwrap();
        assert_eq!(map.len(), 2);
        let movie = &map[&item_key("movie", 1396)];
        let tv = &map[&item_key("tv", 1396)];
        assert_eq!((movie.title.as_str(), movie.tmdb_id), ("Mirror", 1396));
        assert_eq!((tv.title.as_str(), tv.tmdb_id), ("Breaking Bad", 1396));
        assert_eq!(tv.id, 1396 | TV_BIT);
        assert_eq!((tv.vote_count, tv.tv_type.as_deref(), tv.original_language.as_str()), (15000, Some("Scripted"), "en"));
        assert_eq!(resolve_key(&map, 1396, Some("tv")), Some(1396 | TV_BIT));
        assert_eq!(resolve_key(&map, 1396, Some("movie")), Some(1396));
        assert_eq!(resolve_key(&map, 1396, None), Some(1396), "movie first without a media type");
        assert_eq!(resolve_key(&map, 999, None), None);
    }

    #[test]
    fn popularity_overlay_replaces_stored_popularity() {
        let path = temp_db("overlay");
        let c = Connection::open(&path).unwrap();
        c.execute_batch(
            "CREATE TABLE catalog (id INTEGER PRIMARY KEY, tmdb_id INTEGER, media_type TEXT, title TEXT, release_year TEXT,
               genres TEXT, overview TEXT, poster_path TEXT, backdrop_path TEXT, vote_average REAL, popularity REAL,
               is_anime INTEGER, tagline TEXT, director TEXT);
             INSERT INTO catalog VALUES (550, 550, 'movie', 'Fight Club', '1999', 'Drama', 'o', '/p', '/b', 8.4, 5.0, 0, '', '');
             INSERT INTO catalog VALUES (2147484198, 550, 'tv', 'Some Show', '2020', 'Drama', 'o', '/p', '/b', 7.0, 5.0, 0, '', '');",
        ).unwrap();
        drop(c);
        let dir = std::path::Path::new(&path).parent().unwrap();
        let o = Connection::open(dir.join("popularity_refresh.db")).unwrap();
        o.execute_batch(
            "CREATE TABLE pop (media_type TEXT, id INTEGER, popularity REAL, PRIMARY KEY (media_type, id));
             CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
             INSERT INTO pop VALUES ('movie', 550, 77.5);
             INSERT INTO meta VALUES ('asof', '2026-10-04');",
        ).unwrap();
        drop(o);
        let map = load_catalog(&path).unwrap();
        assert_eq!(map[&550].popularity, 77.5, "movie overlaid");
        assert_eq!(map[&(550 | TV_BIT)].popularity, 5.0, "the series with the same id is not touched");
        assert_eq!(POPULARITY_ASOF.load(Ordering::Relaxed), 20261004);
    }

    #[test]
    fn legacy_catalog_without_new_columns_still_loads() {
        let path = temp_db("legacy");
        let c = Connection::open(&path).unwrap();
        c.execute_batch(
            "CREATE TABLE catalog (id INTEGER PRIMARY KEY, media_type TEXT, title TEXT, release_year TEXT, director TEXT,
               cast_members TEXT, tagline TEXT, genres TEXT, overview TEXT, poster_path TEXT, backdrop_path TEXT,
               vote_average REAL, popularity REAL, is_anime INTEGER);
             INSERT INTO catalog VALUES (1399, 'tv', 'Game of Thrones', '2011', '', '', '', 'Drama', 'o', '/p', '/b', 8.4, 215.0, 0);
             INSERT INTO catalog VALUES (550, 'movie', 'Fight Club', '1999', '', '', '', 'Drama', 'o', '/p', '/b', 8.4, 39.0, 0);",
        ).unwrap();
        drop(c);
        let map = load_catalog(&path).unwrap();
        assert_eq!(map.len(), 2);
        let got = &map[&(1399 | TV_BIT)];
        assert_eq!((got.tmdb_id, got.media_type.as_str(), got.vote_count, got.adult), (1399, "tv", 0, false));
        assert!(map.contains_key(&550));
    }
}

#[cfg(test)]
mod cert_tests {
    use super::cert_rank;

    #[test]
    fn british_certificate_wins_over_the_us_one() {
        assert_eq!(cert_rank(Some(r#"{"GB":"15","US":"PG"}"#)), 4);
        assert_eq!(cert_rank(Some(r#"{"GB":"12A"}"#)), 3);
        assert_eq!(cert_rank(Some(r#"{"GB":"U","US":"R"}"#)), 1);
    }

    #[test]
    fn the_us_certificate_is_mapped_when_there_is_no_british_one() {
        assert_eq!(cert_rank(Some(r#"{"US":"TV-Y7"}"#)), 2);
        assert_eq!(cert_rank(Some(r#"{"US":"PG-13"}"#)), 3);
        assert_eq!(cert_rank(Some(r#"{"US":"TV-MA"}"#)), 5);
    }

    #[test]
    fn both_real_certificates_are_kept() {
        use super::{cert_codes, us_label};
        assert_eq!(cert_codes(Some(r#"{"GB":"15","US":"TV-14"}"#)), (4, 10));
        assert_eq!(cert_codes(Some(r#"{"US":"PG-13"}"#)), (0, 3));
        assert_eq!(cert_codes(Some(r#"{"GB":"12A"}"#)), (3, 0));
        assert_eq!(cert_codes(Some(r#"{"US":"NR"}"#)), (0, 0));
        assert_eq!((us_label(10), us_label(4), us_label(0)), (Some("TV-14"), Some("R"), None));
    }

    #[test]
    fn missing_or_unrated_is_unknown() {
        assert_eq!(cert_rank(None), 0);
        assert_eq!(cert_rank(Some(r#"{"US":"NR"}"#)), 0);
        assert_eq!(cert_rank(Some("not json")), 0);
        assert_eq!(cert_rank(Some(r#"{"GB":"X","US":"NR"}"#)), 0);
    }
}
