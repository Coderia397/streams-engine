//! Reading the catalog from the database: the query, the decoding of a row, the shared pointer.
use super::*;

/// Parse a JSON array of strings (catalog_v2 stores lists that way). Empty on null or bad input.
pub(crate) fn json_list(raw: Option<String>, lower: bool) -> Vec<String> {
    raw.and_then(|r| serde_json::from_str::<Vec<String>>(&r).ok())
        .map(|v| if lower { v.into_iter().map(|s| s.to_lowercase()).collect() } else { v })
        .unwrap_or_default()
}

/// One row of the catalog query as an item.
pub(crate) fn row_to_item(row: &rusqlite::Row) -> rusqlite::Result<CatalogItem> {
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
        networks:      json_list(row.get::<_, Option<String>>(21).unwrap_or(None), false),
        cert:          cert_rank(certs.as_deref()),
        cert_gb:       gb,
        cert_us:       us,
        flags,
    })
}

/// The catalog query: the column list of `catalog` (older files lack some columns, which then take defaults) and the name of the id column.
pub(crate) fn catalog_sql(conn: &Connection) -> Result<(String, &'static str)> {
    // catalog_v2 has `tmdb_id` (the primary key `id` there is the internal key) plus optional extra columns.
    let cols: crate::det::HashSet<String> = conn
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
                {vc}, {ad}, {ol}, {tt}, {oc}, {kw}, {rt}, {ce}, {nw}
         FROM catalog",
        vc = opt("vote_count", "0"),
        ad = opt("adult", "0"),
        ol = opt("original_language", "''"),
        tt = opt("tv_type", "NULL"),
        oc = opt("origin_country", "NULL"),
        kw = opt("keywords", "NULL"),
        rt = opt("runtime", "0"),
        ce = opt("certs", "NULL"),
        nw = opt("networks", "NULL"),
    );
    Ok((sql, id_col))
}

/// Load the entire catalog from SQLite into a HashMap keyed by `item_key` (type aware).
/// This is the ONLY place SQLite is accessed — never on the request hot path.
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
    let mut map = crate::det::map_with_capacity(900_000);
    for part in results {
        for item in part? {
            map.insert(item.id, item);
        }
    }

    apply_popularity_overlay(&mut map, db_path);
    apply_awards(&mut map, db_path);
    crate::catalog::overlays::apply_netflix_tags(&mut map, db_path);
    apply_categories(&mut map, db_path);
    let catalog = Catalog::new(map);
    info!("Catalog loaded: {} items, {} on the chart list", catalog.len(), catalog.chart_keys.len());
    Ok(catalog)
}

/// Create a shared, atomically-swappable catalog pointer.
pub fn make_shared_catalog(db_path: &str) -> Result<SharedCatalog> {
    let map = load_catalog(db_path)?;
    Ok(Arc::new(ArcSwap::from_pointee(map)))
}
