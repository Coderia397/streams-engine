/// Pointer tables for genre rows (notes/52): for every genre of the site's pickers (Home, Movies, TV) the groups that can fill rows of it, found with the engine's own genre rules
/// and quality gates, written into the database as `genre_pointers` (one line per surface, genre and world level 2 group) and `genre_coverage` (one line per surface and genre).
///   build-pointers --db /Users/ibrahim/streams-catalog/streams_v3.db
use anyhow::Result;
use clap::Parser;
use rusqlite::Connection;
use std::collections::HashMap;
use streams_engine::catalog::load_catalog;
use streams_engine::genre_spec::resolve;
use streams_engine::groups;
use streams_engine::page::{meets_vote_floor, passes_member_quality};

#[derive(Parser)]
struct Args {
    #[arg(long)] db: String,
    /// Smallest number of quality titles of a genre a group needs to make a row
    #[arg(long, default_value_t = 8)] min_row: usize,
}

const MOVIE: &[&str] = &["Action", "Anime", "Black Stories", "British", "Classics", "Comedies", "Crime", "Cult", "Documentaries", "Dramas", "European", "Fantasy", "Halloween", "Hollywood", "Horror", "Independent", "International", "Kids & Family", "LGBTQ", "Music & Musicals", "Romance", "Sci-Fi", "Shorts", "Sports", "Stand-up Comedy", "Thriller"];
const TV: &[&str] = &["Action", "Anime", "Black Stories", "British", "Comedies", "Crime", "Documentary", "Dramas", "Emmys", "European", "Halloween", "Horror", "International", "Kids", "Mystery", "Reality", "Romance", "Sci-Fi & Fantasy", "Science & Nature", "Sports", "Stand-up & Chat Shows", "Teen", "Thriller", "US"];
const HOME: &[&str] = &["Action", "Anime", "Astrology", "Black Stories", "Book Adaptations", "British", "Classics", "Comedies", "Crime", "Cult", "Documentaries", "Dramas", "European", "Fantasy", "Hollywood", "Horror", "Independent", "International", "Kids & Family", "Moods", "Music & Musicals", "Pride", "Romance", "Sci-Fi", "Shorts", "Sport", "Thriller", "Kids", "Reality", "Sci-Fi & Fantasy", "Science & Nature", "Documentary Series", "Teen", "US", "Mysteries"];

/// Levels from the finest to the coarsest: world level 2 (3,138 groups), themes level 3 (621), level 2 (386), level 1 (141), world level 1 (47)
const LEVELS: &[&str] = &["w2", "th3", "th2", "th1", "w1"];
fn group_of(g: &groups::GroupIds, level: &str) -> i32 {
    match level { "w2" => g.w2, "th3" => g.th3, "th2" => g.theme, "th1" => g.th1, _ => g.w1 }
}

fn main() -> Result<()> {
    let a = Args::parse();
    let catalog = load_catalog(&a.db)?;
    let gmap = groups::load(&a.db);
    let conn = Connection::open(&a.db)?;
    conn.execute_batch("DROP TABLE IF EXISTS genre_pointers; DROP TABLE IF EXISTS genre_coverage;
        CREATE TABLE genre_pointers(surface TEXT, genre TEXT, level TEXT, group_id INTEGER, n_quality INTEGER, rank INTEGER, PRIMARY KEY(surface, genre, group_id));
        CREATE TABLE genre_coverage(surface TEXT, genre TEXT, pool INTEGER, level TEXT, rows INTEGER, covered INTEGER, coverage REAL, rows_w2 INTEGER, rows_th2 INTEGER, rows_w1 INTEGER);")?;
    let tx = conn.unchecked_transaction()?;
    println!("{:<6} {:<22} {:>6}  {:<4} {:>5} {:>8}   rows by level w2 / th3 / th2 / th1 / w1", "page", "genre", "pool", "use", "rows", "coverage");
    for (surface, list) in [("movie", MOVIE), ("tv", TV), ("both", HOME)] {
        for name in list {
            let spec = resolve(name, surface);
            let pool: Vec<&streams_engine::catalog::CatalogItem> = catalog.values().filter(|item| {
                (surface == "both" || item.media_type == surface) && spec.matches(item) && passes_member_quality(item) && meets_vote_floor(item, true)
                    && (spec.keeps_children() || !item.genres.iter().any(|g| g == "Kids" || g == "Family"))
            }).collect();
            // rows a page can show: at least 8 titles, and as many as the pool supports (a page holds up to 40 rows of about 15 titles)
            let want = 40usize.min((pool.len() / 20).max(6));
            let mut per_level: Vec<(String, Vec<(i32, usize)>)> = Vec::new();
            for level in LEVELS {
                let mut by: HashMap<i32, usize> = HashMap::new();
                for item in &pool { if let Some(g) = gmap.get(&item.id) { let id = group_of(g, level); if id >= 0 { *by.entry(id).or_insert(0) += 1; } } }
                let mut rows: Vec<(i32, usize)> = by.into_iter().filter(|(_, n)| *n >= a.min_row).collect();
                rows.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
                per_level.push((level.to_string(), rows));
            }
            // the finest level that gives enough rows; if none does, the level with the most rows
            let chosen = per_level.iter().position(|(_, r)| r.len() >= want).unwrap_or_else(|| per_level.iter().enumerate().max_by_key(|(_, (_, r))| r.len()).map(|(i, _)| i).unwrap_or(0));
            let (level, rows) = &per_level[chosen];
            let covered: usize = rows.iter().map(|r| r.1).sum();
            for (rank, (g, n)) in rows.iter().enumerate() {
                tx.execute("INSERT INTO genre_pointers VALUES (?1,?2,?3,?4,?5,?6)", rusqlite::params![surface, name, level, g, *n as i64, rank as i64])?;
            }
            let cnt = |l: &str| per_level.iter().find(|(x, _)| x == l).map(|(_, r)| r.len()).unwrap_or(0);
            tx.execute("INSERT INTO genre_coverage VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)", rusqlite::params![surface, name, pool.len() as i64, level, rows.len() as i64, covered as i64, covered as f64 / pool.len().max(1) as f64, cnt("w2") as i64, cnt("th2") as i64, cnt("w1") as i64])?;
            println!("{:<6} {:<22} {:>6}  {:<4} {:>5} {:>7.0}%   {} / {} / {} / {} / {}", surface, name, pool.len(), level, rows.len(), 100.0 * covered as f64 / pool.len().max(1) as f64, cnt("w2"), cnt("th3"), cnt("th2"), cnt("th1"), cnt("w1"));
        }
    }
    tx.commit()?;
    Ok(())
}
