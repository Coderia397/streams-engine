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
