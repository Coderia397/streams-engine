//! What is added to the titles after loading: the daily popularity, awards, judged categories and topics.
use super::*;

/// Date (yyyymmdd) of the popularity overlay in use, 0 when there is none. Shown by /api/health.
pub static POPULARITY_ASOF: AtomicU32 = AtomicU32::new(0);

/// Daily freshness: `popularity_refresh.db` next to the catalog (written by ai-engine/scripts/refresh_popularity.py from
/// TMDB's daily export) replaces the stored popularity, so the Top 10, trending and ranking follow TMDB day by day.
/// A missing or unreadable file leaves the catalog as it is. Re-read on every catalog reload (every 5 minutes).
pub(crate) fn apply_popularity_overlay(map: &mut HashMap<u32, CatalogItem>, db_path: &str) {
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

/// Adds the awards of `awards_v1.json` (next to the database) to the keywords of the titles as `award:<family>:won` or `award:<family>:nominated`, so any
/// page rule or banner line can ask for them. The file is `{ "emmy": { "<tmdb id>": { "won": n, "nominated": n } }, "oscar": ..., "bafta": ..., "globe": ... }`
/// with `_tv` after the family name for series ids (`emmy` is series only). A missing file adds nothing.
pub(crate) fn apply_awards(map: &mut HashMap<u32, CatalogItem>, db_path: &str) {
    let Some(path) = std::path::Path::new(db_path).parent().map(|d| d.join("awards_v1.json")) else { return };
    let Some(raw) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { return };
    let Some(obj) = raw.as_object() else { return };
    let mut added = 0usize;
    for (fam, titles) in obj {
        let (base, media) = match fam.strip_suffix("_tv") { Some(b) => (b, "tv"), None if fam == "emmy" => ("emmy", "tv"), None => (fam.as_str(), "movie") };
        let Some(titles) = titles.as_object() else { continue };
        for (id, e) in titles {
            let Ok(id) = id.parse::<u32>() else { continue };
            let Some(item) = map.get_mut(&item_key(media, id)) else { continue };
            let kind = if e.get("won").and_then(|w| w.as_u64()).unwrap_or(0) > 0 { "won" } else { "nominated" };
            let tag = format!("award:{base}:{kind}");
            if !item.keywords.contains(&tag) {
                item.keywords.push(tag);
                added += 1;
            }
        }
    }
    info!("Awards: {added} titles marked from {}", path.display());
}

/// Adds Netflix's per title tags (`netflix_tags.json` next to the database: item key -> ["Suspenseful", "Organized Crime", ...], note 102) to the keywords of the
/// titles as `nf:<tag>` in lower case, so a composed row can ask for a mood or subject Netflix itself gave the title. A missing file adds nothing.
pub(crate) fn apply_netflix_tags(map: &mut HashMap<u32, CatalogItem>, db_path: &str) {
    let Some(path) = std::path::Path::new(db_path).parent().map(|d| d.join("netflix_tags.json")) else { return };
    let Some(raw) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<HashMap<String, Vec<String>>>(&t).ok()) else { return };
    let mut titles = 0usize;
    for (key, tags) in raw {
        let Ok(key) = key.parse::<u32>() else { continue };
        let Some(item) = map.get_mut(&key) else { continue };
        titles += 1;
        for t in tags {
            let tag = format!("nf:{}", t.trim().to_lowercase());
            if !item.keywords.contains(&tag) {
                item.keywords.push(tag);
            }
        }
    }
    info!("Netflix tags: {titles} titles marked from {}", path.display());
}

/// The genre pages whose membership is learned from Netflix's own pages (`genre_signals_v1.json`, ai-engine/pipeline/netflix_membership.py, note 107), as
/// (media type, genre); bit i of GENRE_SIGNALS_LOADED is set when entry i was applied. A member carries the keyword `learned:<genre in lower case>`.
pub const LEARNED_GENRES: &[(&str, &str)] = &[("tv", "Romance"), ("tv", "Horror"), ("tv", "Teen")];
pub static GENRE_SIGNALS_LOADED: AtomicU32 = AtomicU32::new(0);

/// Whether the membership of this genre for this media type comes from `genre_signals_v1.json`.
pub fn genre_learned(media_type: &str, genre: &str) -> bool {
    LEARNED_GENRES
        .iter()
        .position(|(m, g)| *m == media_type && *g == genre)
        .is_some_and(|i| GENRE_SIGNALS_LOADED.load(Ordering::Relaxed) & (1 << i) != 0)
}

/// Marks the members of the learned genre pages (`genre_signals_v1.json` next to the database: genres -> media type -> genre -> {tags, keywords, keywords_none}).
/// A title with Netflix tags (one Netflix carries and has judged) is a member when it carries one of the learned tags; any other title when it carries one of
/// the learned TMDB keywords and none of the learned exclusions. Runs after `apply_netflix_tags`. A missing file leaves the keyword rules of genre_spec.rs in place.
pub(crate) fn apply_genre_signals(map: &mut HashMap<u32, CatalogItem>, db_path: &str) {
    let Some(path) = std::path::Path::new(db_path).parent().map(|d| d.join("genre_signals_v1.json")) else { return };
    let Some(raw) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { return };
    let words = |v: &serde_json::Value| -> std::collections::HashSet<String> {
        v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_lowercase())).collect()).unwrap_or_default()
    };
    let mut loaded = 0u32;
    for (i, (media, genre)) in LEARNED_GENRES.iter().enumerate() {
        let Some(spec) = raw.pointer(&format!("/genres/{media}/{genre}")) else { continue };
        let (tags, keywords, none) = (words(&spec["tags"]), words(&spec["keywords"]), words(&spec["keywords_none"]));
        if tags.is_empty() && keywords.is_empty() {
            continue;
        }
        let mark = format!("learned:{}", genre.to_lowercase());
        let mut members = 0usize;
        for item in map.values_mut().filter(|it| it.media_type == *media) {
            let tagged = item.keywords.iter().any(|k| k.starts_with("nf:"));
            let member = if tagged {
                item.keywords.iter().any(|k| tags.contains(k))
            } else {
                // a keyword member is kept off by a learned exclusion: a keyword, or a TMDB genre as `genre:<name>`
                item.keywords.iter().any(|k| keywords.contains(k))
                    && !item.keywords.iter().any(|k| none.contains(k))
                    && !item.genres.iter().any(|g| none.contains(&format!("genre:{}", g.to_lowercase())))
            };
            if member {
                item.keywords.push(mark.clone());
                members += 1;
            }
        }
        loaded |= 1 << i;
        info!("Learned genre {media} {genre}: {members} members ({} tags, {} keywords)", tags.len(), keywords.len());
    }
    GENRE_SIGNALS_LOADED.store(loaded, Ordering::Relaxed);
}

/// True once `categories_v1.json` was applied: the pages it covers then use the judged membership alone (the old keyword rules only propose candidates).
pub static CATEGORIES_LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Adds the judged memberships of `categories_v1.json` (page categories) and `topics_v1.json` (topic rows) next to the database to the keywords of the titles as
/// `cat:<slug>` and `topic:<slug>`. Both files are `{ "<slug>": { "<movie|tv>:<tmdb id>": probability } }`, only titles the judge accepted. A missing file adds nothing.
pub(crate) fn apply_categories(map: &mut HashMap<u32, CatalogItem>, db_path: &str) {
    let Some(dir) = std::path::Path::new(db_path).parent() else { return };
    for (file, prefix) in [("categories_v1.json", "cat"), ("topics_v1.json", "topic")] {
        let Some(raw) = std::fs::read_to_string(dir.join(file)).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { continue };
        let Some(obj) = raw.as_object() else { continue };
        let mut added = 0usize;
        for (slug, members) in obj {
            let Some(members) = members.as_object() else { continue };
            let tag = format!("{prefix}:{slug}");
            for key in members.keys() {
                let Some((media, id)) = key.split_once(':') else { continue };
                let Ok(id) = id.parse::<u32>() else { continue };
                if let Some(item) = map.get_mut(&item_key(media, id)) {
                    item.keywords.push(tag.clone());
                    added += 1;
                }
            }
        }
        if prefix == "cat" && added > 0 { CATEGORIES_LOADED.store(true, std::sync::atomic::Ordering::Relaxed); }
        info!("{file}: {added} memberships added");
    }
}
