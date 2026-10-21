//! The rows of the desktop page and the shape of a title in them.
use super::*;

pub(super) struct RowOut {
    pub(super) title: String,
    /// Translation key of the title (`row.np_new_movies`) and its arguments; `title` is the English fallback
    pub(super) key: &'static str,
    pub(super) args: Value,
    pub(super) row_type: &'static str,
    pub(super) kind: &'static str,
    pub(super) items: Vec<Cand>,
}

pub(super) fn item_json(c: &Cand, rank: Option<usize>, today: i64) -> Value {
    json!({
        "id": c.tmdb_id, "media_type": c.media_type, "title": c.title,
        "release_year": c.release_year.map(|y| y.to_string()),
        "release_date": if c.release_date.is_empty() { Value::Null } else { json!(c.release_date) },
        "days_until": parse_days(&c.release_date).map(|d| d - today),
        "genres": c.genres.join(", "), "overview": c.overview, "poster_path": c.poster_path, "backdrop_path": c.backdrop_path,
        "vote_average": c.vote_average, "tv_type": c.tv_type, "language": c.language,
        "rating": crate::catalog::cert_label(c.cert), "rating_gb": crate::catalog::gb_label(c.cert_gb), "rating_us": crate::catalog::us_label(c.cert_us),
        "rank": rank,
        "badges": crate::contract::item_badges(rank, (!c.release_date.is_empty()).then_some(c.release_date.as_str()), today),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn desktop_rows(catalog: &CatalogMap, fresh: &FreshStore, pool: &[&Cand], maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, today: &str, today_n: i64, year: i32, limit: usize, session: &str, slot: usize) -> Vec<RowOut> {
    let mut rows: Vec<RowOut> = Vec::new();
    // rows are cut to `limit` after titles shared with an earlier row are removed, so each starts from a wider list
    let wide = limit * 4;
    let recent = |c: &&Cand, days: i64| c.released(today) && c.days_since(today_n).is_some_and(|d| d <= days);

    // the charts are built here and placed among the other rows at the end: how many there are, which, and where changes with the session and the slot
    let mut charts: Vec<RowOut> = vec![
        RowOut { title: "Top 10 Movies Today".into(), key: "row.np_top10_movies", args: Value::Null, row_type: "top_ten", kind: "chart", items: chart(catalog, fresh, maturity, profiles, "movie", today, year) },
        RowOut { title: "Top 10 Series Today".into(), key: "row.np_top10_series", args: Value::Null, row_type: "top_ten", kind: "chart", items: chart(catalog, fresh, maturity, profiles, "tv", today, year) },
    ];

    // what is being watched today, in TMDB's own order
    let mut trending: Vec<&Cand> = pool.iter().copied().filter(|c| c.released(today) && c.in_list("trending_day")).collect();
    trending.sort_by_key(|c| c.rank_in("trending_day"));
    rows.push(RowOut { title: "Trending Today".into(), key: "row.np_trending_today", args: Value::Null, row_type: "vibe", kind: "trending", items: trending.into_iter().take(limit).cloned().collect() });

    let new_movies: Vec<&Cand> = pool.iter().copied().filter(|c| c.media_type == "movie" && recent(c, 45)).collect();
    rows.push(RowOut { title: "New Movies".into(), key: "row.np_new_movies", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(new_movies, wide) });

    let episodes: Vec<&Cand> = pool.iter().copied().filter(|c| c.media_type == "tv" && (c.in_list("airing_today") || c.in_list("on_the_air")) && c.vote_average >= 6.0 && !matches!(c.tv_type.as_deref(), Some("Talk Show") | Some("News") | Some("Soap"))).collect();
    rows.push(RowOut { title: "New Episodes This Week".into(), key: "row.np_new_episodes", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(episodes, wide) });

    let mut soon_movies: Vec<Cand> = take_top(pool.iter().copied().filter(|c| c.media_type == "movie" && c.upcoming(today) && c.days_since(today_n).is_some_and(|d| d >= -120)).collect(), wide);
    soon_movies.sort_by(|a, b| a.release_date.cmp(&b.release_date));
    rows.push(RowOut { title: "Coming Soon to Cinemas".into(), key: "row.np_coming_soon_cinemas", args: Value::Null, row_type: "vibe", kind: "soon", items: soon_movies });

    let acclaimed: Vec<&Cand> = pool.iter().copied().filter(|c| recent(c, 270) && c.vote_count >= 100 && c.vote_average >= 7.2).collect();
    rows.push(RowOut { title: "New and Acclaimed".into(), key: "row.np_new_acclaimed", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(acclaimed, wide) });

    let mut soon_tv: Vec<Cand> = take_top(pool.iter().copied().filter(|c| c.media_type == "tv" && c.upcoming(today) && c.days_since(today_n).is_some_and(|d| d >= -150)).collect(), wide);
    soon_tv.sort_by(|a, b| a.release_date.cmp(&b.release_date));
    rows.push(RowOut { title: "Coming Soon: Series".into(), key: "row.np_coming_soon_series", args: Value::Null, row_type: "vibe", kind: "soon", items: soon_tv });

    let new_tv: Vec<&Cand> = pool.iter().copied().filter(|c| c.media_type == "tv" && recent(c, 90)).collect();
    rows.push(RowOut { title: "New Series".into(), key: "row.np_new_series", args: Value::Null, row_type: "vibe", kind: "new", items: take_top(new_tv, wide) });

    // languages with enough new and trending titles each get a row, the biggest first
    let fresh_released: Vec<&Cand> = pool.iter().copied().filter(|c| recent(c, 120)).collect();
    let mut by_lang: HashMap<&str, usize> = HashMap::default();
    for c in &fresh_released {
        if c.language != "en" && crate::row_label::language_label(&c.language).is_some() {
            *by_lang.entry(c.language.as_str()).or_insert(0) += 1;
        }
    }
    let mut langs: Vec<(&str, usize)> = by_lang.into_iter().filter(|(_, n)| *n >= MIN_ROW_ITEMS).collect();
    langs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    // the biggest language also gets a Top 10 of its own among the charts
    if let Some((code, _)) = langs.first() {
        let (code, label) = (*code, crate::row_label::language_label(code).unwrap_or(code));
        charts.push(RowOut { title: format!("Top 10 {label} Titles Today"), key: "row.np_top10_language", args: json!({ "language": label }), row_type: "top_ten", kind: "chart", items: chart_where(catalog, fresh, maturity, profiles, today, year, |i| i.original_language == code) });
    }
    // the four biggest languages compete, two get a row: which two changes with the session and the slot
    for (code, _) in pick_by_noise(langs, 4, 2, session, slot) {
        let v: Vec<&Cand> = fresh_released.iter().copied().filter(|c| c.language == code).collect();
        rows.push(RowOut { title: format!("New in {}", crate::row_label::language_label(code).unwrap_or(code)), key: "row.np_new_in_language", args: json!({ "language": crate::row_label::language_label(code).unwrap_or(code) }), row_type: "vibe", kind: "language", items: take_top(v, wide) });
    }

    // genres with the most new titles
    let mut by_genre: HashMap<&str, usize> = HashMap::default();
    for c in pool.iter().filter(|c| recent(c, 90)) {
        for g in &c.genres {
            if crate::row_label::genre_noun(g).is_some() && g != "Drama" {
                *by_genre.entry(g.as_str()).or_insert(0) += 1;
            }
        }
    }
    let mut genres: Vec<(&str, usize)> = by_genre.into_iter().filter(|(_, n)| *n >= MIN_ROW_ITEMS + 4).collect();
    genres.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    if let Some((g, _)) = genres.first() {
        let (g, noun) = (*g, crate::row_label::genre_noun(g).unwrap_or(g));
        charts.push(RowOut { title: format!("Top 10 {noun} Today"), key: "row.np_top10_genre", args: json!({ "genre": noun }), row_type: "top_ten", kind: "chart", items: chart_where(catalog, fresh, maturity, profiles, today, year, |i| i.genres.iter().any(|x| x == g)) });
    }
    for (g, _) in pick_by_noise(genres, 5, 3, session, slot) {
        let v: Vec<&Cand> = pool.iter().copied().filter(|c| recent(c, 90) && c.genres.iter().any(|x| x == g)).collect();
        let noun = crate::row_label::genre_noun(g).unwrap_or(g);
        rows.push(RowOut { title: format!("New {noun}"), key: "row.np_new_genre", args: json!({ "genre": noun }), row_type: "vibe", kind: "genre", items: take_top(v, wide) });
    }

    // good titles of the last three years that are not yet everywhere: rated well by enough people, not among the most popular. Chosen by how well they are rated and how many rated them (not by
    // popularity, which would bring the same few every time), with a little noise per session and slot, at most three of a genre and films and series alternating
    let mut gems: Vec<(f32, &Cand)> = pool
        .iter()
        .copied()
        .filter(|c| c.in_catalog && !c.upcoming(today) && c.release_year.is_some_and(|y| y >= year - 2) && c.vote_average >= 7.2 && c.vote_count >= 80 && c.popularity < 60.0 && c.popularity >= 5.0)
        .map(|c| (c.vote_average + 0.35 * (c.vote_count as f32).ln() + 1.6 * crate::page::entry_noise(session, c.tmdb_id as usize + slot), c))
        .collect();
    gems.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.tmdb_id.cmp(&b.1.tmdb_id)));
    let mut per_genre: HashMap<&str, usize> = HashMap::default();
    let (mut films, mut series): (Vec<&Cand>, Vec<&Cand>) = (Vec::new(), Vec::new());
    for (_, c) in gems {
        let g = c.genres.first().map_or("", |g| g.as_str());
        let n = per_genre.entry(g).or_insert(0);
        if *n >= 3 { continue; }
        *n += 1;
        if c.media_type == "tv" { series.push(c) } else { films.push(c) }
    }
    let mut mixed: Vec<Cand> = Vec::new();
    let (mut fi, mut si) = (films.into_iter(), series.into_iter());
    while mixed.len() < wide {
        match (fi.next(), si.next()) {
            (None, None) => break,
            (f, t) => { mixed.extend(f.cloned()); mixed.extend(t.cloned()); }
        }
    }
    let (gem_title, gem_key) = [("Under the Radar", "row.np_under_the_radar"), ("Hidden Gems", "row.np_hidden_gems"), ("Worth a Closer Look", "row.np_closer_look")][slot % 3];
    rows.push(RowOut { title: gem_title.into(), key: gem_key, args: Value::Null, row_type: "vibe", kind: "gems", items: mixed });

    // the order of the other rows moves a little with the session and the slot
    let mut keyed: Vec<(f32, RowOut)> = rows.into_iter().enumerate().map(|(i, r)| (i as f32 + (crate::page::entry_noise(session, i + slot * 31) - 0.5) * 6.0, r)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ordered: Vec<RowOut> = keyed.into_iter().map(|(_, r)| r).collect();

    // between one and four charts (the nearest to the news first), spread out: the first after zero to two other rows, then every three to five rows, never next to each other
    charts.retain(|c| !c.items.is_empty());
    let want = 1 + (crate::page::entry_noise(session, 7 + slot) * 4.0) as usize;
    let mut chosen: Vec<(f32, RowOut)> = charts.into_iter().enumerate().map(|(i, c)| (i as f32 * 0.6 + crate::page::entry_noise(session, 100 + i + slot * 13) * 2.0, c)).collect();
    chosen.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let start = (crate::page::entry_noise(session, 11 + slot) * 3.0) as usize;
    let gap = 3 + (crate::page::entry_noise(session, 13 + slot) * 3.0) as usize;
    for (n, (_, chart_row)) in chosen.into_iter().take(want).enumerate() {
        let at = (start + n * gap).min(ordered.len());
        ordered.insert(at, chart_row);
    }
    ordered
}

/// `take` of the first `pool` candidates, chosen by their place plus noise from the session and the slot.
pub(super) fn pick_by_noise<T>(mut v: Vec<(T, usize)>, pool: usize, take: usize, session: &str, slot: usize) -> Vec<(T, usize)> {
    v.truncate(pool);
    let mut keyed: Vec<(f32, (T, usize))> = v.into_iter().enumerate().map(|(i, x)| (i as f32 + crate::page::entry_noise(session, i * 17 + slot) * 2.5, x)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().take(take).map(|(_, x)| x).collect()
}
