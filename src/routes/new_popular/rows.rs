//! The rows of the desktop page (Netflix's New & Popular structure) and the shape of a title in them.
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

/// New on Streams: titles released within this many days.
const NEW_DAYS: i64 = 60;

/// The rows of the desktop page, in Netflix's order (note 97, captured 2026-10-21). An adult profile: New on Streams, Top 10 Films Today, Top 10 Series Today, Coming This Week,
/// Coming Next Week, Worth the Wait. A Kids profile: New on Streams, Top 10 Kids Series Today, Coming This Week, Worth the Wait, Top 10 Kids Films Today, Coming Next Week. An
/// adult's page carries no children's titles. The rows are the same for every session.
#[allow(clippy::too_many_arguments)]
pub(super) fn desktop_rows(catalog: &CatalogMap, fresh: &FreshStore, pool: &[&Cand], maturity: Option<Maturity>, profiles: &crate::profiles::ProfileStore, today: &str, today_n: i64, year: i32, limit: usize, _session: &str, _slot: usize) -> Vec<RowOut> {
    let kids = maturity.is_some_and(|m| m.is_kids());
    let pool: Vec<&Cand> = pool.iter().copied().filter(|c| kids || !c.for_children).collect();
    let wide = limit * 4;
    let pop = |a: &Cand, b: &Cand| b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal);
    let days_until = |c: &Cand| c.days_since(today_n).map(|d| -d);

    // the most popular titles of the last NEW_DAYS days, newest first
    let mut new_on: Vec<Cand> = take_top(pool.iter().copied().filter(|c| has_backdrop(c) && c.released(today) && c.days_since(today_n).is_some_and(|d| d <= NEW_DAYS)).collect(), limit);
    new_on.sort_by(|a, b| b.release_date.cmp(&a.release_date).then_with(|| pop(a, b)));

    // the titles due in a window of days: the most anticipated of the window, soonest first
    let soon = |lo: i64, hi: i64| -> Vec<Cand> {
        let mut v: Vec<Cand> = take_top(pool.iter().copied().filter(|c| c.upcoming(today) && days_until(c).is_some_and(|d| d >= lo && d <= hi)).collect(), wide);
        v.sort_by(|a, b| a.release_date.cmp(&b.release_date).then_with(|| pop(a, b)));
        v
    };
    let this_week = soon(0, 7);
    let next_week = soon(8, 14);
    // further out: the most anticipated first
    let wait: Vec<Cand> = take_top(pool.iter().copied().filter(|c| c.upcoming(today) && days_until(c).is_some_and(|d| d > 14)).collect(), limit);

    let top_films = chart(catalog, fresh, maturity, profiles, "movie", today, year);
    let top_series = chart(catalog, fresh, maturity, profiles, "tv", today, year);
    let row = |title: &str, key: &'static str, row_type: &'static str, kind: &'static str, items: Vec<Cand>| RowOut { title: title.into(), key, args: Value::Null, row_type, kind, items };
    if kids {
        vec![
            row("New on Streams", "row.np_new_on_streams", "vibe", "new", new_on),
            row("Top 10 Kids Series Today", "row.np_top10_kids_series", "top_ten", "chart", top_series),
            row("Coming This Week", "row.np_coming_this_week", "vibe", "soon", this_week),
            row("Worth the Wait", "row.np_worth_the_wait", "vibe", "soon", wait),
            row("Top 10 Kids Films Today", "row.np_top10_kids_films", "top_ten", "chart", top_films),
            row("Coming Next Week", "row.np_coming_next_week", "vibe", "soon", next_week),
        ]
    } else {
        vec![
            row("New on Streams", "row.np_new_on_streams", "vibe", "new", new_on),
            row("Top 10 Films Today", "row.np_top10_movies", "top_ten", "chart", top_films),
            row("Top 10 Series Today", "row.np_top10_series", "top_ten", "chart", top_series),
            row("Coming This Week", "row.np_coming_this_week", "vibe", "soon", this_week),
            row("Coming Next Week", "row.np_coming_next_week", "vibe", "soon", next_week),
            row("Worth the Wait", "row.np_worth_the_wait", "vibe", "soon", wait),
        ]
    }
}
