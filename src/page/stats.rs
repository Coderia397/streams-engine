//! Small pure helpers over rows and titles: leading genre, era, ordering by genre, stable noise per session.
use super::*;

/// Prior weight per row type, mirroring the page policy type mix in
/// ai-engine/config/row_architecture.json (vibe and theme rows dominate).
pub(crate) fn kind_weight(kind: &str) -> f32 {
    match kind {
        "vibe" => 1.0,
        "theme" => 0.85,
        "combo" => 0.7,
        "country" => 0.5,
        "setting" => 0.4,
        "animal" | "job" | "hobby" | "place" | "learning" => 0.6,
        "variant" => 0.5,
        "decade" | "character" | "audience" => 0.35,
        "animation" | "style" | "editorial" => 0.3,
        "identity" => 0.2,
        _ => 0.3,
    }
}

/// Stable value in (0, 1) per (session, entry): exploration without flicker.
pub(crate) fn entry_noise(session: &str, idx: usize) -> f32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    session.hash(&mut h);
    idx.hash(&mut h);
    ((h.finish() >> 40) as f32) / ((1u64 << 24) as f32)
}

/// The genre most of a row's titles carry (at least 40 percent of them), empty when no genre leads.
pub(crate) fn dominant_genre(row: &Row) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::default();
    for i in &row.items {
        for g in i.genres.iter().take(3) { *counts.entry(g.as_str()).or_insert(0) += 1; }
    }
    counts.into_iter().filter(|(_, n)| *n * 5 >= row.items.len() * 2).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0))).map(|(g, _)| g.to_string()).unwrap_or_default()
}

/// Reorders rows (looking ahead at most eight rows) so that a row's leading genre differs from that of the three rows before it whenever the following rows allow; the order is otherwise kept.
pub(crate) fn spread_by_genre(rows: Vec<Row>) -> Vec<Row> {
    let dom: Vec<String> = rows.iter().map(dominant_genre).collect();
    let mut slots: Vec<Option<Row>> = rows.into_iter().map(Some).collect();
    let mut left: Vec<usize> = (0..slots.len()).collect();
    let mut out: Vec<Row> = Vec::with_capacity(left.len());
    let mut recent: Vec<usize> = Vec::new();
    while !left.is_empty() {
        let pos = left.iter().take(8).position(|&i| dom[i].is_empty() || !recent.iter().rev().take(3).any(|&r| dom[r] == dom[i])).unwrap_or(0);
        let i = left.remove(pos);
        recent.push(i);
        out.extend(slots[i].take());
    }
    out
}

/// Second most common genre, if any.
pub(crate) fn second_genre<'a>(genres_iter: impl Iterator<Item = &'a [String]>, first: &str) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::default();
    for genres in genres_iter {
        for g in genres {
            if g != first && !g.contains(first) && !first.contains(g.as_str()) {
                *counts.entry(g.as_str()).or_insert(0) += 1;
            }
        }
    }
    counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0))).map(|(g, _)| g.to_string())
}

/// Decade label ("90s", "2010s") when at least half of the items share a decade.
pub(crate) fn dominant_era(items: &[RowItem]) -> Option<String> {
    let mut counts: HashMap<i32, usize> = HashMap::default();
    let mut total = 0usize;
    for it in items {
        if let Some(y) = it.release_year {
            *counts.entry(y / 10 * 10).or_insert(0) += 1;
            total += 1;
        }
    }
    let (decade, n) = counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))?;
    if total == 0 || n * 10 < total * 7 {
        return None;
    }
    Some(if decade >= 2000 { format!("{decade}s") } else { format!("{}s", decade % 100) })
}

/// Dominant genre from an iterator of genre slices (majority vote).
pub(crate) fn majority_genre<'a>(genres_iter: impl Iterator<Item = &'a [String]>) -> String {
    let mut counts: HashMap<String, usize> = HashMap::default();
    for genres in genres_iter {
        for genre in genres {
            *counts.entry(genre.clone()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(g, _)| g)
        .unwrap_or_else(|| "Drama".to_string())
}
