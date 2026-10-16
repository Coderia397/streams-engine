//! Facts the engine attaches to every title of a feed response so the client only draws them: up to three tags (tone words of the title's profile, then its genres) and badges
//! (Top 10 rank, New). Nothing here is written by hand per title; tags come from the profile tone words and the genre list, badges from the chart position and the release date.
use crate::profiles::TitleProfile;

/// Up to `max` tags: tone words of the profile (as display names, no repeats), then genres until the list is full. A title without a profile gets genres only.
pub fn item_tags(profile: Option<&TitleProfile>, genres: &[String], max: usize) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    if let Some(p) = profile {
        for w in &p.tone {
            if tags.len() >= max.saturating_sub(1) { break; }
            if let Some(label) = crate::profiles::tone_label(w) {
                if !tags.iter().any(|t| t == label) { tags.push(label.to_string()); }
            }
        }
    }
    for g in genres {
        if tags.len() >= max { break; }
        if !tags.iter().any(|t| t.eq_ignore_ascii_case(g)) { tags.push(g.clone()); }
    }
    tags
}

/// A title is New for this many days after its release date.
pub const NEW_DAYS: i64 = 45;

/// Badges of a title as `{ "kind": ..., ... }` values: `top10` with the rank when it sits in a chart row, `new` when it came out within `NEW_DAYS` of today.
pub fn item_badges(chart_rank: Option<usize>, release_date: Option<&str>, today_days: i64) -> Vec<serde_json::Value> {
    let mut badges = Vec::new();
    if let Some(rank) = chart_rank {
        badges.push(serde_json::json!({ "kind": "top10", "rank": rank }));
    }
    if let Some(days) = release_date.and_then(crate::fresh::parse_days) {
        if (0..=NEW_DAYS).contains(&(today_days - days)) {
            badges.push(serde_json::json!({ "kind": "new" }));
        }
    }
    badges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_put_tone_before_genres_and_never_repeat() {
        let genres = vec!["Crime".to_string(), "Drama".to_string(), "Thriller".to_string()];
        assert_eq!(item_tags(None, &genres, 3), vec!["Crime", "Drama", "Thriller"]);
        let p = crate::profiles::parse_profile(&serde_json::json!({"audience": "adult", "tone": ["gritty", "dark", "gritty"]})).unwrap();
        assert_eq!(item_tags(Some(&p), &genres, 3), vec!["Gritty", "Dark", "Crime"]);
        assert!(item_tags(None, &[], 3).is_empty());
    }

    #[test]
    fn badges_mark_chart_rank_and_recent_releases() {
        let today = crate::fresh::parse_days("2026-10-16").unwrap();
        assert!(item_badges(None, Some("2026-01-01"), today).is_empty());
        let b = item_badges(Some(3), Some("2026-10-01"), today);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0]["rank"], 3);
        assert_eq!(b[1]["kind"], "new");
        // not yet released is not New
        assert!(item_badges(None, Some("2026-12-01"), today).is_empty());
    }
}
