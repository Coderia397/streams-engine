//! GET /api/characters?rating=PG&limit=50: the Characters page of a children's profile (Netflix's Characters, note 97: one strip of franchise leads, round pictures that
//! open the franchise). One row of the children's franchises the profile may see, each franchise once (its titles grouped by the name before any colon, bracket, dash,
//! sequel number or "the movie"), the most watched first. The page is a child's: an adult caller sees the older kids tier.
use crate::app::AppState;
use crate::catalog::CatalogItem;
use crate::identity::{maturity_of, MaybeCaller, Maturity};
use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

const CHARACTERS: usize = 50;
/// Members listed with a franchise
const MEMBERS: usize = 8;

#[derive(Deserialize)]
pub struct CharactersQuery {
    pub rating: Option<String>,
    pub limit: Option<usize>,
}

/// The franchise a title belongs to by its name: the words before a colon, bracket or dash, without trailing sequel numbers, roman numerals, "part N", "the movie" and a
/// leading "the". "Despicable Me 2", "PAW Patrol: The Movie" and "Sonic the Hedgehog 3" fall in with their first titles.
pub fn franchise_key(title: &str) -> String {
    let t = title.to_lowercase();
    let t = t.split([':', '(', '\u{2013}', '\u{2014}']).next().unwrap_or("").trim();
    let t = t.split(" - ").next().unwrap_or(t).trim();
    let mut words: Vec<&str> = t.split_whitespace().collect();
    loop {
        let Some(last) = words.last().copied() else { break };
        let numeral = last.chars().all(|c| c.is_ascii_digit()) || matches!(last, "ii" | "iii" | "iv" | "v" | "vi" | "vii" | "viii");
        if numeral || matches!(last, "movie" | "film" | "returns" | "strikes") {
            words.pop();
            if words.last().is_some_and(|w| matches!(*w, "part" | "the" | "chapter" | "season")) {
                words.pop();
            }
            continue;
        }
        break;
    }
    if words.first() == Some(&"the") {
        words.remove(0);
    }
    words.join(" ")
}

/// Whether a title is a children's franchise candidate: made for children, or an animated family film (Despicable Me, Shrek, Ice Age have no Kids genre).
fn franchise_candidate(item: &CatalogItem) -> bool {
    crate::page::is_childrens_title(item) || (item.genres.iter().any(|g| g == "Family") && item.genres.iter().any(|g| g == "Animation"))
}

pub async fn characters(State(state): State<Arc<AppState>>, MaybeCaller(caller): MaybeCaller, Query(q): Query<CharactersQuery>) -> impl IntoResponse {
    let maturity = maturity_of(&caller).or_else(|| q.rating.as_deref().and_then(Maturity::from_rating)).filter(|m| m.is_kids()).unwrap_or(Maturity::OlderKids);
    let limit = q.limit.unwrap_or(CHARACTERS).clamp(1, 200);
    let catalog = state.catalog.load();
    let mut pool: Vec<&CatalogItem> = catalog
        .values()
        .filter(|i| franchise_candidate(i) && i.has_poster() && crate::page::maturity_allows_with(maturity, &state.profiles, i) && !i.title.is_empty())
        .collect();
    pool.sort_by(|a, b| b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
    // group by franchise in order of popularity: the lead is the most popular title of the franchise
    let mut order: Vec<String> = Vec::new();
    let mut groups: crate::det::HashMap<String, Vec<&CatalogItem>> = crate::det::HashMap::default();
    for it in pool {
        let key = franchise_key(&it.title);
        if key.is_empty() {
            continue;
        }
        let g = groups.entry(key.clone()).or_default();
        if g.is_empty() {
            order.push(key);
        }
        g.push(it);
    }
    let items: Vec<serde_json::Value> = order
        .iter()
        .filter_map(|key| {
            let g = groups.get(key)?;
            let lead = g.first()?;
            // a character is a series or a film franchise: a film that stands alone (Spirited Away) is a title, not a character
            if lead.media_type != "tv" && g.len() < 2 {
                return None;
            }
            // the franchise's name as the lead spells it: the words of the key with the lead's own capitals ("Despicable Me", not "Despicable Me 2")
            let words = key.split_whitespace().count();
            let cleaned = lead.title.split([':', '(', '\u{2013}', '\u{2014}']).next().unwrap_or(&lead.title).split(" - ").next().unwrap_or(&lead.title).trim();
            let own: Vec<&str> = cleaned.split_whitespace().collect();
            let skip = usize::from(own.first().is_some_and(|w| w.eq_ignore_ascii_case("the")));
            let name = if own.len() >= skip + words { own[..skip + words].join(" ") } else { cleaned.to_string() };
            Some(json!({
                "id": lead.tmdb_id, "media_type": lead.media_type, "title": name, "lead_title": lead.title, "poster_path": lead.poster_path, "backdrop_path": lead.backdrop_path,
                "titles": g.len(), "popularity": g.iter().map(|i| i.popularity).sum::<f32>(),
                "members": g.iter().take(MEMBERS).map(|i| json!({ "id": i.tmdb_id, "media_type": i.media_type, "title": i.title })).collect::<Vec<_>>(),
            }))
        })
        .take(limit)
        .collect();
    let rows = vec![json!({ "id": "characters", "row_type": "characters", "type": "characters", "title": "Characters", "title_key": "row.characters", "items": items })];
    Json(json!({ "surface": "characters", "maturity": maturity.as_str(), "total_rows": rows.len(), "rows": rows }))
}

#[cfg(test)]
mod tests {
    use super::franchise_key;

    #[test]
    fn sequels_and_subtitles_fall_in_with_their_franchise() {
        assert_eq!(franchise_key("Despicable Me 2"), "despicable me");
        assert_eq!(franchise_key("PAW Patrol: The Movie"), "paw patrol");
        assert_eq!(franchise_key("Sonic the Hedgehog 3"), "sonic the hedgehog");
        assert_eq!(franchise_key("The Amazing World of Gumball"), "amazing world of gumball");
        assert_eq!(franchise_key("Ice Age: Collision Course"), "ice age");
        assert_eq!(franchise_key("Shrek 2"), franchise_key("Shrek"));
        assert_eq!(franchise_key("How to Train Your Dragon: The Hidden World"), "how to train your dragon");
        assert_eq!(franchise_key("Kung Fu Panda 4"), "kung fu panda");
        assert_eq!(franchise_key("Hotel Transylvania 3: Summer Vacation"), "hotel transylvania");
    }
}
