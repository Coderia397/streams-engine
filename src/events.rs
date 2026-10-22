//! The calendar events that have a page of their own (beyond Netflix, note 98 W10): `events_v1.json` next to the database, written by
//! ai-engine/pipeline/make_requests_v2.py from the calendar vocabulary. An event is its label, its window (MM-DD, may wrap the year), the keywords a title
//! needs to belong to it and the genres its rows are composed with. The page of an event is offered in its window only (`genre_pages::EVENT_PAGES` gives the ids).
use crate::taxonomy::Season;
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct EventDef {
    pub id: String,
    pub label: String,
    pub season: Season,
    pub keywords: Vec<String>,
    pub genres: Vec<String>,
}

#[derive(Deserialize)]
struct RawSeason {
    start: String,
    end: String,
}

#[derive(Deserialize)]
struct RawEvent {
    label: String,
    season: RawSeason,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    genres: Vec<String>,
}

#[derive(Deserialize)]
struct RawFile {
    events: std::collections::BTreeMap<String, RawEvent>,
}

static EVENTS: OnceLock<Vec<EventDef>> = OnceLock::new();

fn parse_md(s: &str) -> Option<(u8, u8)> {
    let (m, d) = s.split_once('-')?;
    Some((m.trim().parse().ok()?, d.trim().parse().ok()?))
}

/// Reads `events_v1.json` next to `db_path` once; a missing or unreadable file leaves no events (no event page is then offered).
pub fn load(db_path: &str) {
    let list = std::path::Path::new(db_path)
        .parent()
        .and_then(|d| std::fs::read_to_string(d.join("events_v1.json")).ok())
        .and_then(|t| serde_json::from_str::<RawFile>(&t).map_err(|e| tracing::warn!("events_v1.json is not readable: {e}")).ok())
        .map(|f| {
            f.events
                .into_iter()
                .filter_map(|(id, e)| {
                    let season = Season { start: parse_md(&e.season.start)?, end: parse_md(&e.season.end)? };
                    Some(EventDef { id, label: e.label, season, keywords: e.keywords.into_iter().map(|k| k.to_lowercase()).collect(), genres: e.genres })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    tracing::info!("Events: {} with a page", list.len());
    let _ = EVENTS.set(list);
}

pub fn all() -> &'static [EventDef] {
    EVENTS.get().map_or(&[], |v| v.as_slice())
}

pub fn get(id: &str) -> Option<&'static EventDef> {
    all().iter().find(|e| e.id == id)
}

/// The event whose label is `name` (case does not matter; "and" and "&" alike).
pub fn by_label(name: &str) -> Option<&'static EventDef> {
    let key = name.trim().to_lowercase().replace(" and ", " & ");
    all().iter().find(|e| e.label.to_lowercase() == key)
}

/// Whether the event's window holds today (`STREAMS_TODAY` pretends another day).
pub fn in_season(id: &str) -> bool {
    get(id).is_some_and(|e| e.season.active_on(crate::taxonomy::today_md()))
}

/// Whether a title belongs to the event: it carries one of the event's keywords (TMDB keywords and Netflix tags alike).
pub fn title_belongs(id: &str, item: &crate::catalog::CatalogItem) -> bool {
    get(id).is_some_and(|e| item.keywords.iter().any(|k| e.keywords.iter().any(|w| w == k)))
}

#[cfg(test)]
pub(crate) fn set_for_tests(list: Vec<EventDef>) {
    let _ = EVENTS.set(list);
}
