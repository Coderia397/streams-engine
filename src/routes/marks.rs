use crate::app::AppState;
use axum::{extract::{Query, State}, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::HashMap, sync::{Arc, OnceLock}};

/// Skip marks per title, as ratios of the video's length (note 71): `skip_marks.jsonl` next to the database, one line per show:
/// `{"k":"tv:1396","typical":{"intro":[0.05,0.07],"recap":null,"credits":0.97},"premiere":{...},"finale":{...}}`. Read once, on the first request.
static MARKS: OnceLock<HashMap<String, Value>> = OnceLock::new();

fn load(db_path: &str) -> HashMap<String, Value> {
    let Some(dir) = std::path::Path::new(db_path).parent() else { return HashMap::default() };
    let Ok(text) = std::fs::read_to_string(dir.join("skip_marks.jsonl")) else { return HashMap::default() };
    text.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter_map(|v| v.get("k").and_then(|k| k.as_str()).map(|k| (k.to_string(), v.clone()))).collect()
}

/// The marks of one video: the first episode of a season takes `premiere`, the last `finale`, every other one `typical` (each falls back to `typical`); a film or an unknown title has none.
pub fn pick(entry: Option<&Value>, episode: Option<u32>, last: bool) -> Value {
    let none = json!({ "intro": null, "recap": null, "credits": null });
    let Some(e) = entry else { return none };
    let kind = if episode == Some(1) { "premiere" } else if last { "finale" } else { "typical" };
    let m = e.get(kind).filter(|v| v.is_object()).or_else(|| e.get("typical")).filter(|v| v.is_object());
    match m {
        Some(m) => json!({ "intro": m.get("intro").cloned().unwrap_or(Value::Null), "recap": m.get("recap").cloned().unwrap_or(Value::Null), "credits": m.get("credits").cloned().unwrap_or(Value::Null) }),
        None => none,
    }
}

#[derive(Deserialize)]
pub struct MarksQuery {
    /// movie or tv
    #[serde(rename = "type")]
    pub kind: String,
    pub id: u32,
    pub episode: Option<u32>,
    /// 1 when the episode is the last of its season (the page knows from the season's episode list)
    pub last: Option<u8>,
}

/// GET /api/skip-marks?type=tv&id=1396&episode=3[&last=1]: `{ "intro": [start, end] | null, "recap": [start, end] | null, "credits": start | null }`, all as ratios of the video's length.
pub async fn skip_marks(State(state): State<Arc<AppState>>, Query(q): Query<MarksQuery>) -> impl IntoResponse {
    let map = MARKS.get_or_init(|| load(&state.config.db_path));
    let key = format!("{}:{}", if q.kind == "tv" || q.kind == "series" { "tv" } else { "movie" }, q.id);
    Json(pick(map.get(&key), q.episode, q.last == Some(1)))
}

/// Content reasons per title (note 69), from `content_reasons.jsonl` next to the database (made by pipeline/reasons_apply.py): a film `{"k","r"}`, a show `{"k","d","e"}` where `d` is the answer
/// for every episode without its own summary and `e` holds the episodes whose answer differs (key `season:episode`).
static REASONS: OnceLock<HashMap<String, Value>> = OnceLock::new();

fn load_reasons(db_path: &str) -> HashMap<String, Value> {
    let Some(dir) = std::path::Path::new(db_path).parent() else { return HashMap::default() };
    let Ok(text) = std::fs::read_to_string(dir.join("content_reasons.jsonl")) else { return HashMap::default() };
    text.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter_map(|v| v.get("k").and_then(|k| k.as_str()).map(|k| (k.to_string(), v.clone()))).collect()
}

/// The reasons of one video, or null when there is no answer for it (an unknown title, or an unreadable call): the page then shows the rating alone.
pub fn pick_reasons(entry: Option<&Value>, season: Option<u32>, episode: Option<u32>) -> Value {
    let Some(e) = entry else { return Value::Null };
    if let Some(r) = e.get("r") { return r.clone(); }
    let key = format!("{}:{}", season.unwrap_or(1), episode.unwrap_or(1));
    e.get("e").and_then(|m| m.get(&key)).or_else(|| e.get("d")).cloned().unwrap_or(Value::Null)
}

#[derive(Deserialize)]
pub struct ReasonsQuery {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: u32,
    pub season: Option<u32>,
    pub episode: Option<u32>,
}

/// GET /api/content-reasons?type=tv&id=1396&season=1&episode=3: `{ "reasons": ["violence", "strong language"] | null }`, words from the fixed list of 20.
pub async fn content_reasons(State(state): State<Arc<AppState>>, Query(q): Query<ReasonsQuery>) -> impl IntoResponse {
    let map = REASONS.get_or_init(|| load_reasons(&state.config.db_path));
    let key = format!("{}:{}", if q.kind == "tv" || q.kind == "series" { "tv" } else { "movie" }, q.id);
    Json(json!({ "reasons": pick_reasons(map.get(&key), q.season, q.episode) }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> Value {
        json!({ "k": "tv:1", "typical": { "intro": [0.05, 0.07], "recap": null, "credits": 0.97 }, "premiere": { "intro": [0.1, 0.12], "recap": null, "credits": 0.95 }, "finale": null })
    }

    #[test]
    fn premiere_finale_and_typical() {
        let e = entry();
        assert_eq!(pick(Some(&e), Some(1), false)["credits"], json!(0.95));
        assert_eq!(pick(Some(&e), Some(5), false)["credits"], json!(0.97));
        // a finale without its own entry falls back to the typical one
        assert_eq!(pick(Some(&e), Some(9), true)["credits"], json!(0.97));
    }

    #[test]
    fn reasons_default_override_and_film() {
        let show = json!({ "k": "tv:1", "d": ["violence"], "e": { "1:3": ["sex", "nudity"] } });
        assert_eq!(pick_reasons(Some(&show), Some(1), Some(3)), json!(["sex", "nudity"]));
        assert_eq!(pick_reasons(Some(&show), Some(2), Some(1)), json!(["violence"]));
        assert_eq!(pick_reasons(Some(&json!({ "k": "movie:1", "r": ["war"] })), None, None), json!(["war"]));
        assert!(pick_reasons(None, Some(1), Some(1)).is_null());
    }

    #[test]
    fn unknown_title_has_no_marks() {
        let m = pick(None, Some(2), false);
        assert!(m["intro"].is_null() && m["recap"].is_null() && m["credits"].is_null());
    }
}
