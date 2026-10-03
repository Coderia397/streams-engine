use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

use crate::app::AppState;
use crate::sessions::FeedbackEvent;

#[derive(Deserialize)]
pub struct FeedbackBody {
    pub session_id: String,
    pub item_id: u32,
    pub event: FeedbackEvent,
}

/// POST /api/feedback  { session_id, item_id, event: click|start|complete|like|skip }
/// Updates the session's decayed genre taste and seen set. Later /api/feed and
/// /api/hero-feed calls with the same session_id reflect it.
pub async fn feedback(
    State(state): State<Arc<AppState>>,
    Json(body): Json<FeedbackBody>,
) -> impl IntoResponse {
    if body.session_id.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "session_id required"})));
    }
    let catalog = state.catalog.load();
    let Some(item) = catalog.get(&body.item_id) else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "item not found", "id": body.item_id})));
    };

    let session = state.sessions.get_or_create(&body.session_id);
    let mut s = session.lock().unwrap();
    s.record(item, body.event);
    (StatusCode::OK, Json(json!({ "ok": true, "taste": s.taste(), "version": s.version })))
}
