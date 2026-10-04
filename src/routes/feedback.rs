use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

use crate::app::AppState;
use crate::identity::MaybeCaller;
use crate::sessions::FeedbackEvent;

#[derive(Deserialize)]
pub struct FeedbackBody {
    pub session_id: String,
    pub item_id: u32,
    /// "movie" or "tv". TMDB ids repeat across the two; without it the movie is tried first.
    pub media_type: Option<String>,
    pub event: FeedbackEvent,
}

/// POST /api/feedback  { session_id, item_id, media_type?, event: click|start|complete|like|skip }
/// Updates the session's decayed genre taste and seen set. Later /api/feed and
/// /api/hero-feed calls with the same session_id reflect it.
pub async fn feedback(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Json(body): Json<FeedbackBody>,
) -> impl IntoResponse {
    // A verified caller always writes to its own profile (or user) session, never to the id in the body
    let session_id = caller.as_ref().map(|c| c.session_key()).unwrap_or_else(|| body.session_id.clone());
    if session_id.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "session_id required"})));
    }
    let catalog = state.catalog.load();
    let Some(item) = crate::catalog::resolve_key(&catalog, body.item_id, body.media_type.as_deref()).and_then(|k| catalog.get(&k)) else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "item not found", "id": body.item_id})));
    };

    let session = state.sessions.get_or_create(&session_id);
    let mut s = session.lock().unwrap();
    s.record(item, body.event);
    (StatusCode::OK, Json(json!({ "ok": true, "taste": s.taste(), "version": s.version })))
}
