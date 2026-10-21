use crate::app::AppState;
use crate::hnsw::is_ready;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;
use std::sync::Arc;

/// GET /api/health
/// Returns 200 Ready or 200 Degraded (brute-force mode while HNSW builds).
/// Distinguishes "broken" from "warming up" for monitoring systems. Also says whether the catalog and the vectors describe the same titles (see status.rs).
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (mode, status) = if is_ready() {
        ("ready", StatusCode::OK)
    } else {
        ("degraded_brute_force", StatusCode::OK)
    };
    // date of the popularity overlay in use (yyyymmdd), so a check can warn when the daily refresh stops
    let asof = crate::catalog::POPULARITY_ASOF.load(std::sync::atomic::Ordering::Relaxed);
    let asof = if asof == 0 { serde_json::Value::Null } else { json!(asof) };
    let data = crate::status::Coverage::of(&state.catalog.load(), &state.vectors).to_json();
    (status, Json(json!({ "status": mode, "engine": "streams-engine", "popularity_asof": asof, "data": data })))
}
