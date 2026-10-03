use crate::hnsw::is_ready;
use axum::{http::StatusCode, response::IntoResponse, Json};
use serde_json::json;

/// GET /api/health
/// Returns 200 Ready or 200 Degraded (brute-force mode while HNSW builds).
/// Distinguishes "broken" from "warming up" for monitoring systems.
pub async fn health() -> impl IntoResponse {
    let (mode, status) = if is_ready() {
        ("ready", StatusCode::OK)
    } else {
        ("degraded_brute_force", StatusCode::OK)
    };
    (status, Json(json!({ "status": mode, "engine": "pstream-engine" })))
}
