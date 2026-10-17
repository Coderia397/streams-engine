//! The door of the engine: who may call it and how often.
//!
//! * Account: when the identity layer is on (`SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` are set, as on the live Space), every route except `/api/health` and `/api/features` needs a
//!   verified account token (`Authorization: Bearer ...`). A call without one gets 401 `account required`; a token that does not verify gets 401 from the identity layer. With the layer off
//!   (local development) the engine stays open. `ENGINE_OPEN=1` opens it on purpose.
//! * Origin: browsers may call it only from the site's own addresses (`ENGINE_ALLOWED_ORIGINS`, comma separated; the defaults are streams.watch, www.streams.watch, the Pages address,
//!   localhost and private network addresses for development). A call that is not from a browser has no origin and is judged by the account rule alone.
//! * Rate: a caller (by token, else by forwarded address) may make `ENGINE_RATE_PER_MINUTE` calls a minute (default 600); more get 429.

use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use moka::sync::Cache;
use serde_json::json;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::app::AppState;

const OPEN_ROUTES: [&str; 2] = ["/api/health", "/api/features"];

fn counters() -> &'static Cache<String, u32> {
    static C: OnceLock<Cache<String, u32>> = OnceLock::new();
    C.get_or_init(|| Cache::builder().time_to_live(Duration::from_secs(60)).max_capacity(100_000).build())
}

fn per_minute() -> u32 {
    std::env::var("ENGINE_RATE_PER_MINUTE").ok().and_then(|v| v.parse().ok()).unwrap_or(600)
}

fn is_open() -> bool {
    std::env::var("ENGINE_OPEN").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
}

fn caller_key(req: &Request) -> String {
    let h = req.headers();
    if let Some(a) = h.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        // the end of a token differs for every login; enough to tell callers apart without keeping the token
        return format!("t:{}", &a[a.len().saturating_sub(24)..]);
    }
    let ip = h.get("x-forwarded-for").and_then(|v| v.to_str().ok()).and_then(|v| v.split(',').next()).unwrap_or("-").trim();
    format!("ip:{ip}")
}

pub async fn guard(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    if req.method() == Method::OPTIONS || OPEN_ROUTES.contains(&path.as_str()) {
        return next.run(req).await;
    }
    // the rate first, so a flood of bad calls costs little
    let key = caller_key(&req);
    let n = counters().get(&key).unwrap_or(0) + 1;
    counters().insert(key, n);
    if n > per_minute() {
        return (StatusCode::TOO_MANY_REQUESTS, [(header::RETRY_AFTER, "30")], Json(json!({ "error": "too many requests" }))).into_response();
    }
    if let (Some(identity), false) = (&state.identity, is_open()) {
        match identity.caller(req.headers()).await {
            Ok(Some(_)) => {}
            Ok(None) => return (StatusCode::UNAUTHORIZED, Json(json!({ "error": "account required" }))).into_response(),
            Err(e) => return (e.status(), Json(json!({ "error": e.message() }))).into_response(),
        }
    }
    next.run(req).await
}

fn origin_allowed(origin: &str) -> bool {
    let configured = std::env::var("ENGINE_ALLOWED_ORIGINS").unwrap_or_default();
    let list: Vec<String> = if configured.trim().is_empty() {
        vec!["https://streams.watch".into(), "https://www.streams.watch".into(), "https://streams-app.pages.dev".into(), "https://help.streams.watch".into()]
    } else {
        configured.split(',').map(|s| s.trim().to_string()).collect()
    };
    if list.iter().any(|o| o == origin) { return true; }
    // development: this computer and private network addresses (a phone on the same Wi-Fi)
    let host = origin.split("://").nth(1).unwrap_or("").split(':').next().unwrap_or("");
    host == "localhost" || host == "127.0.0.1" || host.starts_with("192.168.") || host.starts_with("10.") || (host.starts_with("172.") && host.split('.').nth(1).and_then(|s| s.parse::<u8>().ok()).is_some_and(|n| (16..=31).contains(&n)))
}

pub fn cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin: &HeaderValue, _| origin.to_str().is_ok_and(origin_allowed)))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE, header::HeaderName::from_static("x-profile-id")])
        .max_age(Duration::from_secs(3600))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_site_and_development_origins_are_allowed_and_others_are_not() {
        assert!(origin_allowed("https://streams.watch"));
        assert!(origin_allowed("http://localhost:8080"));
        assert!(origin_allowed("http://172.24.116.235:8080"));
        assert!(!origin_allowed("https://example.com"));
        assert!(!origin_allowed("http://172.40.0.1:8080"));
    }
}
