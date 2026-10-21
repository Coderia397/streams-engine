//! The verified caller of a request: who asks, which profile, which age tier, and the extractor the routes use.
use super::Maturity;
use crate::app::AppState;
use axum::{extract::FromRequestParts, http::{request::Parts, StatusCode}, Json};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Caller {
    pub user_id: String,
    pub profile_id: Option<String>,
    pub maturity: Maturity,
}

impl Caller {
    /// The id the engine keeps taste and seen titles under: the verified profile, else the verified user.
    pub fn session_key(&self) -> String {
        self.profile_id.clone().unwrap_or_else(|| self.user_id.clone())
    }
}

/// The maturity of the caller, or none for an anonymous request.
pub fn maturity_of(caller: &Option<Caller>) -> Option<Maturity> {
    caller.as_ref().map(|c| c.maturity)
}

/// `MaybeCaller(None)` for anonymous requests (no token, or the identity layer is off).
pub struct MaybeCaller(pub Option<Caller>);

impl FromRequestParts<Arc<AppState>> for MaybeCaller {
    type Rejection = (StatusCode, Json<Value>);

    async fn from_request_parts(parts: &mut Parts, state: &Arc<AppState>) -> Result<Self, Self::Rejection> {
        let Some(identity) = &state.identity else {
            return Ok(Self(None));
        };
        match identity.caller(&parts.headers).await {
            Ok(c) => Ok(Self(c)),
            Err(e) => Err((e.status(), Json(json!({ "error": e.message() })))),
        }
    }
}
