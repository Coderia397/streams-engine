//! Verified caller identity for the engine routes, and the maturity policy of the active profile.
//!
//! A request may carry `Authorization: Bearer <Supabase access token>` and `X-Profile-Id: <profile uuid>`.
//!  * The token is verified here, locally, against the project's JWKS (ES256 signature, expiry, audience `authenticated`,
//!    issuer `<SUPABASE_URL>/auth/v1`). No call to Supabase is needed per request.
//!  * The profile is looked up through PostgREST with the caller's own token, so row level security proves the profile
//!    belongs to the caller (an empty answer means it does not). The answer (kind and maturity) is cached for a minute.
//!  * The engine session id becomes the verified profile id (or the user id when no profile is sent). The client can no
//!    longer read or write another person's taste by choosing their session id.
//!  * The profile's maturity filters every title the engine returns (see `Maturity::allows`).
//!
//! Without `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` the identity layer is off and every request is anonymous, as before.
//! A request without a token stays anonymous (full catalogue, session id from the query); a request with a token that does
//! not verify is refused with 401 so the client refreshes it. The maturity filter protects against client bugs and tampering
//! of the profile id; it cannot stop a person who calls the engine without a token, which is what the profile PIN is for.

use crate::catalog::{CatalogItem, FLAG_SUGGESTIVE};
use axum::{
    extract::FromRequestParts,
    http::{request::Parts, HeaderMap, StatusCode},
    Json,
};
use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use moka::future::Cache;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::app::AppState;

// ── Maturity policy ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Maturity {
    LittleKids,
    OlderKids,
    Twelve,
    Fifteen,
    Adult,
}

/// Genres that keep a title out of both kids tiers (TMDB movie and TV names).
const KIDS_BLOCKED_GENRES: [&str; 11] = [
    "Horror", "Crime", "Thriller", "Mystery", "War", "War & Politics", "Reality", "Talk", "News", "Soap", "Western",
];
/// Keywords that mark violent or sexual content. The catalogue has no certifications, so this is a conservative stand in;
/// it only applies to titles that carry keywords (pulled titles).
const KIDS_BLOCKED_KEYWORDS: [&str; 14] = [
    "gore", "violence", "blood", "murder", "serial killer", "torture", "rape", "sex", "nudity", "drugs", "suicide", "zombie",
    "slasher", "erotic",
];

impl Maturity {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "little_kids" => Some(Self::LittleKids),
            "older_kids" => Some(Self::OlderKids),
            "teens" | "twelve" => Some(Self::Twelve),
            "fifteen" => Some(Self::Fifteen),
            "adult" => Some(Self::Adult),
            _ => None,
        }
    }

    /// From a British age rating as the profile setting saves it: U, PG, 12, 15 or 18.
    pub fn from_rating(s: &str) -> Option<Self> {
        match s.trim().to_ascii_uppercase().as_str() {
            "U" => Some(Self::LittleKids),
            "PG" => Some(Self::OlderKids),
            "12" | "12A" => Some(Self::Twelve),
            "15" => Some(Self::Fifteen),
            "18" => Some(Self::Adult),
            _ => None,
        }
    }

    /// The highest certificate rank this tier may see (1 U, 2 PG, 3 12, 4 15, 5 18).
    pub fn max_cert(self) -> u8 {
        match self {
            Self::LittleKids => 1,
            Self::OlderKids => 2,
            Self::Twelve => 3,
            Self::Fifteen => 4,
            Self::Adult => 5,
        }
    }

    /// Kids tiers (U and PG) are the ones for children's profiles.
    pub fn is_kids(self) -> bool {
        matches!(self, Self::LittleKids | Self::OlderKids)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LittleKids => "little_kids",
            Self::OlderKids => "older_kids",
            Self::Twelve => "twelve",
            Self::Fifteen => "fifteen",
            Self::Adult => "adult",
        }
    }

    /// May a profile with this maturity see this title?
    ///  * adult: everything (the page builders' own safety nets still apply)
    ///  * teens: no adult flagged and no suggestive titles
    ///  * older kids: a known rating up to PG, no blocked genre or keyword, and the title must be kids, family or animation (never unrated)
    ///  * little kids: a known rating of U, as older kids, but animation alone is not enough (it must be kids or family)
    pub fn allows(self, item: &CatalogItem) -> bool {
        if self == Self::Adult {
            return true;
        }
        if item.adult || item.flags & FLAG_SUGGESTIVE != 0 {
            return false;
        }
        if self.is_kids() {
            // a title for a Kids profile needs a known rating within the tier AND to be made for children: a PG murder mystery or a rated sitcom is not (the rating alone
            // never decides for Kids), and an unrated title is never shown
            if item.cert == 0 || item.cert > self.max_cert() {
                return false;
            }
            let has = |name: &str| item.genres.iter().any(|g| g.eq_ignore_ascii_case(name));
            if KIDS_BLOCKED_GENRES.iter().any(|g| has(g)) {
                return false;
            }
            if item.keywords.iter().any(|k| KIDS_BLOCKED_KEYWORDS.iter().any(|b| k.contains(b))) {
                return false;
            }
            let kids_or_family = has("Kids") || has("Family");
            return match self {
                Self::LittleKids => kids_or_family,
                _ => kids_or_family || has("Animation"),
            };
        }
        // a known age rating decides (12 or 15 tiers); an unrated title is allowed there
        if item.cert != 0 {
            return item.cert <= self.max_cert();
        }
        true
    }
}

// ── Caller ───────────────────────────────────────────────────────────────────

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

#[derive(Debug)]
pub enum IdentityError {
    /// Token missing its parts, expired, wrong signature, audience or issuer
    Unauthorized(String),
    /// The profile does not belong to the token's user
    Forbidden,
    /// Supabase unreachable (keys or profile lookup)
    Unavailable(String),
}

impl IdentityError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
    pub fn message(&self) -> String {
        match self {
            Self::Unauthorized(m) => format!("invalid token: {m}"),
            Self::Forbidden => "profile not found for this account".to_string(),
            Self::Unavailable(m) => format!("identity service unavailable: {m}"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Claims {
    sub: String,
    #[allow(dead_code)]
    exp: u64,
}

#[derive(Debug, Clone, Copy)]
struct ProfileInfo {
    maturity: Maturity,
}

// ── Identity service ─────────────────────────────────────────────────────────

pub struct Identity {
    supabase_url: String,
    publishable_key: String,
    http: reqwest::Client,
    keys: RwLock<HashMap<String, DecodingKey>>,
    last_fetch: Mutex<Option<Instant>>,
    profiles: Cache<String, ProfileInfo>,
}

/// Do not refetch the key set more often than this, whatever the tokens say.
const JWKS_MIN_INTERVAL: Duration = Duration::from_secs(30);
/// Refetch the key set at least this often so rotated keys are picked up.
const JWKS_MAX_AGE: Duration = Duration::from_secs(3600);

impl Identity {
    /// Built from `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`; none when either is missing.
    pub fn from_env() -> Option<Arc<Self>> {
        let url = std::env::var("SUPABASE_URL").ok().filter(|s| !s.trim().is_empty())?;
        let key = std::env::var("SUPABASE_PUBLISHABLE_KEY").ok().filter(|s| !s.trim().is_empty())?;
        Some(Arc::new(Self::new(&url, &key)))
    }

    pub fn new(supabase_url: &str, publishable_key: &str) -> Self {
        Self {
            supabase_url: supabase_url.trim_end_matches('/').to_string(),
            publishable_key: publishable_key.to_string(),
            http: reqwest::Client::builder().timeout(Duration::from_secs(5)).build().expect("http client"),
            keys: RwLock::new(HashMap::new()),
            last_fetch: Mutex::new(None),
            profiles: Cache::builder().max_capacity(10_000).time_to_live(Duration::from_secs(60)).build(),
        }
    }

    fn issuer(&self) -> String {
        format!("{}/auth/v1", self.supabase_url)
    }

    /// Install a key set (JWKS JSON). Returns how many ES256 keys it held.
    pub fn set_jwks(&self, jwks: &Value) -> usize {
        let parsed = parse_jwks(jwks);
        let n = parsed.len();
        *self.keys.write().unwrap() = parsed;
        *self.last_fetch.lock().unwrap() = Some(Instant::now());
        n
    }

    async fn refresh_keys(&self) -> Result<(), IdentityError> {
        {
            let last = self.last_fetch.lock().unwrap();
            if last.is_some_and(|t| t.elapsed() < JWKS_MIN_INTERVAL) {
                return Ok(());
            }
        }
        let url = format!("{}/auth/v1/.well-known/jwks.json", self.supabase_url);
        let jwks: Value = self
            .http
            .get(&url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| IdentityError::Unavailable(e.to_string()))?
            .json()
            .await
            .map_err(|e| IdentityError::Unavailable(e.to_string()))?;
        self.set_jwks(&jwks);
        Ok(())
    }

    fn keys_stale(&self) -> bool {
        self.last_fetch.lock().unwrap().is_none_or(|t| t.elapsed() > JWKS_MAX_AGE)
    }

    /// Verify the token and return the user id (`sub`).
    pub async fn verify(&self, token: &str) -> Result<String, IdentityError> {
        let header = decode_header(token).map_err(|e| IdentityError::Unauthorized(e.to_string()))?;
        if header.alg != Algorithm::ES256 {
            return Err(IdentityError::Unauthorized("unsupported algorithm".into()));
        }
        let kid = header.kid.ok_or_else(|| IdentityError::Unauthorized("no key id".into()))?;

        if self.keys_stale() || !self.keys.read().unwrap().contains_key(&kid) {
            self.refresh_keys().await?;
        }
        let key = self
            .keys
            .read()
            .unwrap()
            .get(&kid)
            .cloned()
            .ok_or_else(|| IdentityError::Unauthorized("unknown signing key".into()))?;

        let mut validation = Validation::new(Algorithm::ES256);
        validation.set_audience(&["authenticated"]);
        validation.set_issuer(&[self.issuer()]);
        validation.leeway = 30;
        let data = decode::<Claims>(token, &key, &validation).map_err(|e| IdentityError::Unauthorized(e.to_string()))?;
        if data.claims.sub.is_empty() {
            return Err(IdentityError::Unauthorized("empty subject".into()));
        }
        Ok(data.claims.sub)
    }

    /// Resolve the request headers to a caller. `Ok(None)` means anonymous (no token sent).
    pub async fn caller(&self, headers: &HeaderMap) -> Result<Option<Caller>, IdentityError> {
        let Some(token) = bearer_token(headers) else {
            return Ok(None);
        };
        let user_id = self.verify(token).await?;
        let profile_id = headers
            .get("x-profile-id")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let (profile_id, maturity) = match profile_id {
            None => (None, Maturity::Adult),
            Some(pid) => {
                if !is_uuid(&pid) {
                    return Err(IdentityError::Forbidden);
                }
                let info = self.profile(token, &user_id, &pid).await?;
                (Some(pid), info.maturity)
            }
        };
        Ok(Some(Caller { user_id, profile_id, maturity }))
    }

    async fn profile(&self, token: &str, user_id: &str, profile_id: &str) -> Result<ProfileInfo, IdentityError> {
        let key = format!("{user_id}:{profile_id}");
        if let Some(hit) = self.profiles.get(&key).await {
            return Ok(hit);
        }
        let url = format!("{}/rest/v1/profiles?id=eq.{profile_id}&select=id,kind,maturity,profile_settings(age_rating)", self.supabase_url);
        let rows: Vec<Value> = self
            .http
            .get(&url)
            .header("apikey", &self.publishable_key)
            .bearer_auth(token)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| IdentityError::Unavailable(e.to_string()))?
            .json()
            .await
            .map_err(|e| IdentityError::Unavailable(e.to_string()))?;
        let row = rows.first().ok_or(IdentityError::Forbidden)?;
        let info = profile_from_row(row).ok_or_else(|| IdentityError::Unavailable("unreadable profile row".into()))?;
        self.profiles.insert(key, info).await;
        Ok(info)
    }
}

fn profile_from_row(row: &Value) -> Option<ProfileInfo> {
    let base = Maturity::parse(row.get("maturity")?.as_str()?)?;
    // the profile's own age rating (profile_settings.age_rating, the British rating the owner set) wins; the old maturity is the fallback
    let rating = row
        .get("profile_settings")
        .and_then(|s| if s.is_array() { s.get(0) } else { Some(s) })
        .and_then(|s| s.get("age_rating"))
        .and_then(Value::as_str)
        .and_then(Maturity::from_rating);
    let maturity = rating.unwrap_or(base);
    // a kids profile can never be above PG (the database enforces it too); belt and braces
    let kind = row.get("kind").and_then(Value::as_str).unwrap_or("adult");
    let maturity = if kind == "kids" && !maturity.is_kids() { Maturity::OlderKids } else { maturity };
    Some(ProfileInfo { maturity })
}

pub fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let v = headers.get(axum::http::header::AUTHORIZATION)?.to_str().ok()?.trim();
    let (scheme, token) = v.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty() {
        Some(token.trim())
    } else {
        None
    }
}

/// 8-4-4-4-12 hex, so a profile id can be placed in a PostgREST filter without escaping.
pub fn is_uuid(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 36
        && b.iter().enumerate().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => *c == b'-',
            _ => c.is_ascii_hexdigit(),
        })
}

pub fn parse_jwks(jwks: &Value) -> HashMap<String, DecodingKey> {
    let mut out = HashMap::new();
    for k in jwks.get("keys").and_then(Value::as_array).into_iter().flatten() {
        let (Some(kid), Some(x), Some(y)) = (
            k.get("kid").and_then(Value::as_str),
            k.get("x").and_then(Value::as_str),
            k.get("y").and_then(Value::as_str),
        ) else {
            continue;
        };
        if k.get("kty").and_then(Value::as_str) != Some("EC") || k.get("crv").and_then(Value::as_str) != Some("P-256") {
            continue;
        }
        if let Ok(key) = DecodingKey::from_ec_components(x, y) {
            out.insert(kid.to_string(), key);
        }
    }
    out
}

// ── Axum extractor ───────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, EncodingKey, Header};
    use serde_json::json;

    // A throwaway test key pair (generated for these tests only, never used anywhere else).
    const TEST_PEM: &str = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg9vSGPbAJB14SsCVQ\nadOkhnib6hRF7+lDNKNhSNW5WDWhRANCAAS3Ch4S64dod4g1im63LJ5xaNhE3kz+\n6dN4gGsK+ovYs2lO0jMR4ZyO6ktWnANlQbBgcgOdVcRcmlbcqGIxTDCT\n-----END PRIVATE KEY-----\n";
    const TEST_X: &str = "twoeEuuHaHeINYputyyecWjYRN5M_unTeIBrCvqL2LM";
    const TEST_Y: &str = "aU7SMxHhnI7qS1acA2VBsGByA51VxFyaVtyoYjFMMJM";
    const URL: &str = "https://example.supabase.co";

    fn identity() -> Identity {
        let id = Identity::new(URL, "pk");
        let n = id.set_jwks(&json!({"keys":[{"kty":"EC","crv":"P-256","alg":"ES256","kid":"k1","x":TEST_X,"y":TEST_Y}]}));
        assert_eq!(n, 1);
        id
    }

    fn token(kid: &str, aud: &str, iss: &str, exp_in: i64, sub: &str) -> String {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        let mut h = Header::new(Algorithm::ES256);
        h.kid = Some(kid.to_string());
        encode(&h, &json!({"sub": sub, "aud": aud, "iss": iss, "exp": now + exp_in, "role": "authenticated"}), &EncodingKey::from_ec_pem(TEST_PEM.as_bytes()).unwrap()).unwrap()
    }

    fn good(sub: &str) -> String {
        token("k1", "authenticated", &format!("{URL}/auth/v1"), 3600, sub)
    }

    #[tokio::test]
    async fn accepts_a_valid_token() {
        assert_eq!(identity().verify(&good("user-1")).await.unwrap(), "user-1");
    }

    #[tokio::test]
    async fn refuses_expired_wrong_audience_wrong_issuer_and_unknown_key() {
        let id = identity();
        let iss = format!("{URL}/auth/v1");
        assert!(matches!(id.verify(&token("k1", "authenticated", &iss, -3600, "u")).await, Err(IdentityError::Unauthorized(_))));
        assert!(matches!(id.verify(&token("k1", "anon", &iss, 3600, "u")).await, Err(IdentityError::Unauthorized(_))));
        assert!(matches!(id.verify(&token("k1", "authenticated", "https://evil.example/auth/v1", 3600, "u")).await, Err(IdentityError::Unauthorized(_))));
        // an unknown key id triggers a refetch, which cannot succeed here (no network to this host), so it is refused either way
        assert!(id.verify(&token("other", "authenticated", &iss, 3600, "u")).await.is_err());
    }

    #[tokio::test]
    async fn refuses_a_token_signed_by_another_key_and_garbage() {
        let id = identity();
        let mut t = good("u");
        // flip a character inside the signature
        let last = t.pop().unwrap();
        t.push(if last == 'A' { 'B' } else { 'A' });
        assert!(matches!(id.verify(&t).await, Err(IdentityError::Unauthorized(_))));
        assert!(matches!(id.verify("not.a.token").await, Err(IdentityError::Unauthorized(_))));
        assert!(matches!(id.verify("").await, Err(IdentityError::Unauthorized(_))));
    }

    #[tokio::test]
    async fn refuses_hmac_tokens() {
        // an HS256 token signed with the public key bytes must not pass (algorithm confusion)
        let mut h = Header::new(Algorithm::HS256);
        h.kid = Some("k1".into());
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        let t = encode(&h, &json!({"sub":"u","aud":"authenticated","iss":format!("{URL}/auth/v1"),"exp":now+3600}), &EncodingKey::from_secret(TEST_X.as_bytes())).unwrap();
        assert!(matches!(identity().verify(&t).await, Err(IdentityError::Unauthorized(_))));
    }

    #[tokio::test]
    async fn anonymous_without_a_token_and_bad_profile_ids_are_refused() {
        let id = identity();
        let mut h = HeaderMap::new();
        assert!(id.caller(&h).await.unwrap().is_none());
        h.insert("authorization", format!("Bearer {}", good("u1")).parse().unwrap());
        let c = id.caller(&h).await.unwrap().unwrap();
        assert_eq!((c.user_id.as_str(), c.profile_id.is_none(), c.maturity), ("u1", true, Maturity::Adult));
        assert_eq!(c.session_key(), "u1");
        h.insert("x-profile-id", "1' or '1'='1".parse().unwrap());
        assert!(matches!(id.caller(&h).await, Err(IdentityError::Forbidden)));
    }

    #[test]
    fn bearer_parsing_and_uuid_check() {
        let mut h = HeaderMap::new();
        assert!(bearer_token(&h).is_none());
        h.insert("authorization", "Basic abc".parse().unwrap());
        assert!(bearer_token(&h).is_none());
        h.insert("authorization", "bearer  tok ".parse().unwrap());
        assert_eq!(bearer_token(&h), Some("tok"));
        assert!(is_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!is_uuid("550e8400e29b41d4a716446655440000"));
        assert!(!is_uuid("550e8400-e29b-41d4-a716-44665544000g"));
    }

    #[test]
    fn profile_rows_map_to_maturity_and_kids_never_adult() {
        assert_eq!(profile_from_row(&json!({"kind":"adult","maturity":"adult"})).unwrap().maturity, Maturity::Adult);
        assert_eq!(profile_from_row(&json!({"kind":"kids","maturity":"little_kids"})).unwrap().maturity, Maturity::LittleKids);
        assert_eq!(profile_from_row(&json!({"kind":"kids","maturity":"adult"})).unwrap().maturity, Maturity::OlderKids);
        assert!(profile_from_row(&json!({"kind":"adult","maturity":"wat"})).is_none());
    }

    type Seen = Arc<Mutex<Vec<(String, String, String)>>>;

    /// A local stand in for PostgREST. It records (apikey, bearer token, id filter) of every request and returns the
    /// profile row only for the one profile it owns, like row level security would.
    async fn fake_postgrest(owned_profile: &'static str, kind: &'static str, maturity: &'static str) -> (String, Seen) {
        use axum::{extract::Query, routing::get, Router};
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let app = Router::new().route(
            "/rest/v1/profiles",
            get(move |headers: HeaderMap, Query(q): Query<HashMap<String, String>>| {
                let log = log.clone();
                async move {
                    let apikey = headers.get("apikey").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                    let bearer = bearer_token(&headers).unwrap_or("").to_string();
                    let id = q.get("id").cloned().unwrap_or_default();
                    log.lock().unwrap().push((apikey, bearer, id.clone()));
                    if id == format!("eq.{owned_profile}") {
                        Json(json!([{"id": owned_profile, "kind": kind, "maturity": maturity}]))
                    } else {
                        Json(json!([]))
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), seen)
    }

    #[tokio::test]
    async fn a_valid_token_and_an_owned_profile_give_the_profile_maturity() {
        const PROFILE: &str = "550e8400-e29b-41d4-a716-446655440000";
        const OTHER: &str = "11111111-2222-3333-4444-555555555555";
        let (base, seen) = fake_postgrest(PROFILE, "kids", "older_kids").await;
        let id = Identity::new(&base, "pk");
        id.set_jwks(&json!({"keys":[{"kty":"EC","crv":"P-256","alg":"ES256","kid":"k1","x":TEST_X,"y":TEST_Y}]}));
        let tok = token("k1", "authenticated", &format!("{base}/auth/v1"), 3600, "user-1");

        let mut h = HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        h.insert("x-profile-id", PROFILE.parse().unwrap());
        let c = id.caller(&h).await.unwrap().unwrap();
        assert_eq!(c.maturity, Maturity::OlderKids);
        assert_eq!(c.session_key(), PROFILE, "the session is the verified profile");
        {
            let log = seen.lock().unwrap();
            assert_eq!(log.len(), 1);
            assert_eq!(log[0], ("pk".to_string(), tok.clone(), format!("eq.{PROFILE}")), "lookup uses the caller's own token and the publishable key");
        }
        // second call comes from the cache
        id.caller(&h).await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 1);

        // a profile row the caller cannot see (RLS returns nothing) is refused
        h.insert("x-profile-id", OTHER.parse().unwrap());
        assert!(matches!(id.caller(&h).await, Err(IdentityError::Forbidden)));
    }

    /// Network check against the real project key set: `cargo test live_jwks -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn live_jwks_parses() {
        let url = std::env::var("SUPABASE_URL").expect("set SUPABASE_URL");
        let id = Identity::new(&url, "unused");
        id.refresh_keys().await.expect("fetch jwks");
        let n = id.keys.read().unwrap().len();
        println!("{n} ES256 key(s) loaded from {url}");
        assert!(n >= 1);
    }

    fn rated(cert: u8) -> CatalogItem {
        let mut it = item(&["Drama"], &[], false, 0);
        it.cert = cert;
        it
    }

    /// The same title with a known rating (U unless said otherwise).
    fn with_cert(mut it: CatalogItem, cert: u8) -> CatalogItem {
        it.cert = cert;
        it
    }

    #[test]
    fn a_known_age_rating_decides_what_each_tier_sees() {
        // little kids see U only, older kids up to PG (and only titles made for children), teens up to 15, adults everything
        let kids_title = |c| with_cert(item(&["Kids", "Family"], &[], false, 0), c);
        assert!(Maturity::LittleKids.allows(&kids_title(1)) && !Maturity::LittleKids.allows(&kids_title(2)));
        assert!(Maturity::OlderKids.allows(&kids_title(2)) && !Maturity::OlderKids.allows(&kids_title(3)));
        // a PG drama or murder mystery is not for children whatever its rating says, and an unrated title is never shown to them
        assert!(!Maturity::OlderKids.allows(&rated(2)));
        assert!(!Maturity::OlderKids.allows(&with_cert(item(&["Mystery", "Drama", "Crime"], &[], false, 0), 2)));
        assert!(!Maturity::OlderKids.allows(&with_cert(item(&["Family"], &[], false, 0), 0)));
        assert!(Maturity::Twelve.allows(&rated(3)) && !Maturity::Twelve.allows(&rated(4)));
        assert!(Maturity::Fifteen.allows(&rated(4)) && !Maturity::Fifteen.allows(&rated(5)));
        assert!(Maturity::Adult.allows(&rated(5)));
    }

    fn item(genres: &[&str], keywords: &[&str], adult: bool, flags: u8) -> CatalogItem {
        CatalogItem {
            id: 1,
            tmdb_id: 1,
            media_type: "tv".into(),
            title: "t".into(),
            release_year: None,
            genres: genres.iter().map(|s| s.to_string()).collect(),
            overview: String::new(),
            tagline: None,
            director: None,
            poster_path: None,
            backdrop_path: None,
            vote_average: 7.0,
            popularity: 10.0,
            is_anime: false,
            vote_count: 100,
            adult,
            original_language: "en".into(),
            tv_type: None,
            countries: vec![],
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            runtime: 0,
            flags,
            cert: 0, cert_gb: 0, cert_us: 0,
        }
    }

    #[test]
    fn maturity_policy() {
        let kids_show = with_cert(item(&["Kids", "Animation"], &[], false, 0), 1);
        let family_film = with_cert(item(&["Family", "Comedy"], &["friendship"], false, 0), 1);
        let adult_cartoon = with_cert(item(&["Animation", "Comedy"], &[], false, 0), 2);
        let horror = with_cert(item(&["Horror", "Family"], &[], false, 0), 1);
        let violent_anime = with_cert(item(&["Animation", "Action & Adventure"], &["gore", "war"], false, 0), 2);
        let drama = with_cert(item(&["Drama"], &[], false, 0), 2);
        let adult_flag = item(&["Drama"], &[], true, 0);
        let suggestive = item(&["Animation"], &[], false, FLAG_SUGGESTIVE);

        for m in [Maturity::LittleKids, Maturity::OlderKids, Maturity::Twelve, Maturity::Fifteen, Maturity::Adult] {
            assert!(m.allows(&kids_show), "{m:?} kids show");
        }
        assert!(Maturity::LittleKids.allows(&family_film));
        assert!(!Maturity::LittleKids.allows(&adult_cartoon), "animation alone is not enough for little kids");
        assert!(Maturity::OlderKids.allows(&adult_cartoon), "older kids may see unlabelled animation");
        for m in [Maturity::LittleKids, Maturity::OlderKids] {
            assert!(!m.allows(&horror), "{m:?} horror");
            assert!(!m.allows(&violent_anime), "{m:?} violent keywords");
            assert!(!m.allows(&drama), "{m:?} plain drama");
        }
        assert!(Maturity::Fifteen.allows(&drama) && Maturity::Fifteen.allows(&horror));
        assert!(!Maturity::Fifteen.allows(&adult_flag) && !Maturity::Fifteen.allows(&suggestive));
        assert!(Maturity::Adult.allows(&adult_flag), "adult profiles keep the existing page safety nets only");
        assert_eq!(Maturity::parse(Maturity::OlderKids.as_str()), Some(Maturity::OlderKids));
    }
}
