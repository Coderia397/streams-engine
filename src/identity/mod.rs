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

mod caller;
mod maturity;
mod service;

pub use caller::{maturity_of, Caller, MaybeCaller};
pub use maturity::Maturity;
pub use service::{bearer_token, is_uuid, parse_jwks, Identity, IdentityError};
