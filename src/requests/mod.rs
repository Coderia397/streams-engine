//! Rows made from requests (note 96). A request is a name and fact rules: the catalog is filtered by the rules, what passes is ranked by closeness to the centre of the
//! request's titles, and the row carries the request's own name, so no row is named from examples. The page builder is `page/requests.rs`.
//!
//! On while `STREAMS_REQUEST_ROWS=1` and `requests_v1.json` sits next to the database; off otherwise, and the pages are built as before.
mod rules;
mod store;

pub use rules::Rules;
pub use store::{Request, RequestDef, RequestStore};

/// The switch, read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("STREAMS_REQUEST_ROWS").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false))
}
