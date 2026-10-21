//! GET /api/feed: a page of rows for a surface (home, films, series), a genre, a language or a collection.
//!
//! The steps, each in its own module: `query` (what was asked, normalised once), `session` (what the engine remembers of the caller), `cache` (the page key),
//! `policy` (how many rows), the page builder in `crate::page`, `removed` (rows taken off every page), `render` (the client's JSON shape).
mod cache;
mod policy;
mod query;
mod removed;
mod render;
mod row;
mod session;

pub use policy::{genre_row_cap, language_row_cap, MIN_GENRE_PAGE_ROWS};
pub use query::FeedQuery;
pub use row::feed_row;

/// Titles of a row the feed sends with the page; the rest come from `/api/feed/row` as the viewer pages the row (composed rows go to 150 titles, note 101).
pub const ROW_PAGE: usize = 24;

use crate::app::AppState;
use crate::identity::MaybeCaller;
use crate::page::{PageAsk, PageContext};
use axum::{
    extract::{Query, State},
    http::header,
    response::IntoResponse,
};
use query::{parse_taste, FeedRequest};
use serde_json::json;
use session::SessionView;
use std::sync::Arc;

/// GET /api/feed
/// Main discovery feed endpoint. Returns a page of thematic recommendation rows.
pub async fn feed(
    State(state): State<Arc<AppState>>,
    MaybeCaller(caller): MaybeCaller,
    Query(params): Query<FeedQuery>,
) -> impl IntoResponse {
    let req = FeedRequest::parse(&params, &caller);
    // Session memory: fall back to the session's learned taste when the client sends none, and drop titles the session has already started.
    let session = SessionView::of(&state, req.session_id.as_deref());
    let slot = cache::slot();
    // the page text differs with full_rows, the rows behind it do not (row_cache keeps the page key)
    let key = format!("{}{}", cache::page_key(&req, &session, slot), if req.full_rows { "|full" } else { "" });
    // A page is built once per key: callers that ask for it while it is being built wait for that build, and a cached page goes out as the text it was stored as.
    let body = state.feed_cache.get_with(key, build_page_body(&state, &req, &session, slot)).await;
    ([(header::CONTENT_TYPE, "application/json")], body)
}

/// Whether the caller asked for every title of every row (`full_rows=1`).
fn params_full_rows(req: &FeedRequest) -> bool {
    req.full_rows
}

/// The response of one page as JSON text.
async fn build_page_body(state: &AppState, req: &FeedRequest, session: &SessionView, slot: u64) -> String {
    let catalog = state.catalog.load();
    let fresh = state.fresh.load();
    let genre_spec = req.genre.as_deref().map(|g| crate::genre_spec::resolve(g, req.surface));
    let plan = policy::PagePlan::new(&catalog, &state.profiles, req.surface, genre_spec.as_ref(), req.language.as_deref(), req.maturity, req.offset + req.per_page);

    // Build taste query vector from client-provided genre affinities
    let mut affinities = parse_taste(&req.taste);
    if affinities.is_empty() {
        affinities = session.taste.clone();
    }
    let taste_vec = if affinities.is_empty() {
        // Cold start: no taste signal → use global average (zero vector → centroid blend fails gracefully)
        vec![0.0f32; crate::vectors::DIMS]
    } else {
        state.vault.blend_taste(&affinities)
    };

    let noise_session = cache::noise_session(req, slot);
    let ask = PageAsk {
        surface: req.surface,
        genre: req.genre.as_deref(),
        genre_spec: genre_spec.as_ref(),
        language: req.language.as_deref(),
        maturity: req.maturity,
        late_night: req.late_night,
        taste: taste_vec,
        target_rows: plan.target_rows,
        session: if req.session_id.is_some() { noise_session.as_str() } else { "" },
        signals: &req.signals,
    };
    let ctx = PageContext::new(state, &catalog, &fresh, ask);

    let found = match req.collection.as_deref() {
        Some(c) if c.starts_with("kids_") => ctx.build_kids_collection(c).await,
        Some(c) => ctx.build_mood_collection(c),
        None => None,
    };
    let (page_rows, collection_header) = match found {
        Some((rows, header)) => (rows, Some(header)),
        None => (ctx.build_page().await, None),
    };
    let page_rows = removed::without_removed(page_rows, &state.config.db_path);
    let all_rows = render::to_client_rows(page_rows, &session.seen, &state.profiles, &fresh, &catalog, ctx.composed_rows());
    let has_more = all_rows.len() > req.offset + req.per_page && req.offset + req.per_page < plan.cap;
    // the full rows are kept for paging inside a row; the page carries the first ROW_PAGE titles of each
    let key = cache::page_key(req, session, slot);
    state.row_cache.insert(key, Arc::new(all_rows.clone())).await;
    let full = params_full_rows(req);
    let rows: Vec<serde_json::Value> = all_rows.into_iter().skip(req.offset).take(req.per_page).map(|r| if full { row::with_total(r) } else { row::first_page(r) }).collect();
    let response = json!({
        "surface":     req.surface,
        "genre":       req.genre,
        "row_cap":     plan.cap,
        "genre_page_supported": plan.supported(genre_spec.is_some(), req.genre.is_some(), req.language.is_some()),
        "language":    req.language,
        "total_rows":  rows.len(),
        "offset":      req.offset,
        "next_offset": req.offset + rows.len(),
        "has_more":    has_more,
        "header":      collection_header.or_else(|| genre_spec.as_ref().and_then(|g| g.event_id()).and_then(|ev| crate::personal::event_header(&state.vibe, &catalog, ev))),
        "chips":       if req.genre.is_none() && req.language.is_none() { ctx.season_chips() } else { Vec::new() },
        "rows":        rows,
    });
    serde_json::to_string(&response).unwrap()
}
