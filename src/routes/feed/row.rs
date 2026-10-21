//! GET /api/feed/row: more titles of one row of a page. The page is named by the same query as `/api/feed` (surface, genre, session and so on), the row by its `row`
//! id, and `item_offset` and `item_limit` say which slice of the row's titles to send. The full rows come from the page built for `/api/feed` (`row_cache`); when
//! they have gone, the page is built again.
use super::{cache, session::SessionView, query::FeedRequest, FeedQuery, ROW_PAGE};
use crate::app::AppState;
use crate::identity::MaybeCaller;
use axum::{extract::{Query, State}, http::header, response::IntoResponse};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Deserialize)]
pub struct RowQuery {
    #[serde(flatten)]
    pub page: FeedQuery,
    pub row: String,
    #[serde(default)]
    pub item_offset: usize,
    pub item_limit: Option<usize>,
}

/// A row as the feed sends it: its first ROW_PAGE titles, with `row_total` and `has_more_items` saying what `/api/feed/row` can add.
pub(super) fn first_page(mut row: Value) -> Value {
    if let Some(obj) = row.as_object_mut() {
        let total = obj.get("items").and_then(|i| i.as_array()).map_or(0, |a| a.len());
        if let Some(Value::Array(items)) = obj.get_mut("items") {
            items.truncate(ROW_PAGE);
        }
        obj.insert("row_total".to_string(), json!(total));
        obj.insert("has_more_items".to_string(), json!(total > ROW_PAGE));
    }
    row
}

/// A row with every title and its `row_total` (checks ask for `full_rows=1`).
pub(super) fn with_total(mut row: Value) -> Value {
    if let Some(obj) = row.as_object_mut() {
        let total = obj.get("items").and_then(|i| i.as_array()).map_or(0, |a| a.len());
        obj.insert("row_total".to_string(), json!(total));
        obj.insert("has_more_items".to_string(), json!(false));
    }
    row
}

pub async fn feed_row(State(state): State<Arc<AppState>>, MaybeCaller(caller): MaybeCaller, Query(q): Query<RowQuery>) -> impl IntoResponse {
    let req = FeedRequest::parse(&q.page, &caller);
    let session = SessionView::of(&state, req.session_id.as_deref());
    let slot = cache::slot();
    let key = cache::page_key(&req, &session, slot);
    let rows = match state.row_cache.get(&key).await {
        Some(r) => r,
        None => {
            // the page has gone from the cache: build it again (this also refills row_cache under the same key)
            let _ = state.feed_cache.get_with(key.clone(), super::build_page_body(&state, &req, &session, slot)).await;
            state.row_cache.get(&key).await.unwrap_or_default()
        }
    };
    let limit = q.item_limit.unwrap_or(ROW_PAGE * 2).clamp(1, crate::page::MAX_ROW_DEPTH);
    let found = rows.iter().find(|r| r.get("id").and_then(|i| i.as_str()) == Some(q.row.as_str()));
    let body = match found {
        Some(row) => {
            let items = row.get("items").and_then(|i| i.as_array()).cloned().unwrap_or_default();
            let total = items.len();
            let slice: Vec<Value> = items.into_iter().skip(q.item_offset).take(limit).collect();
            let next = q.item_offset + slice.len();
            json!({ "id": q.row, "title": row.get("title"), "row_total": total, "item_offset": q.item_offset, "next_item_offset": next, "has_more_items": next < total, "items": slice })
        }
        None => json!({ "id": q.row, "row_total": 0, "items": [], "has_more_items": false, "error": "row not on this page" }),
    };
    ([(header::CONTENT_TYPE, "application/json")], serde_json::to_string(&body).unwrap())
}
