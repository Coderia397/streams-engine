//! Rows made for one profile from the signals the client sends (signals.rs): Continue Watching, My List, Because You Watched, More Like, Today's Top Picks for You, Your Next
//! Watch and Gems for You. Also the New Releases row and the collection rows (an event or a mood shown as a banner with a few lead titles). A profile without signals still gets
//! Today's Top Picks for You and Gems for You, built from the age limit, the quality of the catalog and a per session rotation.
//! Every row carries `meta` with a translation key and arguments (`title_key`, `title_args`) and, for rows about a title, a `reason`, so a client can write the title in its own language.
use crate::catalog::CatalogItem;
use crate::page::{entry_noise, PageContext, Row};
use crate::vectors::{cosine_sim, l2_normalize, DIMS};
use serde_json::json;
use crate::det::HashSet;

mod collections;
mod rows;

pub use collections::{event_header, events_in_season, prettify};
pub(crate) use collections::collection_meta;
pub use rows::is_gem;
pub(crate) use collections::blend_types;
#[cfg(test)]
pub(crate) use collections::event_target;
#[cfg(test)]
pub(crate) use rows::meta;

#[cfg(test)]
mod tests;
