use anyhow::Result;
use arc_swap::ArcSwap;
use rusqlite::{Connection, OpenFlags};
use std::sync::atomic::{AtomicU32, Ordering};
use crate::det::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

mod certs;
mod index;
mod item;
mod keys;
mod load;
mod overlays;
mod refresh;

pub use certs::*;
pub use index::*;
pub use item::*;
pub use keys::*;
pub use load::{load_catalog, make_shared_catalog};
pub use overlays::{genre_learned, CATEGORIES_LOADED, POPULARITY_ASOF};
pub use refresh::spawn_catalog_refresh;
pub(crate) use overlays::*;

#[cfg(test)]
mod tests;
