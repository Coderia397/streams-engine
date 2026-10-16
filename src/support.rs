//! Row support: how many quality titles each taxonomy entry can actually fill.
//!
//! Computed once per (taxonomy, index) with one ANN query per entry and cached on
//! disk next to the taxonomy file, so page building only considers entries that can
//! fill a row instead of discovering the shortfall at request time.

use crate::catalog::CatalogMap;
use crate::hnsw::HnswIndex;
use crate::page::passes_member_quality;
use crate::taxonomy::TaxonomyStore;
use rayon::prelude::*;
use std::sync::atomic::Ordering;
use tracing::info;

/// ANN depth used when counting support (matches the depth pages retrieve with).
pub const SUPPORT_K: usize = 500;

fn cache_path(taxonomy_bin: &str) -> String {
    format!("{taxonomy_bin}.support")
}

/// Try to load cached support. The cache is valid for the same entry count and index size.
pub fn load_cache(store: &TaxonomyStore, taxonomy_bin: &str, index_len: usize) -> bool {
    let Ok(bytes) = std::fs::read(cache_path(taxonomy_bin)) else { return false };
    if bytes.len() < 12 {
        return false;
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let len = u64::from_le_bytes(bytes[4..12].try_into().unwrap()) as usize;
    if n != store.entries.len() || len != index_len || bytes.len() != 12 + n * 4 {
        return false;
    }
    for i in 0..n {
        let o = 12 + i * 4;
        store.set_support(
            i,
            u16::from_le_bytes([bytes[o], bytes[o + 1]]),
            u16::from_le_bytes([bytes[o + 2], bytes[o + 3]]),
        );
    }
    info!("Taxonomy support loaded from cache ({n} entries)");
    true
}

/// Count quality movie and tv titles among each entry's nearest neighbours, in parallel.
pub fn compute(store: &TaxonomyStore, catalog: &CatalogMap, index: &HnswIndex, taxonomy_bin: &str) {
    let t0 = std::time::Instant::now();
    (0..store.entries.len()).into_par_iter().for_each(|i| {
        let e = &store.entries[i];
        let (mut movie, mut tv) = (0u16, 0u16);
        if let Ok(hits) = index.search(&e.embedding, SUPPORT_K) {
            for (id, _) in hits {
                if let Some(item) = catalog.get(&id) {
                    if passes_member_quality(item) {
                        if item.media_type == "tv" { tv += 1 } else { movie += 1 }
                    }
                }
            }
        }
        store.set_support(i, movie, tv);
    });
    let n = store.entries.len();
    let mut out = Vec::with_capacity(12 + n * 4);
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(index.len() as u64).to_le_bytes());
    for i in 0..n {
        let (m, t) = store.support_raw(i);
        out.extend_from_slice(&m.to_le_bytes());
        out.extend_from_slice(&t.to_le_bytes());
    }
    let path = cache_path(taxonomy_bin);
    let tmp = format!("{path}.tmp");
    if std::fs::write(&tmp, &out).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
    let ok = (0..n).filter(|&i| store.support_for(i, "both").is_some_and(|s| s >= 12)).count();
    info!("Taxonomy support computed for {n} entries in {:.1}s: {ok} can fill a row", t0.elapsed().as_secs_f32());
    let _ = Ordering::Relaxed;
}
