use crate::vectors::{cosine_sim, l2_normalize, DIMS};
use anyhow::{bail, Result};
use pathfinding::matrix::Matrix;
use pathfinding::prelude::kuhn_munkres_min;
use serde::Deserialize;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use tracing::info;

/// A single taxonomy entry from `generate_taxonomy.py`.
#[derive(Debug, Clone)]
pub struct TaxonomyEntry {
    pub id: String,
    pub title: String,
    /// "movie", "tv", or "both"
    pub media_type: String,
    /// Row type: vibe, theme, setting, decade, country, animation, character,
    /// audience, style, identity, editorial, calendar (empty for legacy files)
    pub kind: String,
    /// Date window for seasonal rows (None means always eligible)
    pub season: Option<Season>,
    /// Seasonal rows only: catalog keywords a title must carry to fill the row, and whether the title profile must say LGBTQ+ is central
    pub keywords: Vec<String>,
    pub lgbtq_central: bool,
    /// L2-normalised 2560d embedding of the `vibe_query` string
    pub embedding: Vec<f32>,
    /// `embedding` minus the corpus mean vector, renormalised. Cosine in this
    /// space is not squashed by the shared offset of the embedding cone.
    pub centered: Vec<f32>,
}

/// Month/day window, inclusive. May wrap the new year (start after end).
#[derive(Debug, Clone, Copy)]
pub struct Season {
    pub start: (u8, u8),
    pub end: (u8, u8),
}

impl Season {
    pub fn active_on(&self, md: (u8, u8)) -> bool {
        if self.start <= self.end {
            md >= self.start && md <= self.end
        } else {
            md >= self.start || md <= self.end
        }
    }
}

fn parse_md(s: &str) -> Option<(u8, u8)> {
    let (m, d) = s.split_once('-')?;
    Some((m.trim().parse().ok()?, d.trim().parse().ok()?))
}

/// Legacy `seasonal` string keys from taxonomy_1000.json, or a `{start, end}` object.
fn parse_season(v: &serde_json::Value) -> Option<Season> {
    match v {
        serde_json::Value::String(k) => {
            let (a, b) = match k.as_str() {
                "holiday" => ((11, 20), (12, 31)),
                "halloween" => ((10, 1), (10, 31)),
                "summer" => ((6, 1), (8, 31)),
                "valentines" => ((2, 1), (2, 14)),
                _ => return None,
            };
            Some(Season { start: a, end: b })
        }
        serde_json::Value::Object(o) => Some(Season {
            start: parse_md(o.get("start")?.as_str()?)?,
            end: parse_md(o.get("end")?.as_str()?)?,
        }),
        _ => None,
    }
}

/// Today's (month, day) in UTC, without a date library.
pub fn today_md() -> (u8, u8) {
    // STREAMS_TODAY=MM-DD pretends it is that day (to check seasonal rows)
    if let Some((m, d)) = std::env::var("STREAMS_TODAY").ok().as_deref().and_then(|v| v.split_once('-')).and_then(|(m, d)| Some((m.parse().ok()?, d.parse().ok()?))) {
        return (m, d);
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    let _ = era;
    (m, d)
}

/// Sidecar JSON produced alongside `taxonomy_embeddings.bin`.
#[derive(Deserialize)]
struct TaxonomySidecar {
    entries: Vec<TaxonomyMeta>,
}

#[derive(Deserialize)]
struct TaxonomyMeta {
    id: String,
    title: String,
    media_type: String,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    seasonal: Option<serde_json::Value>,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    lgbtq_central: bool,
}

/// In-memory store of all taxonomy embeddings.
pub struct TaxonomyStore {
    pub entries: Vec<TaxonomyEntry>,
    /// Per entry (movie, tv) count of quality titles among its nearest neighbours.
    /// u16::MAX means not computed yet. See support.rs.
    support: Vec<(AtomicU16, AtomicU16)>,
}

fn unknown_support(n: usize) -> Vec<(AtomicU16, AtomicU16)> {
    (0..n).map(|_| (AtomicU16::new(u16::MAX), AtomicU16::new(u16::MAX))).collect()
}

impl TaxonomyStore {
    /// Load from binary + JSON sidecar produced by `tools/embed_taxonomy.py`.
    ///
    /// Binary format:
    /// - 4 bytes: n_entries (u32 LE)
    /// - n_entries × (DIMS × 4) bytes: f32 embeddings
    ///
    /// JSON sidecar (`taxonomy_embeddings.json`):
    /// `{"entries": [{"id": "...", "title": "...", "media_type": "..."}]}`
    pub fn load(bin_path: &str, json_path: &str) -> Result<Arc<Self>> {
        // Load metadata
        let json_bytes = std::fs::read(json_path)?;
        let sidecar: TaxonomySidecar = serde_json::from_slice(&json_bytes)?;

        // Load embeddings
        let bin_bytes = std::fs::read(bin_path)?;
        let n = u32::from_le_bytes(bin_bytes[0..4].try_into()?) as usize;
        if n != sidecar.entries.len() {
            bail!("taxonomy_embeddings.bin n={n} != sidecar entries={}", sidecar.entries.len());
        }

        let expected_size = 4 + n * DIMS * 4;
        if bin_bytes.len() != expected_size {
            bail!("taxonomy_embeddings.bin size mismatch");
        }

        let mut entries = Vec::with_capacity(n);
        for (i, meta) in sidecar.entries.into_iter().enumerate() {
            let offset = 4 + i * DIMS * 4;
            let mut embedding: Vec<f32> = bin_bytes[offset..offset + DIMS * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            l2_normalize(&mut embedding);

            let season = meta.seasonal.as_ref().and_then(parse_season);
            entries.push(TaxonomyEntry {
                id: meta.id,
                title: meta.title,
                media_type: meta.media_type,
                kind: meta.kind.unwrap_or_default(),
                season,
                keywords: meta.keywords.iter().map(|k| k.to_lowercase()).collect(),
                lgbtq_central: meta.lgbtq_central,
                centered: Vec::new(),
                embedding,
            });
        }

        info!("TaxonomyStore loaded: {} entries", entries.len());
        let support = unknown_support(entries.len());
        Ok(Arc::new(Self { entries, support }))
    }

    /// Fill `centered` for every entry from the corpus mean vector.
    pub fn center_with(&mut self, mean: &[f32]) {
        for e in &mut self.entries {
            let mut c: Vec<f32> = e.embedding.iter().zip(mean).map(|(x, m)| x - m).collect();
            l2_normalize(&mut c);
            e.centered = c;
        }
    }

    /// Fallback: use pre-computed embeddings passed directly (for tests / cold start).
    pub fn from_raw(entries: Vec<TaxonomyEntry>) -> Arc<Self> {
        let support = unknown_support(entries.len());
        Arc::new(Self { entries, support })
    }

    pub fn set_support(&self, i: usize, movie: u16, tv: u16) {
        self.support[i].0.store(movie, Ordering::Relaxed);
        self.support[i].1.store(tv, Ordering::Relaxed);
    }

    pub fn support_raw(&self, i: usize) -> (u16, u16) {
        (self.support[i].0.load(Ordering::Relaxed), self.support[i].1.load(Ordering::Relaxed))
    }

    /// Quality titles this entry can fill for a surface, None while unknown.
    pub fn support_for(&self, i: usize, surface: &str) -> Option<u16> {
        let (m, t) = self.support_raw(i);
        if m == u16::MAX || t == u16::MAX {
            return None;
        }
        Some(match surface {
            "movie" => m,
            "tv" => t,
            _ => m.saturating_add(t),
        })
    }

    /// Hungarian (Kuhn-Munkres) shelf alignment.
    ///
    /// Assigns each of the N row centroids a UNIQUE taxonomy title that
    /// maximises total cosine similarity across the page simultaneously.
    /// Unlike greedy first-come-first-served, this finds the globally optimal
    /// bijective assignment — no row gets a leftover title because row 1 took the best.
    ///
    /// Returns Vec<Option<String>>: title per row (None if best cosine < fallback_threshold).
    pub fn hungarian_assign(
        &self,
        row_centroids: &[Vec<f32>],
        surface: &str, // "movie", "tv", or "both"
        fallback_threshold: f32,
    ) -> Vec<Option<String>> {
        let n_rows = row_centroids.len();
        if n_rows == 0 || self.entries.is_empty() {
            return vec![None; n_rows];
        }

        // Filter compatible taxonomy entries for this surface
        let today = today_md();
        let compatible: Vec<&TaxonomyEntry> = self
            .entries
            .iter()
            .filter(|e| {
                (e.media_type == "both" || e.media_type == surface || surface == "both")
                    && e.season.map_or(true, |s| s.active_on(today))
            })
            .collect();

        if compatible.is_empty() {
            return vec![None; n_rows];
        }

        // Build N×M integer cost matrix using pathfinding::matrix::Matrix.
        // kuhn_munkres_min requires Matrix<C>, not Vec<Vec<C>>, and at least as many columns as rows:
        // when fewer entries are usable than there are rows, dummy columns of high cost are added and a row assigned to one gets no title.
        // (1.0 - cosine_sim) × 10_000 → minimising cost maximises similarity.
        let n_rows = row_centroids.len();
        let n_real = compatible.len();
        let n_cols = n_real.max(n_rows);
        let comp = &compatible;
        let flat: Vec<i64> = row_centroids
            .iter()
            .flat_map(|centroid| {
                (0..n_cols).map(move |j| match comp.get(j) {
                    Some(entry) => ((1.0 - cosine_sim(centroid, &entry.embedding)) * 10_000.0) as i64,
                    None => 30_000,
                })
            })
            .collect();
        let costs = Matrix::from_vec(n_rows, n_cols, flat)
            .expect("cost matrix dimensions must match");

        // Hungarian assignment. Returns (total_cost, assignment[i] = column_j)
        let (_total, assignment) = kuhn_munkres_min(&costs);

        // Translate assignments → titles
        row_centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let col = assignment[i];
                let Some(entry) = compatible.get(col) else { return None };
                let sim = cosine_sim(centroid, &entry.embedding);
                if sim >= fallback_threshold {
                    Some(entry.title.clone())
                } else {
                    None // caller falls back to "{Mood} {Genre}" construction
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod assign_tests {
    use super::*;

    fn entry(title: &str, v: Vec<f32>) -> TaxonomyEntry {
        TaxonomyEntry { id: title.into(), title: title.into(), media_type: "both".into(), kind: "theme".into(), season: None, keywords: vec![], lgbtq_central: false, embedding: v, centered: vec![] }
    }

    /// With fewer usable entries than rows (a vocabulary that holds only the seasonal entries) the matching used to panic; rows left without an entry get no title.
    #[test]
    fn fewer_entries_than_rows_does_not_panic() {
        let store = TaxonomyStore::from_raw(vec![entry("Only One", vec![1.0, 0.0])]);
        let rows = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.8]];
        let titles = store.hungarian_assign(&rows, "both", 0.5);
        assert_eq!(titles.len(), 3);
        assert_eq!(titles.iter().filter(|t| t.is_some()).count(), 1);
        assert_eq!(titles[0].as_deref(), Some("Only One"));
    }
}
