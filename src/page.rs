#![allow(dead_code)] // Public API surface — functions used at runtime via dynamic dispatch

use crate::catalog::{CatalogItem, CatalogMap};
use crate::centroid::CentroidVault;
use crate::cluster::ClusterSeeds;
use crate::hnsw::{HnswIndex, ENGINE_READY};
use crate::mmr::{filter_outliers, greedy_mmr, mean_pairwise_cosine};
use crate::mood::MoodVault;
use crate::quality::{bayesian_quality_score, rerank_score};
use crate::taxonomy::TaxonomyStore;
use crate::titles::{byw_title, clamp_words, derive_title};
use crate::vectors::{cosine_sim, l2_normalize, sphere_mean, VectorStore, DIMS};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::Ordering;
use tokio::sync::RwLock;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Coherence gate thresholds (2560d Qwen3-Embedding space)
/// Measured on the real catalog: unrelated cluster centres have median cosine 0.64,
/// and the mean pairwise cosine inside a cluster is about 0.56, so gates sit near those.
const COHERENCE_STRONG: f32 = 0.66;
const COHERENCE_OK:     f32 = 0.55;
const COHERENCE_WEAK:   f32 = 0.50;
const COHERENCE_MIN_ITEMS: usize = 6;

/// Max cosine similarity between any two row anchors (MMD gate)
const MMD_MAX_INTER_SIM: f32 = 0.76;

/// MMR λ: 0 = pure diversity, 1 = pure relevance. 0.72 = production default.
const MMR_LAMBDA: f32 = 0.72;

/// Genre bitmask window: no genre repeated in the last N rows
const GENRE_WINDOW: usize = 4;

/// Entry driven pages need a real vocabulary; below this the cluster path is used.
const MIN_TAXONOMY_FOR_ENTRY_ROWS: usize = 200;

/// Best members (by quality) kept per seed before MMR
const MEMBER_POOL: usize = 120;

/// Items per row (max)
const ROW_SIZE: usize = 15;

/// ANN top-k for HNSW retrieval
const ANN_K: usize = 200;

/// Deeper retrieval for entry rows: only about 5% of the catalog passes the quality gate
const ANN_ENTRY_K: usize = 500;

/// Taxonomy cosine threshold to accept Hungarian assignment (else fallback title)
const TAXONOMY_THRESHOLD: f32 = 0.55;

/// BYW anchor: cosine to any existing row centroid must be below this
const BYW_MAX_CENTROID_SIM: f32 = 0.68;

// ── PageState ─────────────────────────────────────────────────────────────────

pub struct PageState {
    pub used_titles:   HashSet<String>,
    pub recent_genres: VecDeque<String>,
    /// Centroids of placed thematic rows, in placement order
    pub placed_centroids: Vec<Vec<f32>>,
}

impl PageState {
    pub fn new() -> Self {
        Self {
            used_titles:      HashSet::new(),
            recent_genres:    VecDeque::new(),
            placed_centroids: Vec::new(),
        }
    }

    pub fn can_place_genre(&self, genre: &str) -> bool {
        !self.recent_genres.iter().any(|g| g == genre)
    }

    pub fn record_thematic(&mut self, genre: String, centroid: Vec<f32>) {
        self.recent_genres.push_back(genre);
        if self.recent_genres.len() > GENRE_WINDOW {
            self.recent_genres.pop_front();
        }
        self.placed_centroids.push(centroid);
    }

    /// Claim the first unused title from candidates. Final fallback always wins.
    pub fn claim_title(&mut self, candidates: &[String]) -> String {
        let chosen = candidates
            .iter()
            .find(|t| !self.used_titles.contains(*t))
            .cloned()
            .unwrap_or_else(|| {
                candidates.last().cloned().unwrap_or_else(|| "More Picks".to_string())
            });
        self.used_titles.insert(chosen.clone());
        chosen
    }
}

// ── Row / RowItem ─────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct Row {
    pub title:     String,
    pub archetype: String, // "trending" | "top_picks" | "byw" | "thematic"
    /// Taxonomy row type (vibe, theme, country, calendar, ...) for entry driven rows
    #[serde(skip_serializing_if = "String::is_empty")]
    pub kind:      String,
    pub items:     Vec<RowItem>,
}

#[derive(Serialize, Clone)]
pub struct RowItem {
    /// Internal key (`catalog::item_key`), used for dedup and session state
    pub id:           u32,
    /// TMDB id, what clients get as `id` (only unique together with `media_type`)
    pub tmdb_id:      u32,
    pub title:        String,
    pub media_type:   String,
    pub overview:     String,
    pub poster_path:  Option<String>,
    pub backdrop_path: Option<String>,
    pub vote_average: f32,
    pub popularity:   f32,
    pub release_year: Option<i32>,
    pub genres:       Vec<String>,
}

impl RowItem {
    pub fn from_catalog(item: &CatalogItem) -> Self {
        Self {
            id:            item.id,
            tmdb_id:       item.tmdb_id,
            title:         item.title.clone(),
            media_type:    item.media_type.clone(),
            overview:      item.overview.clone(),
            poster_path:   item.poster_path.clone(),
            backdrop_path: item.backdrop_path.clone(),
            vote_average:  item.vote_average,
            popularity:    item.popularity,
            release_year:  item.release_year,
            genres:        item.genres.clone(),
        }
    }
}

// ── PageContext ───────────────────────────────────────────────────────────────

pub struct PageContext<'a> {
    pub catalog:     &'a CatalogMap,
    pub vectors:     &'a VectorStore,
    pub hnsw:        &'a RwLock<Option<HnswIndex>>,
    pub seeds:       &'a ClusterSeeds,
    pub vault:       &'a CentroidVault,
    pub taxonomy:    &'a TaxonomyStore,
    pub mood_vault:  &'a MoodVault,
    pub surface:     &'a str,   // "movie" | "tv" | "both"
    /// Optional genre as sent by the client (kept for row titles).
    pub genre:       Option<&'a str>,
    /// The rule that genre resolves to, applied to every row (see genre_spec.rs).
    pub genre_spec:  Option<&'a crate::genre_spec::GenreSpec>,
    pub taste:       Vec<f32>,  // blended query vector on unit sphere
    pub target_rows: usize,
    /// Corpus mean vector for centred scoring
    pub mean:        &'a [f32],
    /// Today's (month, day) for calendar rows
    pub today:       (u8, u8),
    /// Session id: seeds the exploration noise so a session sees a stable page
    pub session:     &'a str,
}

impl<'a> PageContext<'a> {
    fn matches_filters(&self, item: &CatalogItem) -> bool {
        (self.surface == "both" || item.media_type == self.surface || item.media_type == "both")
            && self.genre_spec.map_or(true, |spec| spec.matches(item))
    }

    // ── HNSW / brute-force retrieval ─────────────────────────────────────────

    /// Retrieve top-k candidates for a query vector.
    /// HNSW when index is ready; rayon brute-force otherwise.
    async fn retrieve_candidates(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if ENGINE_READY.load(Ordering::Relaxed) {
            let guard = self.hnsw.read().await;
            if let Some(idx) = guard.as_ref() {
                if let Ok(mut results) = idx.search(query, k) {
                    for (id, sim) in &mut results {
                        if let Some(item) = self.catalog.get(id) {
                            *sim = rerank_score(*sim, item.popularity);
                        }
                    }
                    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    return results;
                }
            }
        }
        self.vectors.brute_force_topk(query, k)
    }

    /// Synchronous variant of `retrieve_candidates` for use inside rayon, with the
    /// HNSW read guard already held by the caller.
    fn retrieve_sync(&self, index: &Option<HnswIndex>, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if ENGINE_READY.load(Ordering::Relaxed) {
            if let Some(idx) = index.as_ref() {
                if let Ok(mut results) = idx.search(query, k) {
                    for (id, sim) in &mut results {
                        if let Some(item) = self.catalog.get(id) {
                            *sim = rerank_score(*sim, item.popularity);
                        }
                    }
                    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    return results;
                }
            }
        }
        self.vectors.brute_force_topk(query, k)
    }

    // ── Main page assembly ────────────────────────────────────────────────────

    pub async fn build_page(&self) -> Vec<Row> {
        if self.genre.is_some() {
            return self.build_genre_page();
        }
        if self.taxonomy.entries.len() >= MIN_TAXONOMY_FOR_ENTRY_ROWS {
            let rows = self.build_entry_page().await;
            if rows.len() > 2 {
                return rows;
            }
        }
        let mut state = PageState::new();
        let mut rows: Vec<Row> = Vec::new();

        // thematic_centroids tracks cluster centroids in row-placement order,
        // parallel to the thematic subset of `rows`. Used for Hungarian alignment.
        let mut thematic_centroids: Vec<Vec<f32>> = Vec::new();
        let mut thematic_row_indices: Vec<usize> = Vec::new(); // indices into `rows`

        // ── Fixed rows ────────────────────────────────────────────────────────
        if let Some(row) = self.build_trending_row() {
            rows.push(row);
        }

        // ── Taste-biased seed scoring ─────────────────────────────────────────
        // Score each cluster seed by cosine similarity to taste vector.
        // Cold start (zero taste vector) → all scores equal → pure MMD order.
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let mut scored_seeds: Vec<(f32, usize)> = self
            .seeds
            .seeds
            .iter()
            .enumerate()
            .map(|(i, seed)| {
                let score = if has_taste {
                    cosine_sim(&self.taste, &seed.centroid)
                } else {
                    1.0
                };
                (score, i)
            })
            .collect();

        // Sort by relevance descending (preserves MMD order for equal scores)
        scored_seeds.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // ── Thematic rows from cluster seeds ──────────────────────────────────
        let t_seed_loop = std::time::Instant::now();
        let (mut n_visit, mut n_similar, mut n_empty, mut n_genre, mut n_mmr, mut n_items, mut n_row) = (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
        for (_, seed_idx) in &scored_seeds {
            if rows.len() >= self.target_rows + 1 {
                break; // +1 for trending row
            }

            let seed = &self.seeds.seeds[*seed_idx];
            n_visit += 1;

            // MMD inter-anchor gate: reject if too similar to any placed centroid
            let too_similar = thematic_centroids
                .iter()
                .any(|c| cosine_sim(&seed.centroid, c) > MMD_MAX_INTER_SIM);
            if too_similar {
                n_similar += 1;
                continue;
            }

            // Members that pass surface/genre and quality filters (no vectors yet)
            let mut cand_items: Vec<&CatalogItem> = seed
                .member_ids
                .iter()
                .filter_map(|id| self.catalog.get(id))
                .filter(|item| self.matches_filters(item) && passes_member_quality(item))
                .collect();

            if cand_items.is_empty() {
                n_empty += 1;
                continue;
            }

            // Dominant genre (majority vote)
            let dominant_genre =
                majority_genre(cand_items.iter().map(|item| item.genres.as_slice()));

            // Genre bitmask: no same genre in last GENRE_WINDOW rows.
            // Checked before any vector work so rejected seeds stay cheap.
            // A genre filtered page is one genre by design, so the rule does not apply.
            if self.genre.is_none() && !state.can_place_genre(&dominant_genre) {
                n_genre += 1;
                continue;
            }

            // Best quality members only, then fetch their vectors
            cand_items.sort_unstable_by(|a, b| {
                bayesian_quality_score(b.vote_average, b.popularity)
                    .partial_cmp(&bayesian_quality_score(a.vote_average, a.popularity))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            cand_items.truncate(MEMBER_POOL);
            let members: Vec<(&CatalogItem, Vec<f32>)> = cand_items
                .into_iter()
                .filter_map(|item| {
                    let idx = self.vectors.index_of(item.id)?;
                    Some((item, self.vectors.get_f32(idx)))
                })
                .collect();

            if members.is_empty() {
                continue;
            }

            // MMR: intra-row diversity
            let candidates: Vec<(u32, Vec<f32>)> = members
                .iter()
                .map(|(item, vec)| (item.id, vec.clone()))
                .collect();
            n_mmr += 1;
            let selected_ids =
                greedy_mmr(&seed.centroid, &candidates, ROW_SIZE * 2, MMR_LAMBDA);

            let (sel_ids, sel_vecs): (Vec<u32>, Vec<Vec<f32>>) = selected_ids
                .iter()
                .filter_map(|id| {
                    let idx = self.vectors.index_of(*id)?;
                    Some((*id, self.vectors.get_f32(idx)))
                })
                .unzip();

            // Coherence filter (outliers by medoid distance)
            let (fil_ids, fil_vecs) = filter_outliers(&sel_ids, &sel_vecs, COHERENCE_OK);

            // Coherence gate: weak clusters need more members to pass
            let coherence = mean_pairwise_cosine(&fil_vecs);
            let min_items = if coherence < COHERENCE_WEAK {
                COHERENCE_MIN_ITEMS + 4
            } else {
                COHERENCE_MIN_ITEMS
            };
            if fil_ids.len() < min_items {
                n_items += 1;
                continue;
            }

            // Cluster centroid from filtered items
            let centroid = sphere_mean(&fil_vecs);

            // Build row items (up to ROW_SIZE)
            let row_items: Vec<RowItem> = fil_ids
                .iter()
                .take(ROW_SIZE)
                .filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog))
                .collect();

            if row_items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }

            // Record state
            n_row += 1;
            let row_index = rows.len();
            thematic_row_indices.push(row_index);
            thematic_centroids.push(centroid.clone());
            state.record_thematic(dominant_genre, centroid);

            rows.push(Row {
                title:     "__PENDING__".to_string(), // replaced after Hungarian below
                archetype: "thematic".to_string(),
                kind:      String::new(),
                items:     row_items,
            });
        }

        tracing::debug!(
            "seed loop: {:.0} ms, visited {n_visit}, too similar {n_similar}, no members {n_empty}, genre window {n_genre}, mmr runs {n_mmr}, too few items {n_items}, rows {n_row}",
            t_seed_loop.elapsed().as_secs_f64() * 1000.0
        );

        // ── Optional BYW row ──────────────────────────────────────────────────
        // Only added if there's a taste vector. Anchor must be thematically
        // distinct from all placed thematic rows (cosine < BYW_MAX_CENTROID_SIM).
        if has_taste {
            if let Some(byw) = self.build_byw_row(&state).await {
                rows.push(byw);
            }
        }

        // ── Hungarian shelf alignment ─────────────────────────────────────────
        // Assign unique taxonomy titles to all thematic rows simultaneously.
        // thematic_centroids is PARALLEL to thematic_row_indices — no index arithmetic.
        let taxonomy_titles = self.taxonomy.hungarian_assign(
            &thematic_centroids,
            self.surface,
            TAXONOMY_THRESHOLD,
        );

        for (ti, &row_idx) in thematic_row_indices.iter().enumerate() {
            let centroid = &thematic_centroids[ti];
            let dominant_genre = majority_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
            );
            let taxonomy_title = taxonomy_titles.get(ti).and_then(|t| t.clone());
            let era = dominant_era(&rows[row_idx].items);
            let second = second_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
                &dominant_genre,
            );
            let candidates = derive_title(
                centroid, &dominant_genre, second.as_deref(), era.as_deref(),
                rows[row_idx].items.first().map(|i| i.title.as_str()),
                taxonomy_title, self.mood_vault,
            );
            let title = state.claim_title(&candidates);
            rows[row_idx].title = clamp_words(&title, 4);
        }

        rows
    }

    // ── Entry driven pages ────────────────────────────────────────────────────

    /// Netflix style page: rank row definitions (taxonomy entries) for this
    /// session first, then fill each chosen row with titles retrieved around the
    /// entry's own vector. Each entry is a row query, so the vocabulary can be
    /// far larger than the number of clusters and every row title is a real
    /// promise about what is inside.
    async fn build_entry_page(&self) -> Vec<Row> {
        const MIN_ROW: usize = 8;
        /// Entries known to fill fewer titles than this are not considered.
        const MIN_SUPPORT: u16 = 12;
        const POOL_PER_ROW: usize = 60;
        /// Hard ceiling on centred cosine between two chosen entries (near duplicates).
        const ENTRY_HARD_SIM: f32 = 0.85;
        /// Soft diversification weight: value = score - MU * (max centred cosine to chosen rows).
        const MU: f32 = 1.1;
        const EXPLORE: f32 = 0.6;
        /// Share of rows chosen by taste. The rest explore, so a concentrated taste
        /// still sees new territory (Spotify BaRT style explore/exploit).
        const EXPLOIT_SHARE: f32 = 0.7;

        let t0 = std::time::Instant::now();
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let taste_c: Vec<f32> = if has_taste {
            let mut c: Vec<f32> = self.taste.iter().zip(self.mean).map(|(x, m)| x - m).collect();
            l2_normalize(&mut c);
            c
        } else {
            Vec::new()
        };

        let mut rows: Vec<Row> = Vec::new();
        let mut used: HashSet<u32> = HashSet::new();
        if let Some(r) = self.build_trending_row() {
            r.items.iter().for_each(|i| { used.insert(i.id); });
            rows.push(r);
        }

        // 1. Score row definitions two ways: for taste (exploit) and for exploration
        struct Scored { idx: usize, exploit: f32, explore: f32 }
        let mut scored: Vec<Scored> = self
            .taxonomy
            .entries
            .par_iter()
            .enumerate()
            .filter(|(i, e)| {
                (e.media_type == "both" || e.media_type == self.surface || self.surface == "both")
                    && e.season.map_or(true, |s| s.active_on(self.today))
                    && !e.centered.is_empty()
                    && self.taxonomy.support_for(*i, self.surface).map_or(true, |n| n >= MIN_SUPPORT)
            })
            .map(|(i, e)| {
                let prior = 0.25 * kind_weight(&e.kind);
                let season = if e.season.is_some() { 0.6 } else { 0.0 }; // active window: near the top
                let noise = entry_noise(self.session, i);
                let taste_cos = if has_taste { cosine_sim(&taste_c, &e.centered) } else { 0.0 };
                Scored {
                    idx: i,
                    exploit: prior + season + 1.5 * taste_cos + 0.15 * noise,
                    explore: prior + season + EXPLORE * noise,
                }
            })
            .collect();
        let n_eligible = scored.len();

        let want = self.target_rows;
        let exploit_rows = if has_taste { ((want as f32) * EXPLOIT_SHARE).ceil() as usize } else { 0 };
        // Shortlist both rankings so each phase has enough to satisfy the page rules
        let per_list = (want * 4).clamp(60, 200);
        let mut pick_idx: Vec<usize> = Vec::new();
        if has_taste {
            scored.sort_unstable_by(|a, b| b.exploit.partial_cmp(&a.exploit).unwrap_or(std::cmp::Ordering::Equal));
            pick_idx.extend(scored.iter().take(per_list).map(|s| s.idx));
        }
        scored.sort_unstable_by(|a, b| b.explore.partial_cmp(&a.explore).unwrap_or(std::cmp::Ordering::Equal));
        pick_idx.extend(scored.iter().take(per_list).map(|s| s.idx));
        pick_idx.sort_unstable();
        pick_idx.dedup();
        let score_of: HashMap<usize, (f32, f32)> = scored.iter().map(|s| (s.idx, (s.exploit, s.explore))).collect();
        let t_score = t0.elapsed();

        // 2. Retrieve titles per shortlisted entry in parallel
        struct Cand { idx: usize, exploit: f32, explore: f32, hits: Vec<u32> }
        let guard = self.hnsw.read().await;
        let mut cands: Vec<Cand> = pick_idx
            .par_iter()
            .filter_map(|&idx| {
                let e = &self.taxonomy.entries[idx];
                let hits = self.retrieve_sync(&guard, &e.embedding, ANN_ENTRY_K);
                let good: Vec<(&CatalogItem, f32)> = hits
                    .iter()
                    .filter_map(|(id, sim)| self.catalog.get(id).map(|i| (i, *sim)))
                    .filter(|(i, _)| self.matches_filters(i) && passes_member_quality(i))
                    .collect();
                if good.len() < MIN_ROW {
                    return None;
                }
                let strength = 0.2
                    * good.iter().take(10)
                        .map(|(i, _)| bayesian_quality_score(i.vote_average, i.popularity) / 35.0)
                        .sum::<f32>()
                    / good.len().min(10) as f32;
                let (ex, xp) = score_of[&idx];
                Some(Cand {
                    idx,
                    exploit: ex + strength,
                    explore: xp + strength,
                    hits: good.into_iter().take(POOL_PER_ROW).map(|(i, _)| i.id).collect(),
                })
            })
            .collect();
        drop(guard);
        let t_retrieve = t0.elapsed();

        // 3. Selection: score based diversification (MMR over row definitions) under page rules
        let max_per_type = (((want as f32) * 0.16).ceil() as usize).max(2);
        let max_calendar = 2usize;
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut chosen: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut max_sim: Vec<f32> = vec![0.0; cands.len()];
        let mut alive: Vec<bool> = vec![true; cands.len()];
        let mut last_kind = String::new();

        while chosen.len() < want {
            let exploiting = chosen.len() < exploit_rows;
            let mut best: Option<(usize, f32)> = None;
            let mut relaxed: Option<(usize, f32)> = None; // ignores the adjacency rule
            for (ci, c) in cands.iter().enumerate() {
                if !alive[ci] || max_sim[ci] >= ENTRY_HARD_SIM {
                    continue;
                }
                let e = &self.taxonomy.entries[c.idx];
                let n = *type_counts.get(&e.kind).unwrap_or(&0);
                let cap = if e.kind == "calendar" { max_calendar } else { max_per_type };
                if n >= cap || c.hits.iter().filter(|id| !used.contains(id)).count() < MIN_ROW {
                    continue;
                }
                let base = if exploiting { c.exploit } else { c.explore };
                let value = base - MU * max_sim[ci].max(0.0);
                if relaxed.map_or(true, |(_, v)| value > v) {
                    relaxed = Some((ci, value));
                }
                if (e.kind != last_kind || e.kind.is_empty()) && best.map_or(true, |(_, v)| value > v) {
                    best = Some((ci, value));
                }
            }
            let Some((ci, _)) = best.or(relaxed) else { break };
            alive[ci] = false;
            let e = &self.taxonomy.entries[cands[ci].idx];
            *type_counts.entry(e.kind.clone()).or_insert(0) += 1;
            last_kind = e.kind.clone();
            for (cj, other) in cands.iter().enumerate() {
                if alive[cj] {
                    let s = cosine_sim(&e.centered, &self.taxonomy.entries[other.idx].centered);
                    if s > max_sim[cj] {
                        max_sim[cj] = s;
                    }
                }
            }
            let fresh: Vec<u32> = cands[ci].hits.iter().copied().filter(|id| !used.contains(id)).collect();
            // Reserve ids now so later rows cannot claim them
            fresh.iter().take(ROW_SIZE * 2).for_each(|id| { used.insert(*id); });
            chosen.push((cands[ci].idx, fresh));
        }
        let t_select = t0.elapsed();

        // 4. Fill rows: MMR over each row's own candidates
        let mut state = PageState::new();
        for (idx, fresh) in chosen.iter() {
            let e = &self.taxonomy.entries[*idx];
            let cand_vecs: Vec<(u32, Vec<f32>)> = fresh
                .iter()
                .take(ROW_SIZE * 2)
                .filter_map(|id| {
                    let i = self.vectors.index_of(*id)?;
                    Some((*id, self.vectors.get_f32(i)))
                })
                .collect();
            let picked = greedy_mmr(&e.embedding, &cand_vecs, ROW_SIZE, MMR_LAMBDA);
            let items: Vec<RowItem> = picked
                .iter()
                .filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog))
                .collect();
            if items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }
            state.record_thematic(e.kind.clone(), e.embedding.clone());
            rows.push(Row {
                title: clamp_words(&e.title, 5),
                archetype: "thematic".to_string(),
                kind: e.kind.clone(),
                items,
            });
        }
        let t_fill = t0.elapsed();

        // 5. Taste rows near the top: Top Picks and Because You Watched
        if has_taste {
            let mut extra: Vec<Row> = Vec::new();
            if let Some(byw) = self.build_byw_row(&state).await {
                extra.push(byw);
            }
            let hits = self.retrieve_candidates(&self.taste, ANN_K).await;
            let cand_vecs: Vec<(u32, Vec<f32>)> = hits
                .iter()
                .filter_map(|(id, _)| self.catalog.get(id))
                .filter(|i| self.matches_filters(i) && passes_member_quality(i))
                .take(ROW_SIZE * 3)
                .filter_map(|i| {
                    let idx = self.vectors.index_of(i.id)?;
                    Some((i.id, self.vectors.get_f32(idx)))
                })
                .collect();
            let picked = greedy_mmr(&self.taste, &cand_vecs, ROW_SIZE, MMR_LAMBDA);
            let items: Vec<RowItem> = picked
                .iter()
                .filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog))
                .collect();
            if items.len() >= COHERENCE_MIN_ITEMS {
                extra.insert(0, Row {
                    title: "Top Picks for You".to_string(),
                    archetype: "top_picks".to_string(),
                    kind: String::new(),
                    items,
                });
            }
            // After the trending row, before the thematic rows
            let at = rows.len().min(1);
            for (k, r) in extra.into_iter().enumerate() {
                rows.insert(at + k, r);
            }
        }
        tracing::debug!(
            "entry page: {} eligible, {} retrieved, {} rows chosen of {} | score {:?} retrieve {:?} select {:?} fill {:?} taste-rows {:?}",
            n_eligible, cands.len(), chosen.len(), want,
            t_score, t_retrieve - t_score, t_select - t_retrieve, t_fill - t_select, t0.elapsed() - t_fill
        );

        rows
    }

    // ── Genre pages ───────────────────────────────────────────────────────────

    /// Rows for a genre filtered page. The global seeds hold too few titles of any
    /// single genre, so cluster the genre's own best titles on the fly instead
    /// (farthest first init, a few Lloyd rounds) to get sub-topics inside the genre.
    fn build_genre_page(&self) -> Vec<Row> {
        const POOL: usize = 3000;
        const MIN_CLUSTER: usize = 8;

        let mut rows: Vec<Row> = Vec::new();
        if let Some(r) = self.build_trending_row() {
            rows.push(r);
        }

        let mut pool: Vec<&CatalogItem> = self
            .catalog
            .values()
            .filter(|i| self.matches_filters(i) && passes_member_quality(i))
            .collect();
        pool.sort_unstable_by(|a, b| {
            bayesian_quality_score(b.vote_average, b.popularity)
                .partial_cmp(&bayesian_quality_score(a.vote_average, a.popularity))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pool.truncate(POOL);

        let items: Vec<(&CatalogItem, Vec<f32>)> = pool
            .into_iter()
            .filter_map(|i| {
                let idx = self.vectors.index_of(i.id)?;
                if self.vectors.is_zero_row(idx) {
                    return None;
                }
                Some((i, self.vectors.get_f32(idx)))
            })
            .collect();
        if items.len() < COHERENCE_MIN_ITEMS {
            return rows;
        }

        let k = self.target_rows.min(items.len() / MIN_CLUSTER).max(1);

        // Farthest first initialisation
        let mut centers: Vec<Vec<f32>> = vec![items[0].1.clone()];
        let mut best_sim: Vec<f32> = items.iter().map(|(_, v)| cosine_sim(v, &centers[0])).collect();
        while centers.len() < k {
            let far = best_sim
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let c = items[far].1.clone();
            for (i, (_, v)) in items.iter().enumerate() {
                let s = cosine_sim(v, &c);
                if s > best_sim[i] {
                    best_sim[i] = s;
                }
            }
            centers.push(c);
        }

        let assign_all = |centers: &[Vec<f32>]| -> Vec<(usize, f32)> {
            items
                .par_iter()
                .map(|(_, v)| {
                    centers
                        .iter()
                        .enumerate()
                        .map(|(c, cv)| (c, cosine_sim(v, cv)))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or((0, 0.0))
                })
                .collect()
        };

        let mut assign = assign_all(&centers);
        for _ in 0..6 {
            let mut sums = vec![vec![0.0f32; DIMS]; k];
            for (i, &(c, _)) in assign.iter().enumerate() {
                for (a, x) in sums[c].iter_mut().zip(items[i].1.iter()) {
                    *a += x;
                }
            }
            for (c, sum) in sums.into_iter().enumerate() {
                let mut sum = sum;
                if sum.iter().any(|x| x.abs() > 1e-9) {
                    l2_normalize(&mut sum);
                    centers[c] = sum;
                }
            }
            assign = assign_all(&centers);
        }

        // Order clusters: taste alignment when there is a taste vector, else size
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let mut order: Vec<(usize, f32)> = (0..k)
            .map(|c| {
                let size = assign.iter().filter(|(a, _)| *a == c).count();
                let key = if has_taste { cosine_sim(&self.taste, &centers[c]) } else { size as f32 };
                (c, key)
            })
            .collect();
        order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let first_thematic = rows.len();
        let mut state = PageState::new();
        let mut placed: Vec<usize> = Vec::new();
        for (c, _) in order {
            let mut members: Vec<(usize, f32)> = assign
                .iter()
                .enumerate()
                .filter(|(_, (a, _))| *a == c)
                .map(|(i, (_, s))| (i, *s))
                .collect();
            if members.len() < MIN_CLUSTER {
                continue;
            }
            members.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            members.truncate(40);

            let candidates: Vec<(u32, Vec<f32>)> = members
                .iter()
                .map(|&(i, _)| (items[i].0.id, items[i].1.clone()))
                .collect();
            let picked = greedy_mmr(&centers[c], &candidates, ROW_SIZE, MMR_LAMBDA);
            let row_items: Vec<RowItem> = picked
                .iter()
                .filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog))
                .collect();
            if row_items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }
            state.record_thematic(self.genre.unwrap_or("").to_string(), centers[c].clone());
            rows.push(Row { title: "__PENDING__".to_string(), archetype: "thematic".to_string(), kind: String::new(), items: row_items });
            placed.push(c);
        }

        let genre = self.genre.unwrap_or("").to_string();
        for (ri, &c) in placed.iter().enumerate() {
            let row_idx = first_thematic + ri;
            if row_idx >= rows.len() {
                break;
            }
            let era = dominant_era(&rows[row_idx].items);
            let second = second_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
                &genre,
            );
            let candidates =
                derive_title(
                    &centers[c], &genre, second.as_deref(), era.as_deref(),
                    rows[row_idx].items.first().map(|i| i.title.as_str()),
                    None, self.mood_vault,
                );
            let title = state.claim_title(&candidates);
            rows[row_idx].title = clamp_words(&title, 4);
        }

        rows
    }

    // ── Row builders ──────────────────────────────────────────────────────────

    fn build_trending_row(&self) -> Option<Row> {
        let mut items: Vec<&CatalogItem> = self
            .catalog
            .values()
            .filter(|item| self.matches_filters(item))
            .collect();
        // Only the top ROW_SIZE are needed: select them in linear time instead of sorting the whole catalog
        let by_pop_desc = |a: &&CatalogItem, b: &&CatalogItem| {
            b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal)
        };
        if items.len() > ROW_SIZE {
            items.select_nth_unstable_by(ROW_SIZE - 1, by_pop_desc);
            items.truncate(ROW_SIZE);
        }
        items.sort_unstable_by(by_pop_desc);
        let row_items: Vec<RowItem> = items
            .into_iter()
            .map(RowItem::from_catalog)
            .collect();

        if row_items.is_empty() {
            return None;
        }

        let title = if self.surface == "tv" {
            "Top 10 Series Today"
        } else {
            "Top 10 Today"
        };

        Some(Row {
            title:     title.to_string(),
            archetype: "trending".to_string(),
            kind:      String::new(),
            items:     row_items,
        })
    }

    /// Build a "Because You Watched X" row using HNSW retrieval.
    /// Anchor must pass quality gate AND be thematically distinct from all placed rows.
    async fn build_byw_row(&self, state: &PageState) -> Option<Row> {
        // Find a high-quality anchor that is distinct from placed row centroids
        let candidates = self.retrieve_candidates(&self.taste, ANN_K).await;

        let anchor_id = candidates.iter().find_map(|(id, _)| {
            let item = self.catalog.get(id)?;
            if !self.matches_filters(item) || item.vote_average < 7.2 || item.popularity < 40.0 || item.overview.chars().count() < 80 {
                return None;
            }
            let idx = self.vectors.index_of(*id)?;
            let vec = self.vectors.get_f32(idx);
            // Must be distinct from all placed thematic row centroids
            let too_close = state.placed_centroids.iter().any(|c| cosine_sim(&vec, c) > BYW_MAX_CENTROID_SIM);
            if too_close { return None; }
            Some(*id)
        })?;

        let anchor_item = self.catalog.get(&anchor_id)?;
        let anchor_idx = self.vectors.index_of(anchor_id)?;
        let anchor_vec = self.vectors.get_f32(anchor_idx);

        // Retrieve similar items for the row
        let similar = self.retrieve_candidates(&anchor_vec, ANN_K / 2).await;
        let row_items: Vec<RowItem> = similar
            .iter()
            .filter(|(id, _)| *id != anchor_id)
            .filter_map(|(id, _)| self.catalog.get(id).filter(|i| self.matches_filters(i)))
            .take(ROW_SIZE)
            .map(RowItem::from_catalog)
            .collect();

        if row_items.len() < COHERENCE_MIN_ITEMS {
            return None;
        }

        Some(Row {
            title:     byw_title(&anchor_item.title),
            archetype: "byw".to_string(),
            kind:      String::new(),
            items:     row_items,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Prior weight per row type, mirroring the page policy type mix in
/// ai-engine/config/row_architecture.json (vibe and theme rows dominate).
fn kind_weight(kind: &str) -> f32 {
    match kind {
        "vibe" => 1.0,
        "theme" => 0.85,
        "combo" => 0.7,
        "country" => 0.5,
        "setting" => 0.4,
        "decade" | "character" | "audience" => 0.35,
        "animation" | "style" | "editorial" => 0.3,
        "identity" => 0.2,
        _ => 0.3,
    }
}

/// Stable value in (0, 1) per (session, entry): exploration without flicker.
fn entry_noise(session: &str, idx: usize) -> f32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    session.hash(&mut h);
    idx.hash(&mut h);
    ((h.finish() >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Cheap quality gate for cluster members: drops obscure, unrated and stub entries.
pub fn passes_member_quality(item: &CatalogItem) -> bool {
    // The catalog has no vote_count, so popularity is the proxy for a reliable rating.
    // Measured pool: popularity >= 2 with rating >= 6.0 keeps about 38k titles of 700k.
    item.vote_average >= 6.0
        && item.popularity >= 2.0
        && item.overview_chars() >= 60
        && item.poster_path.as_deref().is_some_and(|p| !p.is_empty())
}

/// Second most common genre, if any.
fn second_genre<'a>(genres_iter: impl Iterator<Item = &'a [String]>, first: &str) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for genres in genres_iter {
        for g in genres {
            if g != first && !g.contains(first) && !first.contains(g.as_str()) {
                *counts.entry(g.as_str()).or_insert(0) += 1;
            }
        }
    }
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(g, _)| g.to_string())
}

/// Decade label ("90s", "2010s") when at least half of the items share a decade.
fn dominant_era(items: &[RowItem]) -> Option<String> {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    let mut total = 0usize;
    for it in items {
        if let Some(y) = it.release_year {
            *counts.entry(y / 10 * 10).or_insert(0) += 1;
            total += 1;
        }
    }
    let (decade, n) = counts.into_iter().max_by_key(|(_, c)| *c)?;
    if total == 0 || n * 2 < total {
        return None;
    }
    Some(if decade >= 2000 { format!("{decade}s") } else { format!("{}s", decade % 100) })
}


/// Dominant genre from an iterator of genre slices (majority vote).
fn majority_genre<'a>(genres_iter: impl Iterator<Item = &'a [String]>) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for genres in genres_iter {
        for genre in genres {
            *counts.entry(genre.clone()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(g, _)| g)
        .unwrap_or_else(|| "Drama".to_string())
}
