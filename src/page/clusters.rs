//! Rows made from the cluster seeds of the whole catalog (the default home, films and series page).
//!
//! Pages of a children's profile, of a genre or a language have builders of their own (kids home, genre.rs, entry.rs) and are routed to them first. The default page walks the
//! cluster seeds from the one that suits the session best: a seed becomes a row when it passes the gates below, and the rows are named together at the end so no two rows share a name.
use super::*;
use crate::cluster::ClusterSeed;

/// Why a seed gave no row (counted for the debug log of a page build).
#[derive(Clone, Copy)]
enum Skip {
    /// Too like a row already placed
    TooSimilar,
    /// No title of the seed may be shown on this page
    NoMembers,
    /// The genre rules of the page (documentary rows capped, no genre twice in a short window)
    GenreRule,
    /// The titles that survive the coherence gate are too few
    TooFew,
    /// Anything else (no vectors, too few distinct titles)
    Other,
}

/// A row taken from a seed, before it is named.
struct SeedRow {
    items: Vec<RowItem>,
    centroid: Vec<f32>,
    dominant_genre: String,
    nonscripted: bool,
}

/// The rows placed so far. `centroids`, `row_index` and `seed_index` are parallel to each other and cover the seed rows only (the Top 10 row and the Because You Watched row are not in them).
#[derive(Default)]
struct Placed {
    rows: Vec<Row>,
    centroids: Vec<Vec<f32>>,
    row_index: Vec<usize>,
    seed_index: Vec<usize>,
}

/// Counts for the debug log of one page build.
#[derive(Default)]
struct SeedStats {
    visited: u32,
    rows: u32,
    similar: u32,
    empty: u32,
    genre: u32,
    few: u32,
}

impl<'a> PageContext<'a> {
    pub(crate) async fn build_page_rows_inner(&self) -> Vec<Row> {
        match self.own_builder_page().await {
            Some(rows) => rows,
            None => self.cluster_page().await,
        }
    }

    /// The pages that have a builder of their own: a children's profile's home, a genre or language page, and (when asked for) the page made wholly from row definitions.
    async fn own_builder_page(&self) -> Option<Vec<Row>> {
        // A children's profile gets a page built from the titles it may see (clustered like a genre page) instead of the global topic
        // seeds, which are mostly adult topics with a few allowed titles left in each.
        let kids_home = self.maturity.is_some_and(|m| m.is_kids()) && self.profiles.has_profiles();
        if kids_home && self.language.is_none() && self.animation.is_none() {
            if self.kids_entries {
                // a genre page keeps the rows that still have enough titles of the genre; a page that stays thin falls back to the clustered genre page
                let rows = self.build_entry_page().await;
                if rows.len() > if self.genre.is_none() { 4 } else { 8 } {
                    return Some(rows);
                }
            }
            if self.genre.is_none() {
                return Some(self.build_kids_home());
            }
        }
        if self.genre.is_some() || self.language.is_some() || kids_home {
            let rows = self.build_genre_page();
            return Some(self.fill_thin_genre_page(rows).await);
        }
        // The page built wholly from row definitions is off unless asked for (STREAMS_ENTRY_PAGES=1): measured 2026-10-13, its row names are vaguer and it fills
        // fewer rows than the topic rows; the vocabulary is used for seasonal rows (build_seasonal_rows).
        if std::env::var("STREAMS_ENTRY_PAGES").is_ok() && self.taxonomy.entries.len() >= MIN_TAXONOMY_FOR_ENTRY_ROWS {
            let rows = self.build_entry_page().await;
            if rows.len() > 2 {
                return Some(rows);
            }
        }
        None
    }

    /// The default page: the Top 10 row, then one row per accepted seed, a Because You Watched row for a profile with taste, and names for all of them.
    async fn cluster_page(&self) -> Vec<Row> {
        let mut state = PageState::new();
        let mut placed = Placed::default();
        if let Some(row) = self.build_trending_row() {
            placed.rows.push(row);
        }

        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let started = std::time::Instant::now();
        let mut stats = SeedStats::default();
        for seed_idx in self.seed_order(has_taste) {
            if placed.rows.len() > self.target_rows {
                break; // +1 for trending row
            }
            stats.visited += 1;
            let seed = &self.seeds.seeds[seed_idx];
            match self.seed_row(seed, &placed.centroids, &state).await {
                Ok(found) => {
                    stats.rows += 1;
                    state.used_ids.extend(found.items.iter().map(|r| r.id));
                    if found.nonscripted {
                        state.nonscripted_rows += 1;
                    }
                    placed.row_index.push(placed.rows.len());
                    placed.seed_index.push(seed_idx);
                    placed.centroids.push(found.centroid.clone());
                    state.record_thematic(found.dominant_genre, found.centroid);
                    placed.rows.push(Row {
                        title:     "__PENDING__".to_string(), // named together with the other seed rows below
                        archetype: "thematic".to_string(),
                        kind:      String::new(),
                        items:     found.items, meta: None,
                    });
                }
                Err(Skip::TooSimilar) => stats.similar += 1,
                Err(Skip::NoMembers) => stats.empty += 1,
                Err(Skip::GenreRule) => stats.genre += 1,
                Err(Skip::TooFew) => stats.few += 1,
                Err(Skip::Other) => {}
            }
        }
        tracing::debug!(
            "seed loop: {:.0} ms, visited {}, too similar {}, no members {}, genre rules {}, too few titles {}, rows {}",
            started.elapsed().as_secs_f64() * 1000.0, stats.visited, stats.similar, stats.empty, stats.genre, stats.few, stats.rows
        );

        // A Because You Watched row only for a profile with taste: its anchor must be distinct from all placed seed rows (cosine < BYW_MAX_CENTROID_SIM).
        if has_taste {
            if let Some(byw) = self.build_byw_row(&state).await {
                placed.rows.push(byw);
            }
        }
        self.name_seed_rows(&mut placed, &mut state);
        placed.rows
    }

    /// The seeds in the order they are tried: by cosine to the taste vector (all equal for a cold start, so the order of the file), moved a little by the session's noise.
    fn seed_order(&self, has_taste: bool) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = self
            .seeds
            .seeds
            .iter()
            .enumerate()
            .map(|(i, seed)| {
                let base = if has_taste { cosine_sim(&self.taste, &seed.centroid) } else { 1.0 };
                // exploration: a session reorders the seeds a little (stable for the session)
                let noise = if self.session.is_empty() || !safety_nets() { 0.0 } else { SEED_NOISE * entry_noise(self.session, i) };
                (base + noise, i)
            })
            .collect();
        // by relevance descending (the order of the file for equal scores)
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().map(|(_, i)| i).collect()
    }

    /// The row of one seed, or the reason it gives none. The gates run from the cheapest (no vectors read) to the dearest.
    async fn seed_row(&self, seed: &ClusterSeed, placed_centroids: &[Vec<f32>], state: &PageState) -> Result<SeedRow, Skip> {
        // inter anchor gate: too like a row already placed
        if placed_centroids.iter().any(|c| cosine_sim(&seed.centroid, c) > MMD_MAX_INTER_SIM) {
            return Err(Skip::TooSimilar);
        }
        // members that pass the page's filters and the quality gates
        let cand_items: Vec<&CatalogItem> = seed
            .member_ids
            .iter()
            .filter_map(|id| self.catalog.get(id))
            .filter(|item| self.matches_filters(item) && self.member_ok(item) && self.reliable(item))
            .collect();
        if cand_items.is_empty() {
            return Err(Skip::NoMembers);
        }
        let dominant_genre = majority_genre(cand_items.iter().map(|item| item.genres.as_slice()));
        // documentary and reality rows are capped on default pages
        let nonscripted = dominant_genre == "Documentary" || dominant_genre == "Reality";
        if nonscripted && self.genre.is_none() && safety_nets() && state.nonscripted_rows >= NONSCRIPTED_ROW_CAP {
            return Err(Skip::GenreRule);
        }
        // no same genre in the last GENRE_WINDOW rows (checked before any vector work so rejected seeds stay cheap); a genre page is one genre by design
        if self.genre.is_none() && !state.can_place_genre(&dominant_genre) {
            return Err(Skip::GenreRule);
        }

        let members = self.members_with_vectors(cand_items);
        if members.is_empty() {
            return Err(Skip::Other);
        }
        let (fil_ids, fil_vecs) = self.coherent_members(&seed.centroid, &members).ok_or(Skip::TooFew)?;
        let centroid = sphere_mean(&fil_vecs);

        let mut items: Vec<RowItem> = self.diverse_titles(&fil_ids, ROW_SIZE);
        if items.len() < COHERENCE_MIN_ITEMS {
            return Err(Skip::Other);
        }
        // mix movies and series where the theme has both (surfaces that show both types)
        if self.surface == "both" {
            let majority_tv = items.iter().filter(|r| r.media_type == "tv").count() * 2 >= items.len();
            let other = if majority_tv { "movie" } else { "tv" };
            let mut exclude = state.used_ids.clone();
            exclude.extend(items.iter().map(|r| r.id));
            let cands = self.other_type_neighbours(&centroid, other, &exclude).await;
            self.blend_other_type(&mut items, &centroid, other, cands);
        }
        Ok(SeedRow { items, centroid, dominant_genre, nonscripted })
    }

    /// The best quality members of a seed (at most MEMBER_POOL, a session leaves some out so two sessions differ) with their vectors.
    fn members_with_vectors<'b>(&self, mut cand_items: Vec<&'b CatalogItem>) -> Vec<(&'b CatalogItem, Vec<f32>)> {
        crate::quality::sort_by_quality(&mut cand_items);
        cand_items.truncate(MEMBER_POOL);
        if !self.session.is_empty() && safety_nets() && cand_items.len() > 60 {
            let kept: Vec<&CatalogItem> = cand_items.iter().copied().filter(|it| entry_noise(self.session, it.id as usize) >= MEMBER_DROP).collect();
            if kept.len() >= 45 {
                cand_items = kept;
            }
        }
        cand_items
            .into_iter()
            .filter_map(|item| {
                let idx = self.vectors.index_of(item.id)?;
                Some((item, self.vectors.get_f32(idx)))
            })
            .collect()
    }

    /// Diversity (MMR) over the members, then the coherence gate: outliers by distance to the medoid are dropped, and a weak cluster needs more titles to pass.
    /// None when too few titles are left.
    fn coherent_members(&self, centroid: &[f32], members: &[(&CatalogItem, Vec<f32>)]) -> Option<(Vec<u32>, Vec<Vec<f32>>)> {
        let candidates: Vec<(u32, Vec<f32>)> = members.iter().map(|(item, vec)| (item.id, vec.clone())).collect();
        let selected_ids = greedy_mmr(centroid, &candidates, ROW_SIZE * 2, MMR_LAMBDA);
        let (sel_ids, sel_vecs): (Vec<u32>, Vec<Vec<f32>>) = selected_ids
            .iter()
            .filter_map(|id| {
                let idx = self.vectors.index_of(*id)?;
                Some((*id, self.vectors.get_f32(idx)))
            })
            .unzip();
        let (fil_ids, fil_vecs) = filter_outliers(&sel_ids, &sel_vecs, COHERENCE_OK);
        let coherence = mean_pairwise_cosine(&fil_vecs);
        let min_items = if coherence < COHERENCE_WEAK { COHERENCE_MIN_ITEMS + 4 } else { COHERENCE_MIN_ITEMS };
        (fil_ids.len() >= min_items).then_some((fil_ids, fil_vecs))
    }

    /// Names every seed row at once. Row definitions are matched to the rows together (Hungarian assignment), so no two rows take the same one; the name written for the
    /// group comes first when it keeps its promise for the titles shown, the names derived from the titles follow as fallbacks.
    fn name_seed_rows(&self, placed: &mut Placed, state: &mut PageState) {
        let taxonomy_titles = self.taxonomy.hungarian_assign(&placed.centroids, self.surface, TAXONOMY_THRESHOLD);
        for (ti, &row_idx) in placed.row_index.iter().enumerate() {
            let rows = &mut placed.rows;
            let dominant_genre = majority_genre(rows[row_idx].items.iter().map(|item| item.genres.as_slice()));
            let taxonomy_title = taxonomy_titles.get(ti).and_then(|t| t.clone());
            let era = dominant_era(&rows[row_idx].items);
            let second = second_genre(rows[row_idx].items.iter().map(|item| item.genres.as_slice()), &dominant_genre);
            let has_taxonomy = taxonomy_title.is_some();
            let mut derived = derive_title(&dominant_genre, second.as_deref(), era.as_deref(), rows[row_idx].items.first().map(|i| i.title.as_str()), taxonomy_title);
            let row_catalog: Vec<&CatalogItem> = rows[row_idx].items.iter().filter_map(|i| self.catalog.get(&i.id)).collect();
            let mut candidates = if has_taxonomy && !derived.is_empty() { vec![derived.remove(0)] } else { Vec::new() };
            if let Some(seed) = placed.seed_index.get(ti).and_then(|i| self.seeds.seeds.get(*i)) {
                if !seed.name.is_empty() && crate::row_label::name_fits(&seed.name, &row_catalog) && !entry_names_other_kind(&seed.name, self.surface) {
                    candidates.insert(0, seed.name.clone());
                }
            }
            candidates.extend(crate::row_label::candidates(&row_catalog));
            candidates.extend(derived);
            let candidates = crate::row_label::mark_animated(candidates, &row_catalog);
            let title = state.claim_title(&candidates);
            rows[row_idx].title = clamp_words(&title, 6);
        }
    }
}
