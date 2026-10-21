//! Rows made from the cluster seeds of the whole catalog (the default home, films and series page).
use super::*;

impl<'a> PageContext<'a> {
    pub(crate) async fn build_page_rows_inner(&self) -> Vec<Row> {
        // A children's profile gets a page built from the titles it may see (clustered like a genre page) instead of the global topic
        // seeds, which are mostly adult topics with a few allowed titles left in each.
        let kids_home = self.maturity.is_some_and(|m| m.is_kids()) && self.profiles.has_profiles();
        if kids_home && self.language.is_none() && self.animation.is_none() {
            if self.kids_entries {
                // a genre page keeps the rows that still have enough titles of the genre; a page that stays thin falls back to the clustered genre page
                let rows = self.build_entry_page().await;
                if rows.len() > if self.genre.is_none() { 4 } else { 8 } {
                    return rows;
                }
            }
            if self.genre.is_none() {
                return self.build_kids_home();
            }
        }
        if self.genre.is_some() || self.language.is_some() || kids_home {
            let rows = self.build_genre_page();
            return self.fill_thin_genre_page(rows).await;
        }
        // The page built wholly from row definitions is off unless asked for (STREAMS_ENTRY_PAGES=1): measured 2026-10-13, its row names are vaguer and it fills
        // fewer rows than the topic rows; the vocabulary is used for seasonal rows (build_seasonal_rows).
        if std::env::var("STREAMS_ENTRY_PAGES").is_ok() && self.taxonomy.entries.len() >= MIN_TAXONOMY_FOR_ENTRY_ROWS {
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
        let mut thematic_seed_idx: Vec<usize> = Vec::new(); // the seed each thematic row came from, parallel to thematic_row_indices

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
                let base = if has_taste {
                    cosine_sim(&self.taste, &seed.centroid)
                } else {
                    1.0
                };
                // exploration: a session reorders the seeds a little (stable for the session)
                let noise = if self.session.is_empty() || !safety_nets() { 0.0 } else { SEED_NOISE * entry_noise(self.session, i) };
                (base + noise, i)
            })
            .collect();

        // Sort by relevance descending (preserves MMD order for equal scores)
        scored_seeds.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // ── Thematic rows from cluster seeds ──────────────────────────────────
        let t_seed_loop = std::time::Instant::now();
        let (mut n_visit, mut n_similar, mut n_empty, mut n_genre, mut n_mmr, mut n_items, mut n_row) = (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
        for (_, seed_idx) in &scored_seeds {
            if rows.len() > self.target_rows {
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
                .filter(|item| self.matches_filters(item) && self.member_ok(item) && self.reliable(item))
                .collect();

            if cand_items.is_empty() {
                n_empty += 1;
                continue;
            }

            // Dominant genre (majority vote)
            let dominant_genre =
                majority_genre(cand_items.iter().map(|item| item.genres.as_slice()));

            // Documentary and reality rows are capped on default pages
            let nonscripted_row = dominant_genre == "Documentary" || dominant_genre == "Reality";
            if nonscripted_row && self.genre.is_none() && safety_nets() && state.nonscripted_rows >= NONSCRIPTED_ROW_CAP {
                n_genre += 1;
                continue;
            }

            // Genre bitmask: no same genre in last GENRE_WINDOW rows.
            // Checked before any vector work so rejected seeds stay cheap.
            // A genre filtered page is one genre by design, so the rule does not apply.
            if self.genre.is_none() && !state.can_place_genre(&dominant_genre) {
                n_genre += 1;
                continue;
            }

            // Best quality members only, then fetch their vectors
            crate::quality::sort_by_quality(&mut cand_items);
            cand_items.truncate(MEMBER_POOL);
            if !self.session.is_empty() && safety_nets() && cand_items.len() > 60 {
                let kept: Vec<&CatalogItem> = cand_items
                    .iter()
                    .copied()
                    .filter(|it| entry_noise(self.session, it.id as usize) >= MEMBER_DROP)
                    .collect();
                if kept.len() >= 45 {
                    cand_items = kept;
                }
            }
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
            let row_items: Vec<RowItem> = self.diverse_titles(&fil_ids, ROW_SIZE);

            if row_items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }

            // Mix movies and series where the theme has both (surfaces that show both types)
            let mut row_items = row_items;
            if self.surface == "both" {
                let majority_tv = row_items.iter().filter(|r| r.media_type == "tv").count() * 2 >= row_items.len();
                let other = if majority_tv { "movie" } else { "tv" };
                let mut exclude = state.used_ids.clone();
                exclude.extend(row_items.iter().map(|r| r.id));
                let cands = self.other_type_neighbours(&centroid, other, &exclude).await;
                self.blend_other_type(&mut row_items, &centroid, other, cands);
            }
            state.used_ids.extend(row_items.iter().map(|r| r.id));
            if nonscripted_row {
                state.nonscripted_rows += 1;
            }

            // Record state
            n_row += 1;
            let row_index = rows.len();
            thematic_row_indices.push(row_index);
            thematic_seed_idx.push(*seed_idx);
            thematic_centroids.push(centroid.clone());
            state.record_thematic(dominant_genre, centroid);

            rows.push(Row {
                title:     "__PENDING__".to_string(), // replaced after Hungarian below
                archetype: "thematic".to_string(),
                kind:      String::new(),
                items:     row_items, meta: None,
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
            let dominant_genre = majority_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
            );
            let taxonomy_title = taxonomy_titles.get(ti).and_then(|t| t.clone());
            let era = dominant_era(&rows[row_idx].items);
            let second = second_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
                &dominant_genre,
            );
            let has_taxonomy = taxonomy_title.is_some();
            let mut derived = derive_title(
                &dominant_genre, second.as_deref(), era.as_deref(),
                rows[row_idx].items.first().map(|i| i.title.as_str()),
                taxonomy_title,
            );
            let row_catalog: Vec<&CatalogItem> = rows[row_idx].items.iter().filter_map(|i| self.catalog.get(&i.id)).collect();
            let mut candidates = if has_taxonomy && !derived.is_empty() { vec![derived.remove(0)] } else { Vec::new() };
            // the name written for the group comes first; the titles derived from the displayed items follow as fallbacks
            if let Some(seed) = thematic_seed_idx.get(ti).and_then(|i| self.seeds.seeds.get(*i)) {
                if !seed.name.is_empty() && crate::row_label::name_fits(&seed.name, &row_catalog) && !entry_names_other_kind(&seed.name, self.surface) { candidates.insert(0, seed.name.clone()); }
            }
            candidates.extend(crate::row_label::candidates(&row_catalog));
            candidates.extend(derived);
            let candidates = crate::row_label::mark_animated(candidates, &row_catalog);
            let title = state.claim_title(&candidates);
            rows[row_idx].title = clamp_words(&title, 6);
        }

        rows
    }
}
