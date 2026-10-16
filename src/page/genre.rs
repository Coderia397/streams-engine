use super::*;

impl<'a> PageContext<'a> {
    /// A genre page with fewer rows than the floor is topped up with rows of neighbouring titles: the titles closest to the mean of what the page shows (any genre, same age and
    /// quality rules, nothing already shown), grouped into rows by the same builder and named by their own groups. Netflix fills thin genre pages the same way.
    pub(super) async fn fill_thin_genre_page(&self, mut rows: Vec<Row>) -> Vec<Row> {
        let count = |rows: &[Row]| rows.iter().filter(|r| r.archetype != "trending").count();
        if self.genre.is_none() || self.language.is_some() || self.animation.is_some() || count(&rows) == 0 || count(&rows) >= THIN_PAGE_FLOOR {
            return rows;
        }
        let mut titles: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
        // up to three passes, each looking further out around the mean of everything shown so far
        for reach in [3000usize, 9000, 25000] {
            let own = count(&rows);
            if own >= THIN_PAGE_FLOOR { break; }
            let mut mean = vec![0f32; DIMS];
            let mut n = 0usize;
            for it in rows.iter().flat_map(|r| &r.items) {
                if let Some(ix) = self.vectors.index_of(it.id) {
                    for (a, x) in mean.iter_mut().zip(self.vectors.row(ix)) { *a += x.to_f32(); }
                    n += 1;
                }
            }
            if n == 0 { break; }
            l2_normalize(&mut mean);
            let keys: Vec<u32> = {
                let guard = self.hnsw.read().await;
                self.retrieve_sync(&guard, &mean, reach).into_iter().map(|(k, _)| k).collect()
            };
            let used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
            let ctx = PageContext { genre: None, genre_spec: None, target_rows: THIN_PAGE_FLOOR - own + 4, fill_pool: Some(&keys), skip: Some(&used), ..self.clone() };
            let mut added = 0;
            for r in ctx.build_genre_page_plain() {
                if count(&rows) >= THIN_PAGE_FLOOR { break; }
                if r.archetype == "trending" || !titles.insert(r.title.to_lowercase()) { continue; }
                rows.push(Row { kind: "related".to_string(), ..r });
                added += 1;
            }
            if added == 0 && reach >= 25000 { break; }
        }
        rows
    }

    /// Genre pages: the Moods page and the event pages (Halloween, Pride) are made of their own rows, topped up with the regular genre rows when the event has fewer sub groups than the page has rows.
    pub(super) fn build_genre_page(&self) -> Vec<Row> {
        if let Some(g) = self.genre {
            let g = g.to_lowercase();
            if g == "moods" && !self.vibe.moods.is_empty() && self.language.is_none() {
                let r = self.build_moods_page();
                if r.len() >= 4 { return r; }
            }
            // any calendar event that has sub groups of titles has a page of its own (halloween, pride, christmas ...)
            let event: Option<String> = match g.as_str() { "halloween" => Some("halloween".to_string()), "pride" | "lgbtq" => Some("pride".to_string()), other if self.vibe.seasons.contains_key(other) => Some(other.to_string()), _ => None };
            if let Some(ev) = event {
                if self.language.is_none() && self.animation.is_none() {
                    let mut rows = self.build_event_page(&ev);
                    if rows.len() >= 5 {
                        if rows.len() <= self.target_rows {
                            let mut used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
                            let mut titles: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).collect();
                            for r in self.build_genre_page_plain() {
                                if rows.len() > self.target_rows { break; }
                                if r.archetype == "trending" || !titles.insert(r.title.to_lowercase()) { continue; }
                                let items: Vec<RowItem> = r.items.into_iter().filter(|i| !used.contains(&i.id)).collect();
                                if items.len() >= 8 { items.iter().for_each(|i| { used.insert(i.id); }); rows.push(Row { items, ..r }); }
                            }
                        }
                        return rows;
                    }
                }
            }
        }
        self.build_genre_page_plain()
    }

    /// Rows for a genre filtered page. The global seeds hold too few titles of any
    /// single genre, so cluster the genre's own best titles on the fly instead
    /// (farthest first init, a few Lloyd rounds) to get sub-topics inside the genre.
    pub(super) fn build_genre_page_plain(&self) -> Vec<Row> {
        const POOL: usize = 3000;
        const MIN_CLUSTER: usize = 8;

        let mut rows: Vec<Row> = Vec::new();
        if let Some(r) = self.build_trending_row() {
            rows.push(r);
        }

        // a children's home holds titles made for children; a genre page chosen by a children's profile (Comedies, Adventure) keeps every title it may see
        let kids_home = self.genre.is_none() && self.language.is_none() && self.maturity.is_some_and(|m| m.is_kids());
        let candidates: Box<dyn Iterator<Item = &CatalogItem> + '_> = match self.fill_pool {
            Some(keys) => Box::new(keys.iter().filter_map(|k| self.catalog.get(k))),
            None => Box::new(self.catalog.values()),
        };
        let mut pool: Vec<&CatalogItem> = candidates
            .filter(|i| self.member_ok(i) && self.reliable(i) && self.matches_filters(i) && self.skip.is_none_or(|s| !s.contains(&i.id)) && (!kids_home || kid_interest(self.profiles, i)) && self.animation.is_none_or(|a| a == i.genres.iter().any(|g| g == "Animation")))
            .collect();
        // ties are broken by id so a page does not depend on the hash map order of the catalog (which changes at every reload)
        crate::quality::sort_by_quality(&mut pool);
        // Variety: on a page that is not about anime, Japanese language titles fill at most a third of the pool (they are many among the best
        // rated series and would otherwise fill most rows).
        if !matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Anime)) && self.language != Some("ja") {
            let mut japanese = 0usize;
            pool.retain(|i| {
                if i.original_language == "ja" {
                    japanese += 1;
                    japanese <= POOL / 3
                } else {
                    true
                }
            });
        }
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

        // Rows from the groups of the database. Every title of the page's pool belongs to a group at each level of the hierarchy (world level 2 down to level 1, themes in between);
        // the finest level that gives enough groups of MIN_CLUSTER titles or more is used, each such group is a row. A pool too small for any level to give four rows is clustered by
        // vectors instead, and its rows are then named after the group most of their titles share.
        let items_all = items;
        let want_rows = self.target_rows.min(items_all.len() / 20).max(6);
        let mut group_mode: Option<GroupMode> = None; // (level, id) per cluster, and the item indices kept
        if !self.groups.is_empty() {
            let mut best: Option<(u8, Vec<GroupItems>)> = None;
            for level in 0..crate::groups::N_LEVELS {
                let mut by: HashMap<i32, Vec<usize>> = HashMap::new();
                for (i, (it, _)) in items_all.iter().enumerate() {
                    if let Some(g) = self.groups.get(&it.id) { let id = g.at(level); if id >= 0 { by.entry(id).or_default().push(i); } }
                }
                let mut kept: Vec<(i32, Vec<usize>)> = by.into_iter().filter(|(_, ix)| ix.len() >= MIN_CLUSTER).collect();
                kept.sort_by(|x, y| y.1.len().cmp(&x.1.len()).then(x.0.cmp(&y.0)));
                let enough = kept.len() >= want_rows;
                if best.as_ref().is_none_or(|(_, b)| kept.len() > b.len()) || enough { best = Some((level, kept)); }
                if enough { break; }
            }
            if let Some((level, kept)) = best {
                if kept.len() >= 4 {
                    let ids: Vec<(u8, i32)> = kept.iter().map(|(g, _)| (level, *g)).collect();
                    let idx: Vec<usize> = kept.into_iter().flat_map(|(_, ix)| ix).collect();
                    group_mode = Some((ids, idx));
                }
            }
        }
        let cluster_group: Vec<(u8, i32)>;
        let items: Vec<(&CatalogItem, Vec<f32>)>;
        let centers: Vec<Vec<f32>>;
        let assign: Vec<(usize, f32)>;
        let k: usize;
        if let Some((ids, idx)) = group_mode {
            // items of the kept groups, in group order; a group's centre is the mean of its members
            let mut sel: Vec<(&CatalogItem, Vec<f32>)> = Vec::with_capacity(idx.len());
            let mut cid: Vec<usize> = Vec::with_capacity(idx.len());
            let mut taken: HashMap<(u8, i32), usize> = HashMap::new();
            for (c, key) in ids.iter().enumerate() { taken.insert(*key, c); }
            for i in idx {
                let it = items_all[i].0;
                if let Some(g) = self.groups.get(&it.id) {
                    if let Some(&c) = taken.get(&(ids[0].0, g.at(ids[0].0))) { sel.push((it, items_all[i].1.clone())); cid.push(c); }
                }
            }
            let mut cs: Vec<Vec<f32>> = vec![vec![0.0f32; DIMS]; ids.len()];
            for (n, (_, v)) in sel.iter().enumerate() { for (a, x) in cs[cid[n]].iter_mut().zip(v) { *a += x; } }
            for c in cs.iter_mut() { l2_normalize(c); }
            assign = sel.iter().enumerate().map(|(n, (_, v))| (cid[n], cosine_sim(v, &cs[cid[n]]))).collect();
            k = ids.len(); centers = cs; cluster_group = ids; items = sel;
        } else {
            let items_k = items_all;
            let k_means = self.target_rows.min(items_k.len() / MIN_CLUSTER).max(1);
            let (c_, a_) = {
                let items = &items_k;
                let k = k_means;

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

                (centers, assign)
            };
            k = k_means; centers = c_; assign = a_; cluster_group = Vec::new(); items = items_k;
        }

        // Order clusters: taste alignment when there is a taste vector, else size
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let anime_page = matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Anime));
        let mut anime_count = vec![0usize; k];
        // Japanese language titles count with anime for the variety cap below (Japanese superhero and mecha shows are not all flagged as anime)
        let mut japanese_count = vec![0usize; k];
        for (i, &(c, _)) in assign.iter().enumerate() {
            if items[i].0.is_anime {
                anime_count[c] += 1;
            }
            if items[i].0.original_language == "ja" {
                japanese_count[c] += 1;
            }
        }
        let mut order: Vec<(usize, f32)> = (0..k)
            .map(|c| {
                let size = assign.iter().filter(|(a, _)| *a == c).count();
                let key = if has_taste {
                    cosine_sim(&self.taste, &centers[c])
                } else {
                    // largest clusters first, with a session jitter, and an anime cluster does not lead a page that is not about anime
                    let jitter = if self.session.is_empty() || !safety_nets() { 1.0 } else { 0.7 + 0.6 * entry_noise(self.session, c) };
                    let anime_penalty = if !anime_page && size > 0 && anime_count[c] * 10 > size * 6 { 0.5 } else { 1.0 };
                    size as f32 * jitter * anime_penalty
                };
                (c, key)
            })
            .collect();
        order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Variety: on a page that is not about anime, at most two anime clusters sit among the first rows; the rest follow the other clusters.
        if !anime_page {
            let is_anime = |c: usize| {
                let size = assign.iter().filter(|(a, _)| *a == c).count();
                size > 0 && anime_count[c].max(japanese_count[c]) * 10 > size * 4
            };
            let mut kept = 0;
            let (mut early, mut late): (Vec<ScoredIdx>, Vec<ScoredIdx>) = (Vec::new(), Vec::new());
            for entry in order.drain(..) {
                if is_anime(entry.0) {
                    kept += 1;
                    if kept <= 2 { early.push(entry) } else { late.push(entry) }
                } else {
                    early.push(entry)
                }
            }
            early.extend(late);
            order = early;
        }

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
            let picked = greedy_mmr(&centers[c], &candidates, ROW_SIZE * 2, MMR_LAMBDA);
            let row_items: Vec<RowItem> = self.diverse_titles(&picked, ROW_SIZE);
            if row_items.len() < COHERENCE_MIN_ITEMS {
                continue;
            }
            let mut row_items = row_items;
            if self.surface == "both" {
                let majority_tv = row_items.iter().filter(|r| r.media_type == "tv").count() * 2 >= row_items.len();
                let other = if majority_tv { "movie" } else { "tv" };
                let in_row: HashSet<u32> = row_items.iter().map(|r| r.id).collect();
                // rank the other type by closeness to the cluster centre without copying vectors, then copy only the best 120
                let mut scored: Vec<(f32, usize)> = items
                    .iter()
                    .enumerate()
                    .filter(|(_, (it, _))| it.media_type == other && !in_row.contains(&it.id) && !state.used_ids.contains(&it.id))
                    .map(|(i, (_, v))| (cosine_sim(v, &centers[c]), i))
                    .collect();
                scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(120);
                let cands: Vec<(u32, Vec<f32>)> = scored.into_iter().map(|(_, i)| (items[i].0.id, items[i].1.clone())).collect();
                self.blend_other_type(&mut row_items, &centers[c], other, cands);
            }
            state.used_ids.extend(row_items.iter().map(|r| r.id));
            state.record_thematic(self.genre.unwrap_or("").to_string(), centers[c].clone());
            rows.push(Row { title: "__PENDING__".to_string(), archetype: "thematic".to_string(), kind: String::new(), items: row_items, meta: None });
            placed.push(c);
        }

        let _genre = self.genre.unwrap_or("").to_string();
        for (ri, &c) in placed.iter().enumerate() {
            let row_idx = first_thematic + ri;
            if row_idx >= rows.len() {
                break;
            }
            let era = dominant_era(&rows[row_idx].items);
            // name the row after what is in it, not after the page genre
            let dom = majority_genre(rows[row_idx].items.iter().map(|item| item.genres.as_slice()));
            let second = second_genre(
                rows[row_idx].items.iter().map(|item| item.genres.as_slice()),
                &dom,
            );
            let row_catalog: Vec<&CatalogItem> = rows[row_idx].items.iter().filter_map(|i| self.catalog.get(&i.id)).collect();
            // concrete names from the catalog's own keywords come first; then the theme most of the row's titles share (from their profiles, when at
            // least half of them have one and half of those carry it); then the older generic names
            let mut candidates = crate::row_label::candidates(&row_catalog);
            let row_profiles: Vec<&crate::profiles::TitleProfile> = rows[row_idx].items.iter().filter_map(|i| self.profiles.get(i.id)).collect();
            if row_profiles.len() * 2 >= rows[row_idx].items.len() {
                if let Some((theme, _)) = crate::profiles::row_theme(&row_profiles, 0.5).or_else(|| self.profiles.row_theme_lift(&row_profiles, 0.3, 8.0)) {
                    candidates.extend(crate::row_label::theme_title(&theme, &row_catalog));
                }
                if let Some(tone) = self.profiles.row_tone(&row_profiles, 0.35, 2.5) {
                    candidates.extend(crate::row_label::tone_title(tone, &row_catalog));
                }
            }
            candidates.extend(derive_title(
                &dom, second.as_deref(), era.as_deref(),
                rows[row_idx].items.first().map(|i| i.title.as_str()),
                None,
            ));
            // the name of the group the row is built from, or of the group most of its titles belong to, comes first when it keeps its promise
            let group_name = cluster_group.get(c).and_then(|key| self.group_names.get(key).cloned()).or_else(|| self.majority_group_name(&row_catalog));
            if let Some(n) = group_name {
                if crate::row_label::name_fits(&n, &row_catalog) && !entry_names_other_kind(&n, self.surface) { candidates.insert(0, n); }
            }
            let candidates = crate::row_label::mark_animated(candidates, &row_catalog);
            let title = state.claim_title(&candidates);
            rows[row_idx].title = clamp_words(&title, 6);
        }

        rows
    }

    /// The name of the group (finest level first) that at least half of the titles belong to, if it has one.
    pub(super) fn majority_group_name(&self, items: &[&CatalogItem]) -> Option<String> {
        if items.is_empty() { return None; }
        for level in 0..crate::groups::N_LEVELS {
            let mut count: HashMap<i32, usize> = HashMap::new();
            for it in items { if let Some(g) = self.groups.get(&it.id) { let id = g.at(level); if id >= 0 { *count.entry(id).or_insert(0) += 1; } } }
            if let Some((id, n)) = count.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))) {
                if n * 2 >= items.len() {
                    if let Some(name) = self.group_names.get(&(level, id)) { return Some(name.clone()); }
                }
            }
        }
        None
    }

    /// Rows on a surface that shows both types (home) should mix movies and series when the theme has both.
    /// Clusters tend to be one type, so a row that is mostly one type gets about 40 percent of its nearest
    /// neighbours of the other type, picked by relevance to the row centroid (with a floor relative to the
    /// row's own items) and diversified with MMR. A row with fewer than MIN_BLEND on theme items of the other
    /// type stays as it is: not every row has to mix.
    pub(super) fn blend_other_type(&self, row: &mut Vec<RowItem>, centroid: &[f32], other: &str, cands: Vec<(u32, Vec<f32>)>) {
        const TARGET_SHARE: f32 = 0.4;
        const MIN_BLEND: usize = 3;
        const FLOOR_DROP: f32 = 0.12;
        if row.is_empty() || cands.is_empty() {
            return;
        }
        let have_other = row.iter().filter(|r| r.media_type == other).count();
        let target = ((ROW_SIZE as f32) * TARGET_SHARE).round() as usize;
        if have_other >= target {
            return;
        }
        // relevance of the items the row already has of its own type
        let mut own: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .filter(|(_, r)| r.media_type != other)
            .filter_map(|(i, r)| {
                let idx = self.vectors.index_of(r.id)?;
                Some((i, cosine_sim(centroid, &self.vectors.get_f32(idx))))
            })
            .collect();
        if own.is_empty() {
            return;
        }
        let mut vals: Vec<f32> = own.iter().map(|x| x.1).collect();
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let floor = vals[vals.len() / 2] - FLOOR_DROP;
        let ok: Vec<(u32, Vec<f32>)> = cands.into_iter().filter(|(_, v)| cosine_sim(centroid, v) >= floor).collect();
        if ok.is_empty() {
            return;
        }
        let picks = greedy_mmr(centroid, &ok, target - have_other, MMR_LAMBDA);
        if have_other + picks.len() < MIN_BLEND {
            return;
        }

        // compose: own items by relevance, then the other type, spread evenly through the row
        own.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let other_existing: Vec<RowItem> = row.iter().filter(|r| r.media_type == other).cloned().collect();
        let other_new: Vec<RowItem> = picks.iter().filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog)).collect();
        let n_other = other_existing.len() + other_new.len();
        let new_len = ROW_SIZE.min(row.len() + other_new.len());
        let keep_own = new_len.saturating_sub(n_other).min(own.len());
        let own_items: Vec<RowItem> = own.iter().take(keep_own).map(|&(i, _)| row[i].clone()).collect();
        let other_items: Vec<RowItem> = other_existing.into_iter().chain(other_new).collect();
        let mut keyed: Vec<(f32, RowItem)> = Vec::with_capacity(own_items.len() + other_items.len());
        for (rank, it) in own_items.iter().enumerate() {
            keyed.push(((rank as f32 + 0.5) / own_items.len() as f32, it.clone()));
        }
        for (rank, it) in other_items.iter().enumerate() {
            keyed.push(((rank as f32 + 0.25) / other_items.len() as f32, it.clone()));
        }
        keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        *row = keyed.into_iter().map(|(_, it)| it).collect();
    }

    /// Other type neighbours of a centroid from the HNSW index, for the seed path.
    pub(super) async fn other_type_neighbours(&self, centroid: &[f32], other: &str, exclude: &HashSet<u32>) -> Vec<(u32, Vec<f32>)> {
        let found = self.retrieve_candidates(centroid, ANN_ENTRY_K).await;
        found
            .into_iter()
            .filter_map(|(id, _)| {
                let item = self.catalog.get(&id)?;
                if item.media_type != other || exclude.contains(&id) || !self.matches_filters(item) || !self.member_ok(item) || !self.reliable(item) {
                    return None;
                }
                let idx = self.vectors.index_of(id)?;
                Some((id, self.vectors.get_f32(idx)))
            })
            .take(120)
            .collect()
    }
}
