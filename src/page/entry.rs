use super::*;

impl<'a> PageContext<'a> {
    /// Netflix style page: rank row definitions (taxonomy entries) for this
    /// session first, then fill each chosen row with titles retrieved around the
    /// entry's own vector. Each entry is a row query, so the vocabulary can be
    /// far larger than the number of clusters and every row title is a real
    /// promise about what is inside.
    pub(super) async fn build_entry_page(&self) -> Vec<Row> {
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
        let mut used: HashSet<u32> = HashSet::default();
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
                    // a row for "both" that names the other kind (a films row on the series page) stays off this page
                    && !entry_names_other_kind(&e.title, self.surface)
                    && e.season.is_none_or(|s| s.active_on(self.today))
                    && !e.centered.is_empty()
                    && self.taxonomy.support_for(*i, self.surface).is_none_or(|n| n >= MIN_SUPPORT)
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
        // a genre page keeps only the neighbours that are of the genre (and allowed), which are few among a row's nearest titles overall: every eligible row is looked at, and deeper
        let on_genre = self.genre.is_some();
        let per_list = if on_genre { n_eligible.clamp(60, 400) } else { (want * 4).clamp(60, 200) };
        let ann_k = if on_genre { ANN_GENRE_K } else { ANN_ENTRY_K };
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
        let cands: Vec<Cand> = pick_idx
            .par_iter()
            .filter_map(|&idx| {
                let e = &self.taxonomy.entries[idx];
                let hits = self.retrieve_sync(&guard, &e.embedding, ann_k);
                let good: Vec<(&CatalogItem, f32)> = hits
                    .iter()
                    .filter_map(|(id, sim)| self.catalog.get(id).map(|i| (i, *sim)))
                    .filter(|(i, _)| self.matches_filters(i) && self.member_ok(i) && self.reliable(i))
                    .collect();
                if good.len() < MIN_ROW {
                    return None;
                }
                let strength = 0.2
                    * good.iter().take(10)
                        .map(|(i, _)| crate::quality::item_quality(i) / 35.0)
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
        if std::env::var("STREAMS_DEBUG_ENTRIES").is_ok() {
            eprintln!("entries: eligible {n_eligible}, shortlisted {}, with enough titles {} (surface {})", pick_idx.len(), cands.len(), self.surface);
        }

        // 3. Selection: score based diversification (MMR over row definitions) under page rules
        let max_per_type = (((want as f32) * 0.16).ceil() as usize).max(2);
        let max_calendar = 2usize;
        let mut type_counts: HashMap<String, usize> = HashMap::default();
        let mut chosen: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut max_sim: Vec<f32> = vec![0.0; cands.len()];
        let mut alive: Vec<bool> = vec![true; cands.len()];
        let mut last_kind = String::new();
        // two rows with the same name apart from filler words ("Hong Kong Action Classics", "Hong Kong HK Action Classics") do not share a page
        let cand_tokens: Vec<HashSet<String>> = cands.iter().map(|c| title_tokens(&self.taxonomy.entries[c.idx].title)).collect();
        let mut chosen_tokens: Vec<HashSet<String>> = Vec::new();

        while chosen.len() < want {
            let exploiting = chosen.len() < exploit_rows;
            let mut best: Option<(usize, f32)> = None;
            let mut relaxed: Option<(usize, f32)> = None; // ignores the adjacency rule
            for (ci, c) in cands.iter().enumerate() {
                if !alive[ci] || max_sim[ci] >= ENTRY_HARD_SIM {
                    continue;
                }
                if chosen_tokens.iter().any(|t| titles_alike(t, &cand_tokens[ci])) {
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
                if relaxed.is_none_or(|(_, v)| value > v) {
                    relaxed = Some((ci, value));
                }
                if (e.kind != last_kind || e.kind.is_empty()) && best.is_none_or(|(_, v)| value > v) {
                    best = Some((ci, value));
                }
            }
            let Some((ci, _)) = best.or(relaxed) else { break };
            alive[ci] = false;
            let e = &self.taxonomy.entries[cands[ci].idx];
            *type_counts.entry(e.kind.clone()).or_insert(0) += 1;
            last_kind = e.kind.clone();
            chosen_tokens.push(cand_tokens[ci].clone());
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
            // a genre page has a small pool, so a row holds back only its first titles from the rows after it (Netflix repeats a title across rows); the other pages keep every title in one row
            let reserve = if on_genre { ROW_SIZE / 2 } else { ROW_SIZE * 2 };
            fresh.iter().take(reserve).for_each(|id| { used.insert(*id); });
            chosen.push((cands[ci].idx, fresh));
        }
        let t_select = t0.elapsed();
        if std::env::var("STREAMS_DEBUG_ENTRIES").is_ok() {
            eprintln!("entries: chosen {} of wanted {want}", chosen.len());
        }

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
                items, meta: None,
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
                .filter(|i| self.matches_filters(i) && self.member_ok(i) && self.reliable(i))
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
                    items, meta: None,
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

    /// Up to `want` rows defined by a catalog fact (runtime, format, source), chosen per session among the kinds that suit the page's surface; a title the page already shows is not used again.
    pub(super) fn build_attribute_rows(&self, taken: &HashSet<u32>, want: usize) -> Vec<Row> {
        use crate::attributes::SPECS;
        let mut order: Vec<(f32, &crate::attributes::AttrSpec)> = SPECS
            .iter()
            .filter(|s| s.media == "both" || self.surface == "both" || s.media == self.surface)
            .map(|s| (entry_noise(self.session, 7919 + s.id.bytes().map(|b| b as usize * 31).sum::<usize>()), s))
            .collect();
        order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut used: HashSet<u32> = taken.clone();
        let mut rows: Vec<Row> = Vec::new();
        for (_, spec) in order {
            if rows.len() >= want { break; }
            let Some(keys) = self.catalog.attr_keys.get(spec.id) else { continue };
            let items = self.eligible(keys, &used, false);
            if items.len() < 12 { continue; }
            // quality order, shaken a little per session so the row is not the same twelve titles every time
            let mut scored: Vec<(f32, &CatalogItem)> = items.into_iter().take(150).enumerate().map(|(rank, i)| (40.0 * entry_noise(self.session, i.id as usize) - rank as f32, i)).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.id.cmp(&b.1.id)));
            let list: Vec<&CatalogItem> = scored.into_iter().map(|s| s.1).collect();
            let title = match self.surface { "movie" => spec.title_movie, "tv" => spec.title_tv, _ => spec.title_both };
            if let Some(r) = self.row_of(title, "attribute", &list) {
                r.items.iter().for_each(|i| { used.insert(i.id); });
                rows.push(r);
            }
        }
        rows
    }

    /// Up to `want` rows of titles that reviewers describe alike (vibe groups), chosen per session, in the page's genre when it has one.
    pub(super) fn build_vibe_rows(&self, taken: &HashSet<u32>, want: usize) -> Vec<Row> {
        if want == 0 || self.vibe.members.is_empty() { return Vec::new(); }
        let has_taste = self.taste.iter().any(|&x| x.abs() > 1e-6);
        let mut groups: Vec<i32> = self.vibe.members.keys().copied().collect();
        groups.sort_unstable();
        // a session sees a stable choice: groups are ordered by a session dependent noise and the first ones that qualify are scored
        groups.sort_by(|a, b| entry_noise(self.session, *b as usize).partial_cmp(&entry_noise(self.session, *a as usize)).unwrap_or(std::cmp::Ordering::Equal));
        let mut scored: Vec<(f32, String, Vec<&CatalogItem>)> = Vec::new();
        for g in groups.into_iter().take(500) {
            let Some(name) = self.group_names.get(&(5, g)) else { continue };
            let items = self.eligible(&self.vibe.members[&g], taken, false);
            if items.len() < 10 || !self.vibe_name_ok(name, &items) { continue; }
            let mass: f32 = items.iter().take(15).map(|i| crate::quality::item_quality(i)).sum::<f32>() / 15.0;
            let score = if has_taste {
                let mut c = vec![0.0f32; DIMS];
                for it in items.iter().take(10) { if let Some(ix) = self.vectors.index_of(it.id) { for (a, x) in c.iter_mut().zip(self.vectors.get_f32(ix)) { *a += x; } } }
                l2_normalize(&mut c);
                cosine_sim(&self.taste, &c)
            } else { mass * (0.7 + 0.6 * entry_noise(self.session, g as usize + 7)) };
            scored.push((score, name.clone(), items));
            if scored.len() >= want * 6 { break; }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new(); let mut used: HashSet<u32> = taken.clone(); let mut names: HashSet<String> = HashSet::default();
        for (_, name, items) in scored {
            if rows.len() >= want { break; }
            if !names.insert(name.to_lowercase()) { continue; }
            let fresh: Vec<&CatalogItem> = items.into_iter().filter(|i| !used.contains(&i.id)).collect();
            if let Some(r) = self.row_of(&name, "vibe", &fresh) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        rows
    }
}

/// The words of a row name that carry its meaning (lower case, without filler words and kind words such as "films").
pub(super) fn title_tokens(title: &str) -> HashSet<String> {
    const FILLER: [&str; 12] = ["and", "the", "of", "for", "a", "in", "hk", "films", "film", "movies", "shows", "series"];
    title
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !FILLER.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// True when two row names say the same thing: the same words, or one is the other plus a single word (when it has two words or more), or they share three quarters of their words.
pub(super) fn titles_alike(a: &HashSet<String>, b: &HashSet<String>) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let common = a.intersection(b).count();
    let (small, large) = if a.len() <= b.len() { (a.len(), b.len()) } else { (b.len(), a.len()) };
    if common == small && (large == small || (small >= 2 && large - small <= 1)) {
        return true;
    }
    common as f32 / (a.len() + b.len() - common) as f32 >= 0.75
}

#[cfg(test)]
mod title_tests {
    use super::*;

    #[test]
    fn same_name_apart_from_filler_is_alike() {
        let t = title_tokens;
        assert!(titles_alike(&t("Hong Kong HK Action Classics"), &t("Hong Kong Action Classics")));
        assert!(titles_alike(&t("Action"), &t("Action Films")));
        assert!(!titles_alike(&t("Action & Comedy"), &t("Action")));
        assert!(!titles_alike(&t("Korean Crime Drama"), &t("Korean Fantasy Dramas")));
    }
}
