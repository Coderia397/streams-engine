use super::*;

impl<'a> PageContext<'a> {
    /// Puts the Top 10 rows among the rows of the page by value, the way rows are chosen for a page: every row has a value,
    /// the page is filled greedily from the highest, and a row loses value for being like a row already placed (two charts
    /// are alike, so the second waits). Nothing says "second" or "third": a profile with a clear taste values a chart less
    /// than rows made for it, so the chart sinks; a new profile values a chart more, so it rises. A session's noise moves a
    /// chart a little so pages differ between sessions.
    pub(super) fn place_charts(&self, rows: Vec<Row>) -> Vec<Row> {
        /// Value of a chart for a profile without taste, and how much taste takes away.
        const CHART_BASE: f32 = 0.80;
        const CHART_TASTE_PENALTY: f32 = 0.30;
        const CHART_NOISE: f32 = 0.35;
        /// How much resemblance to a row already placed costs.
        const SIMILAR_COST: f32 = 0.35;

        let mut charts = self.build_chart_rows();
        let (old_charts, mut body): (Vec<Row>, Vec<Row>) = rows.into_iter().partition(|r| r.archetype == "trending");
        if charts.is_empty() {
            charts = old_charts;
        }
        if charts.is_empty() {
            return body;
        }
        // a title that is in a Top 10 is not repeated in the rows around it (the Top 10 must keep its ten)
        let chart_ids: HashSet<u32> = charts.iter().flat_map(|c| c.items.iter().map(|i| i.id)).collect();
        for row in body.iter_mut() {
            if row.items.iter().any(|i| chart_ids.contains(&i.id)) {
                row.items.retain(|i| !chart_ids.contains(&i.id));
            }
        }
        body.retain(|r| r.items.len() >= COHERENCE_MIN_ITEMS);

        // Fixed slots: the rows of the catalog layer (Top 10, New Releases, collections) and of the personal layer sit at the same positions in every session and on every profile
        // that has them; the topic rows keep their order around them. Positions are zero based and apply once the rows exist.
        let mut slotted: Vec<(usize, Row)> = Vec::new();
        let mut plain: Vec<Row> = Vec::new();
        let (mut byw_n, mut coll_n) = (0usize, 0usize);
        for r in body {
            match r.archetype.as_str() {
                "continue" => slotted.push((SLOT_CONTINUE, r)),
                "top_picks" => slotted.push((SLOT_TOP_PICKS, r)),
                "list" => slotted.push((SLOT_LIST, r)),
                "new" => slotted.push((SLOT_NEW, r)),
                "next_watch" => slotted.push((SLOT_NEXT, r)),
                "gems" => slotted.push((SLOT_GEMS, r)),
                "love_these" => slotted.push((SLOT_LOVE, r)),
                "favourites" => slotted.push((SLOT_FAVOURITES, r)),
                "byw" | "more_like" => { slotted.push((SLOT_ABOUT[byw_n.min(SLOT_ABOUT.len() - 1)], r)); byw_n += 1; }
                "collection" => { slotted.push((SLOT_COLLECTION[coll_n.min(SLOT_COLLECTION.len() - 1)], r)); coll_n += 1; }
                _ => plain.push(r),
            }
        }
        // topic rows of one genre (three horror rows in the first six) do not follow each other: rows are reordered so a dominant genre does not repeat within three rows
        let mut plain = if self.genre.is_none() { spread_by_genre(plain) } else { plain };
        for (k, r) in charts.into_iter().enumerate() {
            slotted.push((SLOT_CHART[k.min(SLOT_CHART.len() - 1)], r));
        }
        // The slots are places to aim at, not places to be at: a session moves each kind of row a little (later by 0 to 3 places, the same for the whole session), so the page is
        // not the same list in the same order for everyone; the first rows (continue watching, your list) stay where they are.
        let jitter = |slot: usize, archetype: &str| -> usize {
            if matches!(archetype, "continue" | "list" | "top_picks") { 0 } else { ((entry_noise(self.session, 7_000 + slot * 13) * 4.0) as usize).min(3) }
        };
        let slotted: Vec<(usize, Row)> = slotted.into_iter().map(|(slot, r)| (slot + jitter(slot, &r.archetype), r)).collect();
        let mut slotted = slotted;
        // equal slots (more rows than slots of a kind) go one after the other
        slotted.sort_by_key(|(slot, _)| *slot);
        let mut next_free = 0usize;
        for (slot, row) in slotted {
            let at = slot.max(next_free).min(plain.len());
            plain.insert(at, row);
            next_free = at + 1;
        }
        plain
    }

    pub(super) fn build_trending_row(&self) -> Option<Row> {
        self.build_chart_row(None)
    }

    /// The Top 10 rows of this page: Movies and Series separately on the home page, one row on the other pages.
    pub(super) fn build_chart_rows(&self) -> Vec<Row> {
        if self.genre.is_none() && self.surface == "both" && safety_nets() {
            ["movie", "tv"].into_iter().filter_map(|t| self.build_chart_row(Some(t))).collect()
        } else {
            self.build_chart_row(None).into_iter().collect()
        }
    }

    pub(super) fn build_chart_row(&self, only: Option<&str>) -> Option<Row> {
        let nets = safety_nets();
        let size = if nets { TOP_N } else { ROW_SIZE };
        let kids_chart = self.genre.is_none() && self.language.is_none() && self.maturity.is_some_and(|m| m.is_kids()) && self.profiles.has_profiles();
        let keep = |item: &CatalogItem| {
            only.is_none_or(|t| item.media_type == t)
                && self.matches_filters(item)
                // a children's Top 10 holds titles made for children, as the rest of their home does
                && (!kids_chart || kid_interest(self.profiles, item))
                // a title that has not come out yet is not in a Top 10 of what people watch
                && !self.fresh.is_upcoming(&item.media_type, item.tmdb_id, &self.today_ymd)
        };
        let by_pop_desc = |a: &&CatalogItem, b: &&CatalogItem| {
            b.popularity.partial_cmp(&a.popularity).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id))
        };
        let mut items: Vec<&CatalogItem> = if nets {
            // the catalog keeps the titles that can be on a chart, most popular first: the first `size` that pass the page's filters are the chart
            self.catalog.chart_keys.iter().filter_map(|k| self.catalog.get(k)).filter(|i| keep(i)).take(size).collect()
        } else {
            let mut all: Vec<&CatalogItem> = self.catalog.values().filter(|i| keep(i)).collect();
            if all.len() > size {
                all.select_nth_unstable_by(size - 1, by_pop_desc);
                all.truncate(size);
            }
            all
        };
        items.sort_unstable_by(by_pop_desc);
        let row_items: Vec<RowItem> = items
            .into_iter()
            .map(RowItem::from_catalog)
            .collect();

        if row_items.is_empty() {
            return None;
        }

        let title = match (only, self.surface, nets, kids_chart) {
            (Some("tv"), _, _, true) | (None, "tv", _, true) => "Top 10 Kids' Series Today",
            (Some("movie"), _, _, true) | (None, "movie", true, true) => "Top 10 Kids' Films Today",
            (_, _, _, true) => "Top 10 for Kids Today",
            (Some("tv"), _, _, _) | (None, "tv", _, _) => "Top 10 Series Today",
            (Some("movie"), _, _, _) | (None, "movie", true, _) => "Top 10 Movies Today",
            _ => "Top 10 Today",
        };

        Some(Row {
            title:     title.to_string(),
            archetype: "trending".to_string(),
            kind:      String::new(),
            items:     row_items, meta: None,
        })
    }

    /// Build a "Because You Watched X" row using HNSW retrieval.
    /// Anchor must pass quality gate AND be thematically distinct from all placed rows.
    pub(super) async fn build_byw_row(&self, state: &PageState) -> Option<Row> {
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
            items:     row_items, meta: None,
        })
    }
}
