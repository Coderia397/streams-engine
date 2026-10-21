use super::*;

impl<'a> PageContext<'a> {
    /// The vector a seasonal row ranks a title by: its catalog vector.
    pub(super) fn season_vector(&self, id: u32) -> Option<Vec<f32>> {
        let idx = self.vectors.index_of(id)?;
        if self.vectors.is_zero_row(idx) { return None; }
        Some(self.vectors.get_f32(idx))
    }

    /// Up to two rows for what is in season (taxonomy entries of type calendar whose window is active today): the entry's description is matched against the title
    /// profiles, the titles are filtered like any page (age limit, quality) and a row needs at least MIN titles that no other row has.
    pub(super) async fn build_seasonal_rows(&self, used: &HashSet<u32>) -> Vec<Row> {
        const MIN: usize = 8;
        let mut cands: Vec<usize> = self
            .taxonomy
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.kind == "calendar"
                    && e.season.is_some_and(|s| s.active_on(self.today))
                    && (e.media_type == "both" || e.media_type == self.surface || self.surface == "both")
                    && !entry_names_other_kind(&e.title, self.surface)
                    && !e.centered.is_empty()
            })
            .map(|(i, _)| i)
            .collect();
        // a session sees a stable choice among the active entries
        cands.sort_by(|a, b| entry_noise(self.session, *b).partial_cmp(&entry_noise(self.session, *a)).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new();
        let mut taken: HashSet<u32> = used.clone();
        let mut chosen_centred: Vec<&Vec<f32>> = Vec::new();
        for idx in cands.into_iter().take(8) {
            if rows.len() >= 2 { break; }
            let e = &self.taxonomy.entries[idx];
            if chosen_centred.iter().any(|c| cosine_sim(c, &e.centered) > 0.85) { continue; }
            // the titles of the season: those that carry one of the event's keywords (or, for Pride, whose profile says LGBTQ+ is central), ranked by how close
            // their profile is to the row's description and by quality; an event with too few such titles gets no row
            let mut pool: Vec<(f32, u32)> = Vec::new();
            for item in self.catalog.values() {
                let in_season = if e.lgbtq_central {
                    self.profiles.get(item.id).is_some_and(|p| p.lgbtq == crate::profiles::Lgbtq::Central)
                } else {
                    !e.keywords.is_empty() && item.keywords.iter().any(|k| e.keywords.iter().any(|w| w == k))
                };
                if !in_season || taken.contains(&item.id) || !self.member_ok(item) || !self.reliable(item) || !self.matches_filters(item) {
                    continue;
                }
                let sim = self.season_vector(item.id).map(|v| cosine_sim(&v, &e.embedding)).unwrap_or(0.0);
                pool.push((sim + 0.15 * (crate::quality::item_quality(item) / 35.0).min(1.5), item.id));
            }
            pool.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            let ids: Vec<u32> = pool.into_iter().take(ROW_SIZE * 4).map(|(_, id)| id).collect();
            if ids.len() < MIN { continue; }
            let cand_vecs: Vec<(u32, Vec<f32>)> = ids.iter().filter_map(|id| self.season_vector(*id).map(|v| (*id, v))).collect();
            let picked = greedy_mmr(&e.embedding, &cand_vecs, ROW_SIZE, MMR_LAMBDA);
            let items: Vec<RowItem> = picked.iter().filter_map(|id| self.catalog.get(id).map(RowItem::from_catalog)).collect();
            if items.len() < MIN { continue; }
            items.iter().for_each(|i| { taken.insert(i.id); });
            chosen_centred.push(&e.centered);
            rows.push(Row { title: clamp_words(&e.title, 5), archetype: "thematic".to_string(), kind: e.kind.clone(), items, meta: None });
        }
        rows
    }

    /// The centred vector of an event: the mean of the centred vectors of its calendar entries (Halloween: Spooky Mystery Films, Gothic Thrillers ...); None when the vocabulary has none.
    pub(super) fn event_vector(&self, event: &str) -> Option<Vec<f32>> {
        let mut sum = vec![0.0f32; DIMS]; let mut n = 0;
        for e in self.taxonomy.entries.iter().filter(|e| e.kind == "calendar" && e.event == event && e.centered.len() == DIMS) {
            for (a, x) in sum.iter_mut().zip(&e.centered) { *a += x; }
            n += 1;
        }
        if n == 0 { return None; }
        l2_normalize(&mut sum);
        Some(sum)
    }

    /// Rows of the calendar events that are in season: each event's pool is split into named sub groups, the page shows two of the first event and one of the second.
    pub(super) fn build_seasonal_sub_rows(&self, taken: &HashSet<u32>) -> Vec<Row> {
        let mut events: Vec<(&str, f32)> = Vec::new();
        for e in &self.taxonomy.entries {
            let all_year = e.season.is_none_or(|s| s.start == (1, 1) && s.end == (12, 31));
            if e.kind == "calendar" && !all_year && e.season.is_some_and(|s| s.active_on(self.today)) && self.vibe.seasons.contains_key(&e.event) && !events.iter().any(|(x, _)| *x == e.event.as_str()) {
                events.push((e.event.as_str(), entry_noise(self.session, e.event.len() * 31 + e.event.bytes().map(|b| b as usize).sum::<usize>())));
            }
        }
        events.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new(); let mut used: HashSet<u32> = taken.clone();
        for (ei, (ev, _)) in events.into_iter().take(2).enumerate() {
            let want = if ei == 0 { 2 } else { 1 };
            let mut subs: Vec<&crate::vibe::SeasonSub> = self.vibe.seasons[ev].iter().collect();
            subs.sort_by(|a, b| entry_noise(self.session, b.name.len() + b.items.len()).partial_cmp(&entry_noise(self.session, a.name.len() + a.items.len())).unwrap_or(std::cmp::Ordering::Equal));
            let mut got = 0;
            for sub in subs {
                if got >= want { break; }
                if sub.kw_share < SEASON_MIN_KW_SHARE { continue; }
                let items = self.season_fit(ev, self.eligible(&sub.items, &used, true));
                if items.len() < 8 { continue; }
                let title = if self.vibe_name_ok(&sub.name, &items) { sub.name.clone() } else { continue };
                if let Some(r) = self.row_of(&title, "calendar", &items) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); got += 1; }
            }
        }
        rows
    }

    /// The page of a calendar event (Halloween, Pride): its sub groups as rows, in the order of their size.
    pub(super) fn build_event_page(&self, event: &str) -> Vec<Row> {
        let Some(subs) = self.vibe.seasons.get(event) else { return Vec::new() };
        let mut rows: Vec<Row> = Vec::new();
        if let Some(r) = self.build_trending_row() { rows.push(r); }
        let mut used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
        let mut titles: HashSet<String> = HashSet::default();
        for sub in subs {
            if rows.len() > self.target_rows { break; }
            if sub.kw_share < SEASON_MIN_KW_SHARE { continue; }
            let items = self.season_fit(event, self.eligible(&sub.items, &used, true));
            if items.len() < 8 || !self.vibe_name_ok(&sub.name, &items) || !titles.insert(sub.name.to_lowercase()) { continue; }
            if let Some(r) = self.row_of(&sub.name, "calendar", &items) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        rows
    }

    /// The Moods page: mood rows first (late night moods first at night), vibe rows after them.
    pub(super) fn build_moods_page(&self) -> Vec<Row> {
        let mut rows: Vec<Row> = Vec::new();
        if let Some(r) = self.build_trending_row() { rows.push(r); }
        let mut used: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
        let moods = self.build_mood_rows(&used, self.target_rows, false);
        moods.iter().for_each(|r| r.items.iter().for_each(|i| { used.insert(i.id); }));
        rows.extend(moods);
        if rows.len() < self.target_rows { rows.extend(self.build_vibe_rows(&used, self.target_rows - rows.len())); }
        rows
    }

    /// A children's home: half the rows from live action titles, half from animated ones, alternating, so the page is not all cartoons. Each half is
    /// a page of its own (clustered from its own titles); the charts come from the live action half, which is built over the whole pool.
    pub(super) fn build_kids_home(&self) -> Vec<Row> {
        let half = (self.target_rows / 2).max(4) + 1;
        let live = PageContext { animation: Some(false), target_rows: half, ..self.clone() }.build_genre_page();
        let anim = PageContext { animation: Some(true), target_rows: half, ..self.clone() }.build_genre_page();
        let (charts, live_rows): (Vec<Row>, Vec<Row>) = live.into_iter().partition(|r| r.archetype == "trending");
        let anim_rows: Vec<Row> = anim.into_iter().filter(|r| r.archetype != "trending").collect();
        let mut out: Vec<Row> = charts;
        let (mut a, mut l) = (anim_rows.into_iter(), live_rows.into_iter());
        loop {
            match (l.next(), a.next()) {
                (None, None) => break,
                (lr, ar) => {
                    out.extend(lr);
                    out.extend(ar);
                }
            }
        }
        // a row name used twice by the two halves keeps its first use
        let mut seen: HashSet<String> = HashSet::default();
        out.retain(|r| r.archetype == "trending" || seen.insert(r.title.clone()));
        out
    }

    /// The page of a mood collection (what Explore all opens): the mood's own row, then one row for each vibe group that fits it, then more mood rows. Returns the rows and the header
    /// (label, sentence, pictures, colour) of the page; None when there is no such mood.
    pub(crate) fn build_mood_collection(&self, id: &str) -> Option<(Vec<Row>, serde_json::Value)> {
        if let Some(slug) = id.strip_prefix("topic_") { return self.build_topic_collection(slug); }
        let m = self.vibe.moods.iter().find(|m| m.id == id)?;
        let mut rows: Vec<Row> = Vec::new();
        let mut used: HashSet<u32> = HashSet::default();
        if let Some(r) = self.mood_row(m, &used) {
            r.items.iter().for_each(|i| { used.insert(i.id); });
            rows.push(r);
        }
        let mut names: Vec<&str> = Vec::new();
        for (g, _) in m.groups.iter().take(16) {
            if rows.len() >= self.target_rows { break; }
            let (Some(keys), Some(name)) = (self.vibe.members.get(g), self.group_names.get(&(5, *g))) else { continue };
            let items = self.eligible(keys, &used, false);
            if items.len() < 8 || !self.vibe_name_ok(name, &items) { continue; }
            if let Some(r) = self.row_of(name, "vibe", &items) {
                r.items.iter().for_each(|i| { used.insert(i.id); });
                if names.len() < 3 { names.push(name.as_str()); }
                rows.push(r);
            }
        }
        if rows.len() < self.target_rows {
            rows.extend(self.build_mood_rows(&used, self.target_rows - rows.len(), false).into_iter().filter(|r| r.title != m.title));
        }
        let images: Vec<&str> = rows.iter().flat_map(|r| r.items.iter()).filter_map(|i| i.backdrop_path.as_deref()).filter(|p| !p.is_empty()).take(8).collect();
        if rows.is_empty() || images.is_empty() { return None; }
        let hue = m.id.bytes().fold(17u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) % 360;
        let header = serde_json::json!({ "id": m.id, "label_key": "collection.for_every_feeling", "label": "For Every Feeling", "title": m.title, "description": names.join(", "), "images": images, "hue": hue });
        Some((rows, header))
    }

    /// The page of a topic collection: the topic's best titles, then its titles by genre and by decade, then rows of other topics. Returns the rows and the header of the page.
    pub(crate) fn build_topic_collection(&self, slug: &str) -> Option<(Vec<Row>, serde_json::Value)> {
        let topic = self.topics.topics.get(slug)?;
        let keys = self.catalog.topic_keys.get(slug)?;
        let all = self.eligible(keys, &HashSet::default(), true);
        if all.len() < 12 { return None; }
        let mut rows: Vec<Row> = Vec::new();
        let mut used: HashSet<u32> = HashSet::default();
        if let Some(r) = self.row_of(&topic.name, "topic", &all) {
            r.items.iter().for_each(|i| { used.insert(i.id); });
            rows.push(r);
        }
        // by genre
        let mut by_genre: crate::det::HashMap<&str, Vec<&CatalogItem>> = crate::det::HashMap::default();
        for i in &all { if let Some(g) = i.genres.first() { by_genre.entry(g.as_str()).or_default().push(i); } }
        let mut genres: Vec<(&str, Vec<&CatalogItem>)> = by_genre.into_iter().filter(|(_, v)| v.len() >= 10).collect();
        genres.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
        for (g, v) in genres.iter().take(6) {
            let fresh: Vec<&CatalogItem> = v.iter().copied().filter(|i| !used.contains(&i.id)).collect();
            let noun = crate::row_label::genre_noun(g).unwrap_or(g);
            // a genre the topic's own name already says (Political Dramas, Dramas) makes no row of its own
            if topic.name.to_lowercase().contains(&noun.to_lowercase()) { continue; }
            let name = format!("{} · {}", topic.name, noun);
            if let Some(r) = self.row_of(&name, "topic", &fresh) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        // by decade
        let mut by_decade: std::collections::BTreeMap<i32, Vec<&CatalogItem>> = std::collections::BTreeMap::new();
        for i in &all { if let Some(y) = i.release_year { by_decade.entry(y / 10 * 10).or_default().push(i); } }
        for (d, v) in by_decade.iter().rev() {
            let fresh: Vec<&CatalogItem> = v.iter().copied().filter(|i| !used.contains(&i.id)).collect();
            if fresh.len() < 10 { continue; }
            if let Some(r) = self.row_of(&format!("{} of the {}s", topic.name, d), "topic", &fresh) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        if rows.len() < self.target_rows {
            let skip: HashSet<String> = rows.iter().map(|r| r.title.to_lowercase()).chain(std::iter::once(topic.name.to_lowercase())).collect();
            rows.extend(self.build_topic_rows(&used, self.target_rows - rows.len(), &skip));
        }
        let images: Vec<String> = rows.iter().flat_map(|r| r.items.iter()).filter_map(|i| i.backdrop_path.clone()).filter(|p| !p.is_empty()).take(8).collect();
        if images.is_empty() { return None; }
        let id = format!("topic_{slug}");
        let hue = id.bytes().fold(17u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) % 360;
        Some((rows, serde_json::json!({ "id": id, "label_key": "collection.topic", "label": "Streams Collection", "title": topic.name, "description": topic.definition, "images": images, "hue": hue })))
    }
}
