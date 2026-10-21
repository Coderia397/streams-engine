//! Collections: an event or a mood shown as a banner with a few lead titles, and the events that are in season.
use super::*;

impl<'a> PageContext<'a> {
    /// The events whose calendar window is open today and that have title groups to show (Halloween in October, Pride in June).
    pub(crate) fn active_events(&self) -> Vec<&str> {
        events_in_season(self.taxonomy, self.vibe, self.today)
    }

    /// Chips for the top of a page: one for each event in season, with the page it opens (the Halloween chip of the Netflix app).
    pub fn season_chips(&self) -> Vec<serde_json::Value> {
        self.active_events().into_iter().map(|ev| json!({ "id": ev, "label_key": format!("event.{ev}"), "label": prettify(ev), "target": event_target(ev) })).collect()
    }

    /// The collection rows of the page: the themes the engine picked for this profile in this part of the day. A theme is an event in season (its sub groups of
    /// titles) or a mood ("For Every Feeling"); the pick rotates every six hours and differs between profiles, events in season are favoured, and a page shows one
    /// or two of them. Each row carries what a banner needs: label, description, accent hue, target page and the lead titles' pictures.
    pub(crate) fn build_collection_rows(&self, taken: &HashSet<u32>, want: usize) -> Vec<Row> {
        let slot = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600)) as usize;
        let seed = format!("{}|{}", self.session, slot);
        // two banners, a third now and then and a fourth rarely; fewer where the page asks for fewer
        let count = (2 + (entry_noise(&seed, 99_991) > 0.45) as usize + (entry_noise(&seed, 99_993) > 0.78) as usize).min(want);
        enum Theme<'a> { Event(&'a str), Mood(&'a crate::vibe::Mood), Topic(&'a str, &'a crate::topics::Topic) }
        let mut themes: Vec<(f32, Theme)> = Vec::new();
        for ev in self.active_events() { themes.push((0.35 + entry_noise(&seed, ev.len() * 31 + ev.bytes().map(|b| b as usize).sum::<usize>()), Theme::Event(ev))); }
        for m in &self.vibe.moods { themes.push((entry_noise(&seed, 7_000 + m.id.len() * 13 + m.id.bytes().map(|b| b as usize).sum::<usize>()) + if self.late_night && m.late { 0.4 } else { 0.0 }, Theme::Mood(m))); }
        // the judged topics are the discoveries: one or two a slot, chosen per session, each with a banner when it holds enough titles
        for (slug, topic) in self.topics.topics.iter() {
            if self.catalog.topic_keys.get(slug).is_some_and(|k| k.len() >= 40) {
                themes.push((entry_noise(&seed, 12_000 + slug.len() * 17 + slug.bytes().map(|b| b as usize).sum::<usize>()) * 0.9, Theme::Topic(slug.as_str(), topic)));
            }
        }
        themes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rows: Vec<Row> = Vec::new();
        let mut used = taken.clone();
        for (_, theme) in themes {
            if rows.len() >= count { break; }
            match theme {
                Theme::Event(ev) => {
                    let mut lead: Vec<&CatalogItem> = Vec::new();
                    let mut names: Vec<&str> = Vec::new();
                    for sub in &self.vibe.seasons[ev] {
                        if sub.kw_share < 0.4 || names.len() >= 4 { continue; }
                        let got = self.season_fit(ev, self.eligible(&sub.items, &used, true));
                        if got.len() < 8 { continue; }
                        names.push(sub.name.as_str());
                        lead.extend(got.into_iter().take(5));
                    }
                    let label = prettify(ev);
                    if lead.len() < 8 { continue; }
                    if let Some(mut r) = self.row_of_any(&format!("{label} Collection"), "collection", &lead) {
                        r.archetype = "collection".to_string();
                        r.meta = Some(collection_meta(ev, &format!("event.{ev}"), &label, &names.join(", "), event_target(ev), &r));
                        r.items.iter().for_each(|i| { used.insert(i.id); });
                        rows.push(r);
                    }
                }
                Theme::Topic(slug, topic) => {
                    let Some(keys) = self.catalog.topic_keys.get(slug) else { continue };
                    let items = self.eligible(keys, &used, true);
                    if items.len() < 12 { continue; }
                    if let Some(mut r) = self.row_of(&topic.name, "collection", &items) {
                        r.archetype = "collection".to_string();
                        r.meta = Some(collection_meta(&format!("topic_{slug}"), "collection.topic", "Streams Collection", &topic.definition, json!({ "collection": format!("topic_{slug}") }), &r));
                        r.items.iter().for_each(|i| { used.insert(i.id); });
                        rows.push(r);
                    }
                }
                Theme::Mood(m) => {
                    if let Some(mut r) = self.mood_row(m, &used) {
                        let names: Vec<&str> = m.groups.iter().take(3).filter_map(|(g, _)| self.group_names.get(&(5, *g)).map(|s| s.as_str())).collect();
                        r.archetype = "collection".to_string();
                        r.kind = "collection".to_string();
                        r.meta = Some(collection_meta(&m.id, "collection.for_every_feeling", "For Every Feeling", &names.join(", "), json!({ "collection": m.id }), &r));
                        r.items.iter().for_each(|i| { used.insert(i.id); });
                        rows.push(r);
                    }
                }
            }
        }
        rows
    }
}

/// Films and series alternate in a list that comes from popularity (series have larger popularity numbers and would fill a row), each kept in its own order.
pub(crate) fn blend_types(items: Vec<&CatalogItem>) -> Vec<&CatalogItem> {
    let (movies, series): (Vec<&CatalogItem>, Vec<&CatalogItem>) = items.into_iter().partition(|i| i.media_type == "movie");
    let (mut m, mut t) = (movies.into_iter(), series.into_iter());
    let mut out = Vec::new();
    loop {
        let (a, b) = (m.next(), t.next());
        if a.is_none() && b.is_none() { break; }
        out.extend(a);
        out.extend(b);
    }
    out
}

pub fn events_in_season<'a>(taxonomy: &'a crate::taxonomy::TaxonomyStore, vibe: &crate::vibe::VibeIndex, today: (u8, u8)) -> Vec<&'a str> {
    let mut events: Vec<&str> = Vec::new();
    for e in &taxonomy.entries {
        let all_year = e.season.is_none_or(|s| s.start == (1, 1) && s.end == (12, 31));
        if e.kind == "calendar" && !all_year && e.season.is_some_and(|s| s.active_on(today)) && vibe.seasons.contains_key(&e.event) && !events.contains(&e.event.as_str()) {
            events.push(e.event.as_str());
        }
    }
    events
}

pub fn prettify(id: &str) -> String {
    id.split('_').map(|w| { let mut c = w.chars(); c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default() }).collect::<Vec<_>>().join(" ")
}

/// The page that shows the whole collection of an event: the picker pages for the events that have one, the event id otherwise.
pub(crate) fn event_target(ev: &str) -> serde_json::Value {
    match ev {
        "halloween" => json!({ "genre": "Halloween" }),
        "pride" => json!({ "genre": "Pride" }),
        _ => json!({ "genre": ev }),
    }
}

/// The header of the page of an event (Halloween): what the banner of Home shows, for the top of the page that Explore all opens. None without the event's groups.
pub fn event_header(vibe: &crate::vibe::VibeIndex, catalog: &crate::catalog::CatalogMap, ev: &str) -> Option<serde_json::Value> {
    let subs = vibe.seasons.get(ev)?;
    let names: Vec<&str> = subs.iter().filter(|s| s.kw_share >= 0.4).take(4).map(|s| s.name.as_str()).collect();
    let images: Vec<&str> = subs.iter().flat_map(|s| s.items.iter()).filter_map(|k| catalog.get(k)).filter_map(|i| i.backdrop_path.as_deref()).filter(|p| !p.is_empty()).take(8).collect();
    if images.is_empty() { return None; }
    let hue = ev.bytes().fold(17u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) % 360;
    Some(json!({ "id": ev, "label_key": format!("event.{ev}"), "label": prettify(ev), "description": names.join(", "), "images": images, "hue": hue }))
}

pub(crate) fn collection_meta(id: &str, label_key: &str, label: &str, description: &str, target: serde_json::Value, row: &Row) -> serde_json::Value {
    let images: Vec<&str> = row.items.iter().filter_map(|i| i.backdrop_path.as_deref().or(i.poster_path.as_deref())).take(8).collect();
    // a colour of its own for each collection, the same on every visit
    let hue = id.bytes().fold(17u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) % 360;
    json!({
        "title_key": "collection.title",
        "title_args": { "label": label },
        "collection": { "id": id, "label_key": label_key, "label": label, "description": description, "target": target, "images": images, "hue": hue },
    })
}
