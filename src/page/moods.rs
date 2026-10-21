//! Mood rows: late night, cozy, mind bending and the rest, made from the vibe groups that fit a mood phrase.
use super::*;

impl<'a> PageContext<'a> {
    /// The row of one mood: titles of its vibe groups ranked by their fit to the mood phrase; None when fewer than 12 titles are free.
    pub(crate) fn mood_row(&self, m: &crate::vibe::Mood, used: &HashSet<u32>) -> Option<Row> {
        let mut cand: Vec<(f32, &CatalogItem)> = Vec::new(); let mut seen: HashSet<u32> = HashSet::default();
        // the phrase vector, centred like the title vectors; a title is ranked by its own fit to the phrase as well as by the review match of its vibe group
        let mv: Option<Vec<f32>> = (m.vector.len() == DIMS).then(|| { let mut c: Vec<f32> = m.vector.iter().zip(self.mean).map(|(x, mu)| x - mu).collect(); l2_normalize(&mut c); c });
        for (g, score) in m.groups.iter().take(6) {
            if let Some(keys) = self.vibe.members.get(g) {
                for it in self.eligible(keys, used, false).into_iter().take(60) {
                    if !seen.insert(it.id) { continue; }
                    let fit = match &mv {
                        Some(v) => {
                            let Some(ix) = self.vectors.index_of(it.id) else { continue };
                            if self.vectors.is_zero_row(ix) { continue; }
                            let mut c: Vec<f32> = self.vectors.get_f32(ix).iter().zip(self.mean).map(|(x, mu)| x - mu).collect();
                            l2_normalize(&mut c);
                            cosine_sim(&c, v)
                        }
                        None => 0.0,
                    };
                    cand.push((fit + 0.5 * *score + 0.1 * (crate::quality::item_quality(it) / 35.0).min(1.5), it));
                }
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.id.cmp(&b.1.id)));
        let items: Vec<&CatalogItem> = cand.into_iter().map(|c| c.1).collect();
        if items.len() < 12 { return None; }
        self.row_of(&m.title, "mood", &items)
    }

    /// Mood rows (late night, cozy, mind-bending ...): the titles of the vibe groups that fit the mood best. On a late night page the late night moods come first.
    pub(crate) fn build_mood_rows(&self, taken: &HashSet<u32>, want: usize, late_only: bool) -> Vec<Row> {
        if want == 0 || self.vibe.moods.is_empty() { return Vec::new(); }
        let mut moods: Vec<&crate::vibe::Mood> = self.vibe.moods.iter().filter(|m| !late_only || m.late).collect();
        moods.sort_by(|a, b| entry_noise(self.session, b.id.len() + b.title.len()).partial_cmp(&entry_noise(self.session, a.id.len() + a.title.len())).unwrap_or(std::cmp::Ordering::Equal));
        if self.late_night && !late_only { moods.sort_by_key(|m| !m.late); }
        let mut rows: Vec<Row> = Vec::new(); let mut used: HashSet<u32> = taken.clone();
        for m in moods {
            if rows.len() >= want { break; }
            if let Some(r) = self.mood_row(m, &used) { r.items.iter().for_each(|i| { used.insert(i.id); }); rows.push(r); }
        }
        rows
    }
}
