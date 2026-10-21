//! Rows made from requests (see `crate::requests`): which requests a page takes, how a request becomes a row, and the check every row passes before it is shown.
use super::*;
use crate::requests::Request;

/// A row from a request needs this many titles.
const REQUEST_MIN_ITEMS: usize = 12;
/// Two rows of one page whose request centres are closer than this are too alike (centred cosine).
const REQUEST_MAX_ANCHOR_SIM: f32 = 0.80;
/// At most this share of a page's rows comes from one kind of request (theme, era, origin, mood, calendar).
const REQUEST_KIND_SHARE: f32 = 0.45;
/// Titles ranked by closeness to the request's centre before diversity is applied.
const REQUEST_RANKED: usize = 60;
/// Eligible titles looked at per request (members are sorted best quality first).
const REQUEST_LOOK: usize = 400;

/// A stable number for a request id, to seed the session's rotation.
fn id_number(id: &str) -> usize {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    id.hash(&mut h);
    h.finish() as usize
}

/// Reorders rows so a row's kind differs from the row before it whenever one of the next rows allows (the first row stays first).
fn spread_by_kind(rows: Vec<Row>) -> Vec<Row> {
    let mut left: Vec<Option<Row>> = rows.into_iter().map(Some).collect();
    let mut out: Vec<Row> = Vec::with_capacity(left.len());
    let mut order: Vec<usize> = (0..left.len()).collect();
    while !order.is_empty() {
        let prev = out.last().map(|r| r.kind.clone());
        let pos = order
            .iter()
            .take(6)
            .position(|&i| prev.as_deref().is_none_or(|p| left[i].as_ref().is_some_and(|r| r.kind != p)))
            .unwrap_or(0);
        let i = order.remove(pos);
        out.extend(left[i].take());
    }
    out
}

impl<'a> PageContext<'a> {
    /// Whether this page is built from requests: the switch is on, requests are loaded, and the page is Home, Films, Series or a genre page of a profile that is not a child's.
    pub(crate) fn request_page(&self) -> bool {
        crate::requests::enabled() && !self.requests.is_empty() && !self.is_kids() && self.language.is_none() && self.animation.is_none()
    }

    /// Does this request belong on this page? In season, a centre to rank by, owned by this page (each request belongs to one page, see requests/store.rs), and a
    /// name that does not name the other kind of title.
    fn request_applies(&self, r: &Request) -> bool {
        !r.anchor.is_empty()
            && r.season.is_none_or(|s| s.active_on(self.today))
            && r.owner_surface == self.surface
            && r.owner_genre.as_ref() == self.genre_spec
            && !entry_names_other_kind(&r.def.name, self.surface)
    }

    /// How much this page wants a request: its kind, the profile's taste when there is one, its size, and the session's rotation (stable inside a session and a slot).
    fn request_score(&self, r: &Request, taste: Option<&[f32]>) -> f32 {
        let prior = match r.def.kind.as_str() {
            "calendar" => 1.0,
            "theme" => 0.8,
            "origin" | "mood" => 0.7,
            _ => 0.6,
        };
        let taste_fit = taste.map_or(0.0, |t| 0.6 * cosine_sim(t, &r.anchor));
        // broadly known requests lead; a request made mostly of children's titles sinks on a page that is not a child's
        let appeal = 0.6 * r.appeal;
        let children = if self.is_kids() { 0.0 } else { 0.6 * r.kid_share };
        let noise = if self.session.is_empty() { 0.0 } else { 0.35 * entry_noise(self.session, id_number(&r.def.id)) };
        prior + taste_fit + appeal - children + noise
    }

    /// The vector of a title centred on the corpus mean, unit length.
    fn centred_vector(&self, ix: usize) -> Vec<f32> {
        let mut c: Vec<f32> = self.vectors.get_f32(ix).iter().zip(self.mean).map(|(x, m)| x - m).collect();
        l2_normalize(&mut c);
        c
    }

    /// The rows of a page made from requests: requests by score, each turned into a row while it is unlike the rows placed (centre and kind share) and its titles are not
    /// on the page yet. The Top 10 rows are added later by `place_charts`, which also takes their titles out of these rows. Each row records the order it was built in
    /// (`request_order`), so a check can rebuild it with exactly the titles that were taken before it (ai-engine/scripts/verify_rows.py).
    pub(crate) fn build_request_page(&self) -> Vec<Row> {
        let mut rows: Vec<Row> = Vec::new();
        let taste: Option<Vec<f32>> = self.taste.iter().any(|x| x.abs() > 1e-6).then(|| {
            let mut t: Vec<f32> = self.taste.iter().zip(self.mean).map(|(x, m)| x - m).collect();
            l2_normalize(&mut t);
            t
        });
        let mut scored: Vec<(f32, &Request)> = self.requests.requests.iter().filter(|r| self.request_applies(r)).map(|r| (self.request_score(r, taste.as_deref()), r)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.def.id.cmp(&b.1.def.id)));

        let max_per_kind = ((self.target_rows as f32 * REQUEST_KIND_SHARE).ceil() as usize).max(2);
        let mut taken: HashSet<u32> = rows.iter().flat_map(|r| r.items.iter().map(|i| i.id)).collect();
        let mut placed: Vec<&[f32]> = Vec::new();
        let mut per_kind: HashMap<&str, usize> = HashMap::default();
        let mut made: Vec<Row> = Vec::new();
        for (_, req) in scored {
            if made.len() >= self.target_rows {
                break;
            }
            if per_kind.get(req.def.kind.as_str()).copied().unwrap_or(0) >= max_per_kind {
                continue;
            }
            if placed.iter().any(|a| cosine_sim(a, &req.anchor) > REQUEST_MAX_ANCHOR_SIM) {
                continue;
            }
            let Some(mut row) = self.request_row(req, &taken) else { continue };
            row.meta = Some(serde_json::json!({ "request": req.def.id, "request_order": made.len() }));
            taken.extend(row.items.iter().map(|i| i.id));
            placed.push(&req.anchor);
            *per_kind.entry(req.def.kind.as_str()).or_insert(0) += 1;
            made.push(row);
        }
        rows.extend(spread_by_kind(made));
        rows
    }

    /// One request as a row: its members this page may show and that are not on the page yet, ranked by closeness to the request's centre (and a little quality), made
    /// diverse, then checked: the row needs REQUEST_MIN_ITEMS titles and must keep its name's promise (genre, decade and nationality words, see row_label::name_fits).
    pub(crate) fn request_row(&self, req: &Request, taken: &HashSet<u32>) -> Option<Row> {
        let eligible: Vec<&CatalogItem> = req
            .members
            .iter()
            .filter_map(|k| self.catalog.get(k))
            .filter(|i| !taken.contains(&i.id) && self.page_pool_member(i))
            .take(REQUEST_LOOK)
            .collect();
        if eligible.len() < REQUEST_MIN_ITEMS {
            return None;
        }
        let mut ranked: Vec<(f32, u32, Vec<f32>)> = eligible
            .iter()
            .filter_map(|it| {
                let ix = self.vectors.index_of(it.id)?;
                if self.vectors.is_zero_row(ix) {
                    return None;
                }
                let c = self.centred_vector(ix);
                let score = cosine_sim(&c, &req.anchor) + 0.1 * (crate::quality::item_quality(it) / 35.0).min(1.5);
                Some((score, it.id, c))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        ranked.truncate(REQUEST_RANKED);
        let candidates: Vec<(u32, Vec<f32>)> = ranked.into_iter().map(|(_, id, v)| (id, v)).collect();
        let picked = greedy_mmr(&req.anchor, &candidates, ROW_SIZE * 2, MMR_LAMBDA);
        let items = self.diverse_titles(&picked, ROW_SIZE);
        if items.len() < REQUEST_MIN_ITEMS {
            return None;
        }
        let row_catalog: Vec<&CatalogItem> = items.iter().filter_map(|i| self.catalog.get(&i.id)).collect();
        if !crate::row_label::name_fits(&req.def.name, &row_catalog) {
            return None;
        }
        Some(Row {
            title:     req.def.name.clone(),
            archetype: "thematic".to_string(),
            kind:      req.def.kind.clone(),
            items,
            meta:      Some(serde_json::json!({ "request": req.def.id })),
        })
    }
}
