//! Rows made from requests (see `crate::requests`): which requests a page takes, how a request becomes a row, and the check every row passes before it is shown.
//!
//! Version 1 requests belong to one owner page and fill rows of ROW_SIZE titles unique across the page. Version 2 requests (composed rows, note 101) follow what
//! Netflix's pages do (note 97): a row runs on every page whose scope its titles fit, goes up to its depth (150 titles at most), and only the first screen of each
//! row (its first FIRST_SCREEN tiles) is kept free of titles already on the page; past the first screen rows may share titles.
use super::*;
use crate::requests::Request;

/// A row from a request needs this many titles.
const REQUEST_MIN_ITEMS: usize = 12;
/// Two rows of one page whose request centres are closer than this are too alike (centred cosine).
const REQUEST_MAX_ANCHOR_SIM: f32 = 0.80;
/// The same limit on a genre page, whose rows all share the genre's direction.
const REQUEST_MAX_ANCHOR_SIM_GENRE: f32 = 0.90;
/// At most this share of a page's rows comes from one kind of request (theme, era, origin, mood, calendar).
const REQUEST_KIND_SHARE: f32 = 0.45;
/// Titles ranked by closeness to the request's centre before diversity is applied.
const REQUEST_RANKED: usize = 60;
/// Eligible titles looked at per request (members are sorted best quality first).
const REQUEST_LOOK: usize = 400;
/// The tiles of a row a viewer sees before paging it: unique across the page (version 2).
pub const FIRST_SCREEN: usize = 6;
/// The deepest a composed row goes.
pub const MAX_ROW_DEPTH: usize = 150;
/// The first tiles of a composed row are made diverse (MMR); the rest follow the ranking.
const DIVERSE_HEAD: usize = 24;
/// Netflix's Home (note 97, 48 rows): 2 rows named for a place, none for an era, about 10 of mood or feel; composed pages keep place and era rows to this share each.
const PLACE_SHARE: f32 = 0.10;
const ERA_SHARE: f32 = 0.08;

/// Share of animated titles a row that is not about animation may hold on a page that is not a child's.
const ANIMATION_SHARE: f32 = 0.25;

/// Moods Netflix keeps below the fold.
const HARSH_MOODS: &[&str] = &["Violent", "Raunchy", "Gory", "Steamy", "Scandalous", "Scary", "Ominous", "Chilling"];

/// Subjects a row may be named for on a child's page (a token of a subject block, matched as a word part).
const KID_SUBJECTS: &[&str] = &["friend", "animal", "dinosaur", "space", "time travel", "superhero", "magic", "princess", "school", "sport", "music", "dance", "robot", "pirate",
    "dragon", "monster", "adventure", "family", "holiday", "christmas", "halloween", "witch", "alien", "road trip", "teen", "high school", "fairy", "toy", "pet", "dog", "cat",
    "horse", "ocean", "sea", "underwater", "jungle", "farm", "train", "car", "truck", "science", "nature", "wildlife", "circus", "camp", "summer", "winter", "snow", "ballet",
    "singing", "band", "unicorn", "mermaid", "knight", "ninja", "martial arts", "slapstick", "prank", "mystery", "detective", "treasure", "explorer", "new york", "london", "paris",
    "tokyo", "coming-of-age", "coming of age", "sibling", "grandparent", "baby", "little", "puppy", "kitten", "bunny", "bear", "penguin", "shark", "dolphin", "whale", "zoo",
    "video game", "cartoon", "comic", "lego", "doll", "superpower", "wizard", "giant", "monster truck", "spy", "secret agent", "talking animal", "girl power", "feel-good"];

/// Moods a row may be named for on a child's page.
const KID_MOODS: &[&str] = &["Feel-Good", "Heartfelt", "Goofy", "Quirky", "Imaginative", "Charming", "Exciting", "Inspiring", "Fun", "Cute", "Whimsical", "Magical", "Humorous", "Relaxing", "Witty", "Epic"];

/// Genres a row may be named for on a child's page.
const KID_GENRES: &[&str] = &["Animation", "Family", "Adventure", "Comedy", "Fantasy", "Kids", "Science Fiction", "Music", "Action", "Documentary", "Reality", "Mystery"];
/// On the page of an event, a row composed for that event scores this much more than a row the scope only cuts.
const EVENT_PAGE_OWN_ROW: f32 = 0.8;
/// On a genre page a composed row needs this share of its titles inside the page, unless its own rules name the page's genre.
const GENRE_PAGE_MIN_SHARE: f32 = 0.15;
/// Or this many of its first REQUEST_LOOK titles inside the page.
const GENRE_PAGE_MIN_FIT: usize = 25;
/// Rows a composed page holds at most (Netflix's Home, Series and Films: 47 to 48 rows).
const COMPOSED_ROWS_CAP: usize = 40;   // plus the personal and chart rows the page adds, about 48 in all

/// The family a composed row belongs to for the page's rhythm: place (an origin block), era (a decade block), or topic (everything else).
fn family_of(r: &Request) -> &'static str {
    match r.def.template.as_deref() {
        Some(t) if t.contains("{origin}") => "place",
        Some(t) if t.contains("{era}") => "era",
        _ => "topic",
    }
}

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
    /// Whether this page is built from requests: the switch is on and requests are loaded. Version 1 serves Home, Films, Series and the genre pages of a profile that is
    /// not a child's; version 2 serves a child's pages too (Netflix's Kids rows are among the composed rows).
    pub(crate) fn request_page(&self) -> bool {
        crate::requests::enabled()
            && !self.requests.is_empty()
            && self.language.is_none()
            && self.animation.is_none()
            && (self.requests.version >= 2 || !self.is_kids())
    }

    /// Composed rows: deep, free to run on any page in scope, first screen unique (see the module note).
    pub(crate) fn composed_rows(&self) -> bool {
        self.requests.version >= 2
    }

    /// Does this request belong on this page? In season, a centre to rank by, a name that does not name the other kind of title, and either owned by this page (version 1)
    /// or of a format and profile kind that fit the page (version 2; whether its titles fit the page's genre is decided when the row is built).
    fn request_applies(&self, r: &Request) -> bool {
        if r.anchor.is_empty() || !r.season.is_none_or(|s| s.active_on(self.today)) || entry_names_other_kind(&r.def.name, self.surface) {
            return false;
        }
        if !r.free {
            return r.owner_surface == self.surface && r.owner_genre.as_ref() == self.genre_spec;
        }
        let format_ok = r.owner_surface == "both" || self.surface == "both" || r.owner_surface == self.surface;
        let profile_ok = match (r.def.profile.as_deref().unwrap_or("any"), self.is_kids()) {
            ("kids", false) => false,
            ("adult", true) => false,
            // a child's page takes the rows Netflix shows to children, and otherwise only rows named for a genre or subject that suits children (no eras, origins,
            // awards or adult genres: a "2000s Romantic Films" row is not a children's row even when its titles are rated PG)
            ("any", true) => {
                let tpl = r.def.template.as_deref().unwrap_or("");
                let plain = matches!(tpl, "{genre} {format}" | "{genre}" | "{subject} {format}" | "{subject} {genre} {format}" | "{subject}" | "{event} {format}" | "{event} {genre} {format}" | "{event}");
                let mood = matches!(tpl, "{mood} {format}" | "{mood} {genre} {format}" | "{mood} {genre}") && KID_MOODS.iter().any(|m| r.def.name.starts_with(m));
                // a subject row reaches a child's page only for subjects children's rows are about ("Revenge Films" with Zootopia in it is not a children's row)
                let subject_ok = r.def.blocks.iter().filter(|b| b.bucket == "subject").all(|b| KID_SUBJECTS.iter().any(|k| b.token.contains(k)));
                (plain || mood) && subject_ok && r.def.rules.genre_specs.iter().all(|g| KID_GENRES.contains(&g.as_str()))
            }
            _ => true,
        };
        format_ok && profile_ok
    }

    /// The share of a request's titles this page may show (its first members, best quality first): a row belongs on a genre page when much of it fits the page,
    /// not when a dozen of its titles happen to pass the page's filter.
    fn page_fit(&self, r: &Request) -> (f32, usize) {
        let head = &r.members[..r.members.len().min(REQUEST_LOOK)];
        if head.is_empty() {
            return (0.0, 0);
        }
        let fit = head.iter().filter_map(|k| self.catalog.get(k)).filter(|i| self.page_pool_member(i)).count();
        (fit as f32 / head.len() as f32, fit)
    }

    /// How much this page wants a request: its kind, the profile's taste when there is one, its size, and the session's rotation (stable inside a session and a slot).
    fn request_score(&self, r: &Request, taste: Option<&[f32]>) -> f32 {
        let prior = match r.def.kind.as_str() {
            "calendar" => 1.0,
            // an animation tier row (note 103) is curated by audience and rhythm: it leads like a calendar row
            "tier" => 0.95,
            "theme" => 0.8,
            "origin" | "mood" => 0.7,
            _ => 0.6,
        };
        // a row Netflix itself shows, under its own name, leads; a row whose format matches the page exactly leads on Films and Series
        let known = if r.def.netflix_name.is_some() { 0.5 } else { 0.0 };
        let fits = if r.free && self.surface != "both" && r.owner_surface == self.surface { 0.15 } else { 0.0 };
        // a deep row (Netflix's median row holds 150 titles) leads a thin one: up to 0.3 for 150 or more members
        let depth = if r.free { 0.3 * ((r.members.len().max(1) as f32).ln() / (MAX_ROW_DEPTH as f32).ln()).min(1.0) } else { 0.0 };
        // Netflix places its harsher moods (Violent, Raunchy, Gory, Steamy, Scandalous, Scary) in the lower half of a page
        let harsh = if r.free && HARSH_MOODS.iter().any(|m| r.def.name.starts_with(m)) { 0.45 } else { 0.0 };
        let taste_fit = taste.map_or(0.0, |t| 0.6 * cosine_sim(t, &r.anchor));
        // broadly known requests lead; a request made mostly of children's titles sinks on a page that is not a child's
        let appeal = 0.6 * r.appeal;
        let children = if self.is_kids() { 0.0 } else { 0.6 * r.kid_share };
        let noise = if self.session.is_empty() { 0.0 } else { 0.35 * entry_noise(self.session, id_number(&r.def.id)) };
        prior + known + fits + depth + taste_fit + appeal - children - harsh + noise
    }

    /// The vector of a title centred on the corpus mean, unit length.
    fn centred_vector(&self, ix: usize) -> Vec<f32> {
        let mut c: Vec<f32> = self.vectors.get_f32(ix).iter().zip(self.mean).map(|(x, m)| x - m).collect();
        l2_normalize(&mut c);
        c
    }

    /// The rows of a page made from requests: requests by score, each turned into a row while it is unlike the rows placed (centre and kind share) and its titles are not
    /// on the page yet (version 2: not in the first screen of a row yet). The Top 10 rows are added later by `place_charts`, which also takes their titles out of these
    /// rows. Each row records the order it was built in (`request_order`), so a check can rebuild it with exactly the titles that were taken before it
    /// (ai-engine/scripts/verify_rows.py).
    pub(crate) fn build_request_page(&self) -> Vec<Row> {
        let composed = self.composed_rows();
        let taste: Option<Vec<f32>> = self.taste.iter().any(|x| x.abs() > 1e-6).then(|| {
            let mut t: Vec<f32> = self.taste.iter().zip(self.mean).map(|(x, m)| x - m).collect();
            l2_normalize(&mut t);
            t
        });
        // on a genre page a composed row competes by how much of it belongs to the page: rows that name the page's genre always may, the others need a share.
        // A page named for an origin, an award, a judged collection or a season (British, Emmys, Black Stories, Halloween) is a filter over every row instead: its
        // rows keep their names and the page's rule keeps the titles (as Netflix's Black Stories page carries Crime Drama Series and Comedies)
        let genre_like = |spec: &crate::genre_spec::GenreSpec| {
            use crate::genre_spec::GenreSpec::*;
            // a child's Reality page is a scope over the children's rows (Netflix fills it with Funny, Good Vibes Only, Watch with the Family cut to reality shows)
            matches!(spec, Tmdb(_) | Thriller | Horror | Romance | Anime | Musical | Teen | Name(_)) || (matches!(spec, Reality) && !self.is_kids())
        };
        let genre_page = composed && self.genre_spec.is_some_and(genre_like);
        let names_genre = |r: &Request| -> bool {
            self.genre_spec.is_some_and(|spec| r.def.rules.genre_specs.iter().any(|g| crate::genre_spec::resolve(g, self.surface) == *spec))
        };
        let mut scored: Vec<(f32, &Request)> = self
            .requests
            .requests
            .iter()
            .filter(|r| {
                let ok = self.request_applies(r);
                if r.def.kind == "tier" { tracing::debug!("tier row {} applies on {} kids={}: {} (anchor {}, owner {}, profile {:?})", r.def.name, self.surface, self.is_kids(), ok, !r.anchor.is_empty(), r.owner_surface, r.def.profile); }
                ok
            })
            .filter_map(|r| {
                let mut score = self.request_score(r, taste.as_deref());
                if genre_page {
                    // a row belongs on a genre page when a share of it fits the page, or enough of its titles do (a mood row on the Documentaries page: the genre is
                    // 6% of the catalog, so the share is small, but 25 gritty documentaries make a true "Gritty Films" row there), or its rules name the genre
                    let (share, fit) = self.page_fit(r);
                    if share < GENRE_PAGE_MIN_SHARE && fit < GENRE_PAGE_MIN_FIT && !names_genre(r) {
                        return None;
                    }
                    score += 0.8 * share + if names_genre(r) { 0.4 } else { 0.0 };
                }
                // on the page of an event (Christmas, Halloween) the rows composed for that event lead the rows the scope merely cuts
                if let Some(ev) = self.genre_spec.and_then(|s| s.event_id()) {
                    if r.def.blocks.iter().any(|b| b.bucket == "event" && b.token == ev) {
                        score += EVENT_PAGE_OWN_ROW;
                    }
                }
                Some((score, r))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.def.id.cmp(&b.1.def.id)));
        let scored_n = scored.len();

        let max_per_kind = if composed { usize::MAX } else { ((self.target_rows as f32 * REQUEST_KIND_SHARE).ceil() as usize).max(2) };
        // composed pages follow Netflix's rhythm: rows named for a place or an era are a small share of the page; the rest are topic rows
        let family_cap = |fam: &str| -> usize {
            match fam {
                "place" => ((self.target_rows as f32 * PLACE_SHARE).ceil() as usize).max(1),
                "era" => ((self.target_rows as f32 * ERA_SHARE).ceil() as usize).max(1),
                _ => usize::MAX,
            }
        };
        let mut taken: HashSet<u32> = HashSet::default();
        let mut placed: Vec<&[f32]> = Vec::new();
        let mut names: HashSet<String> = HashSet::default();
        let mut per_kind: HashMap<&str, usize> = HashMap::default();
        let mut per_family: HashMap<&str, usize> = HashMap::default();
        let mut made: Vec<Row> = Vec::new();
        // Netflix's plain pages hold about 48 rows (note 97); a composed page stops there, a thin page stops when its rows run out
        let target = if composed { self.target_rows.min(COMPOSED_ROWS_CAP) } else { self.target_rows };
        for (_, req) in scored {
            if made.len() >= target {
                break;
            }
            if per_kind.get(req.def.kind.as_str()).copied().unwrap_or(0) >= max_per_kind {
                continue;
            }
            let fam = family_of(req);
            if composed && per_family.get(fam).copied().unwrap_or(0) >= family_cap(fam) {
                continue;
            }
            // rows of one genre page all lean the genre's way, so two of them may sit closer than two rows of Home
            let max_sim = if self.genre_spec.is_some() { REQUEST_MAX_ANCHOR_SIM_GENRE } else { REQUEST_MAX_ANCHOR_SIM };
            // tier rows (note 103) are told apart by facts, not by their centres: every cartoon tier sits in the same corner of the vector space
            let alike = req.def.kind != "tier" && placed.iter().any(|a| cosine_sim(a, &req.anchor) > max_sim);
            if alike || !names.insert(req.def.name.to_lowercase()) {
                continue;
            }
            let Some(mut row) = self.request_row(req, &taken) else { continue };
            row.meta = Some(serde_json::json!({ "request": req.def.id, "request_order": made.len() }));
            // version 2 keeps only the first screen of a row free of titles already shown; version 1 the whole row
            taken.extend(row.items.iter().take(if composed { FIRST_SCREEN } else { usize::MAX }).map(|i| i.id));
            placed.push(&req.anchor);
            *per_kind.entry(req.def.kind.as_str()).or_insert(0) += 1;
            *per_family.entry(fam).or_insert(0) += 1;
            made.push(row);
        }
        tracing::debug!("request page {} {:?}: {} requests apply, {} rows made", self.surface, self.genre, scored_n, made.len());
        if composed { made } else { spread_by_kind(made) }
    }

    /// One request as a row: its members this page may show and that are not on the page yet, ranked by closeness to the request's centre (and a little quality), made
    /// diverse, then checked: the row needs REQUEST_MIN_ITEMS titles and must keep its name's promise (genre, decade and nationality words, see row_label::name_fits).
    /// A composed row goes as deep as its depth allows: its first DIVERSE_HEAD titles are made diverse, the rest follow the ranking.
    pub(crate) fn request_row(&self, req: &Request, taken: &HashSet<u32>) -> Option<Row> {
        // a composed row goes as deep as the page's titles allow (Netflix's rows hold 150 when the supply is there; a row's median on Netflix only reflects its pages' scopes)
        let depth = if req.free { MAX_ROW_DEPTH } else { ROW_SIZE };
        let min_items = if req.free && self.genre_spec.is_some() { FIRST_SCREEN } else { REQUEST_MIN_ITEMS };
        let look = if req.free { REQUEST_LOOK.max(depth * 3) } else { REQUEST_LOOK };
        // a row that is not about children, families or animation carries no children's titles on a page that is not a child's
        let about_children = req.def.blocks.iter().any(|b| b.bucket == "tier")
            || req.def.rules.genre_specs.iter().any(|g| matches!(g.as_str(), "Family" | "Kids" | "Animation"))
            || req.def.profile.as_deref().is_some_and(|p| p == "kids" || p == "both")
            || { let n = req.def.name.to_lowercase(); n.contains("kids") || n.contains("family") || n.contains("cartoon") || n.contains("animat") || n.contains("little") };
        let keep_children = self.is_kids() || !req.free || about_children;
        let eligible: Vec<&CatalogItem> = req
            .members
            .iter()
            .filter_map(|k| self.catalog.get(k))
            .filter(|i| !taken.contains(&i.id) && self.page_pool_member(i) && (keep_children || !self.childrens_title(i)))
            .take(look)
            .collect();
        if req.def.kind == "tier" {
            tracing::debug!("tier row {}: members {}, eligible {} (kids page {})", req.def.name, req.members.len(), eligible.len(), self.is_kids());
        }
        if eligible.len() < min_items {
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
        // Animation is a world of its own: on a page that is not a child's, a row whose rules do not ask for animation or anime holds at most a share of animated
        // titles and none in its first tiles (Netflix's "Exciting Series" has one animated title in five; ours had twelve of twelve before this rule)
        let about_animation = req.def.rules.anime == Some(true)
            || req.def.rules.genre_specs.iter().any(|g| g == "Animation" || g == "anime")
            || req.def.rules.genres_all.iter().flatten().any(|g| g == "Animation")
            || req.def.name.to_lowercase().contains("anim")
            || req.def.name.to_lowercase().contains("anime");
        let mut diverse_pool = REQUEST_RANKED;
        if req.free && !self.is_kids() && !about_animation {
            let animated = |id: &u32| self.catalog.get(id).is_some_and(|i| i.is_anime || i.genres.iter().any(|g| g == "Animation"));
            let total = ranked.len();
            let cap = (total as f32 * ANIMATION_SHARE) as usize;
            let mut kept = 0usize;
            let mut head: Vec<(f32, u32, Vec<f32>)> = Vec::with_capacity(total);
            let mut late: Vec<(f32, u32, Vec<f32>)> = Vec::new();
            for r in ranked {
                if animated(&r.1) {
                    if kept < cap { kept += 1; late.push(r); }
                } else {
                    head.push(r);
                }
            }
            // animated titles sit after the diverse head (the first DIVERSE_HEAD tiles are chosen among the others)
            let at = head.len().min(DIVERSE_HEAD);
            let tail: Vec<_> = head.split_off(at);
            ranked = head;
            ranked.extend(late);
            ranked.extend(tail);
            diverse_pool = DIVERSE_HEAD.min(ranked.len());
        }
        let items = if req.free {
            // the first screens are diverse (MMR over the best ranked), the tail is the ranking itself, with no title twice and the franchise cap on the head
            let head: Vec<(u32, Vec<f32>)> = ranked.iter().take(diverse_pool).map(|(_, id, v)| (*id, v.clone())).collect();
            let picked = greedy_mmr(&req.anchor, &head, DIVERSE_HEAD.min(depth), MMR_LAMBDA);
            let mut items = self.diverse_titles(&picked, DIVERSE_HEAD.min(depth));
            let mut seen: HashSet<u32> = items.iter().map(|i| i.id).collect();
            for (_, id, _) in &ranked {
                if items.len() >= depth {
                    break;
                }
                if seen.insert(*id) {
                    if let Some(it) = self.catalog.get(id) {
                        items.push(RowItem::from_catalog(it));
                    }
                }
            }
            items
        } else {
            ranked.truncate(REQUEST_RANKED);
            let candidates: Vec<(u32, Vec<f32>)> = ranked.into_iter().map(|(_, id, v)| (id, v)).collect();
            let picked = greedy_mmr(&req.anchor, &candidates, ROW_SIZE * 2, MMR_LAMBDA);
            self.diverse_titles(&picked, ROW_SIZE)
        };
        if items.len() < min_items {
            return None;
        }
        let row_catalog: Vec<&CatalogItem> = items.iter().filter_map(|i| self.catalog.get(&i.id)).collect();
        if !crate::row_label::name_fits(&req.def.name, &row_catalog) {
            if req.def.kind == "tier" {
                tracing::debug!("tier row {}: name does not fit its {} titles", req.def.name, row_catalog.len());
            }
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
