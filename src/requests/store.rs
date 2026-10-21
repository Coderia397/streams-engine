//! The requests the engine can make rows from (`requests_v1.json` next to the database, written by ai-engine/pipeline/make_requests.py), with what is worked out once at start:
//! the titles that pass each request's rules (best quality first) and the centre of those titles in the vector space (the anchor rows are ranked by).
use super::rules::Rules;
use crate::catalog::CatalogMap;
use crate::taxonomy::Season;
use crate::vectors::{l2_normalize, VectorStore, DIMS};
use rayon::prelude::*;
use serde::Deserialize;
use tracing::{info, warn};

#[derive(Debug, Clone, Deserialize)]
pub struct SeasonDef {
    /// MM-DD
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RequestDef {
    pub id: String,
    /// The row's name, shown as it is
    pub name: String,
    /// theme, mood, era, origin, calendar
    pub kind: String,
    pub rules: Rules,
    /// The page the request belongs to: home, films or series (by the share of films among its titles); genre pages take any request that suits their genre
    pub page: String,
    pub season: Option<SeasonDef>,
}

#[derive(Deserialize)]
struct RequestFile {
    requests: Vec<RequestDef>,
}

pub struct Request {
    pub def: RequestDef,
    pub season: Option<Season>,
    /// Keys of the catalog titles that pass the rules, best quality first
    pub members: Vec<u32>,
    /// Centre of the quality members in the centred vector space (unit length); empty when no member has a vector
    pub anchor: Vec<f32>,
    /// How widely known the request's best titles are: mean log votes of its 30 best quality members over log 20,000, 0 to 1
    pub appeal: f32,
    /// Share of its first 200 members made for children (Kids or Family genre, or animation rated U or PG)
    pub kid_share: f32,
    /// The one page this request belongs to, so no row is on two kinds of page: the surface ("both" for Home, "movie" for Films, "tv" for Series) and the genre page rule,
    /// none for the plain page. Chosen at start among the pages the request suits (its plain page, and every genre page whose genre covers most of its titles of that surface).
    pub owner_surface: &'static str,
    pub owner_genre: Option<crate::genre_spec::GenreSpec>,
}

#[derive(Default)]
pub struct RequestStore {
    pub requests: Vec<Request>,
}

/// Titles of one surface a request needs before it can belong to a genre page of that surface.
const OWNER_MIN_TITLES: usize = 30;
/// A genre page suits a request when its genre covers this share of the request's titles of that surface, and about as much as the request's most common genre.
const OWNER_GENRE_SHARE: f32 = 0.6;

/// A stable number for a request id.
fn id_number(id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    id.hash(&mut h);
    h.finish()
}

/// The page a request belongs to: one of the pages it suits, picked by its id so the requests spread evenly over the pages and the choice is the same in every process.
fn choose_owner(def: &RequestDef, members: &[&crate::catalog::CatalogItem]) -> (&'static str, Option<crate::genre_spec::GenreSpec>) {
    let plain: &'static str = match def.page.as_str() {
        "films" => "movie",
        "series" => "tv",
        _ => "both",
    };
    let mut suits: Vec<(&'static str, Option<crate::genre_spec::GenreSpec>)> = vec![(plain, None)];
    for surface in ["movie", "tv"] {
        let of_surface: Vec<&crate::catalog::CatalogItem> = members.iter().copied().filter(|i| i.media_type == surface).take(400).collect();
        if of_surface.len() < OWNER_MIN_TITLES {
            continue;
        }
        let mut families: crate::det::HashMap<&str, usize> = crate::det::HashMap::default();
        for it in &of_surface {
            for g in &it.genres {
                *families.entry(crate::row_label::genre_family(g)).or_insert(0) += 1;
            }
        }
        let top = families.values().copied().max().unwrap_or(0) as f32 / of_surface.len() as f32;
        for (_, name) in crate::genre_pages::pages(surface) {
            let spec = crate::genre_spec::resolve(name, surface);
            if matches!(spec, crate::genre_spec::GenreSpec::Any | crate::genre_spec::GenreSpec::Unsupported) || suits.iter().any(|(s, g)| *s == surface && g.as_ref() == Some(&spec)) {
                continue;
            }
            let share = of_surface.iter().filter(|i| spec.matches(i)).count() as f32 / of_surface.len() as f32;
            if share >= OWNER_GENRE_SHARE.max(top - 0.05) {
                suits.push((surface, Some(spec)));
            }
        }
    }
    let pick = (id_number(&def.id) % suits.len() as u64) as usize;
    suits.swap_remove(pick)
}

fn parse_md(s: &str) -> Option<(u8, u8)> {
    let (m, d) = s.split_once('-')?;
    Some((m.trim().parse().ok()?, d.trim().parse().ok()?))
}

impl RequestStore {
    /// Reads `requests_v1.json` next to `db_path`; an empty store when the file is missing or unreadable (request rows are then off whatever the switch says).
    pub fn load(db_path: &str, catalog: &CatalogMap, vectors: &VectorStore, mean: &[f32]) -> Self {
        let Some(path) = std::path::Path::new(db_path).parent().map(|d| d.join("requests_v1.json")) else { return Self::default() };
        let Ok(text) = std::fs::read_to_string(&path) else { return Self::default() };
        let file: RequestFile = match serde_json::from_str(&text) {
            Ok(f) => f,
            Err(e) => {
                warn!("{} is not readable: {e}", path.display());
                return Self::default();
            }
        };
        let started = std::time::Instant::now();
        // every title once, in a fixed order, so members come out the same in every process
        let mut items: Vec<&crate::catalog::CatalogItem> = catalog.values().collect();
        items.sort_by_key(|i| i.id);
        let requests: Vec<Request> = file
            .requests
            .into_par_iter()
            .map(|def| {
                let mut members: Vec<&crate::catalog::CatalogItem> = items.iter().copied().filter(|i| def.rules.passes(i)).collect();
                crate::quality::sort_by_quality(&mut members);
                let mut anchor = vec![0.0f32; DIMS];
                let mut n = 0usize;
                for it in members.iter().filter(|i| crate::page::passes_member_quality(i)) {
                    if let Some(ix) = vectors.index_of(it.id) {
                        let mut c: Vec<f32> = vectors.get_f32(ix).iter().zip(mean).map(|(x, m)| x - m).collect();
                        l2_normalize(&mut c);
                        for (a, x) in anchor.iter_mut().zip(&c) {
                            *a += x;
                        }
                        n += 1;
                    }
                }
                if n > 0 {
                    l2_normalize(&mut anchor);
                } else {
                    anchor.clear();
                }
                let season = def.season.as_ref().and_then(|s| Some(Season { start: parse_md(&s.start)?, end: parse_md(&s.end)? }));
                let best: Vec<f32> = members.iter().filter(|i| crate::page::passes_member_quality(i)).take(30).map(|i| (1.0 + i.vote_count as f32).ln()).collect();
                let appeal = if best.is_empty() { 0.0 } else { (best.iter().sum::<f32>() / best.len() as f32 / 20_001f32.ln()).clamp(0.0, 1.0) };
                let head = &members[..members.len().min(200)];
                let for_children = |i: &&&crate::catalog::CatalogItem| {
                    i.genres.iter().any(|g| g == "Kids" || g == "Family") || (i.genres.iter().any(|g| g == "Animation") && (1..=2).contains(&i.cert))
                };
                let kid_share = if head.is_empty() { 0.0 } else { head.iter().filter(for_children).count() as f32 / head.len() as f32 };
                let (owner_surface, owner_genre) = choose_owner(&def, &members);
                Request { members: members.iter().map(|i| i.id).collect(), anchor, season, def, appeal, kid_share, owner_surface, owner_genre }
            })
            .collect();
        info!("Requests loaded: {} from {} in {:.1} s", requests.len(), path.display(), started.elapsed().as_secs_f32());
        Self { requests }
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}
