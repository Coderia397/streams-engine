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
    /// Composed rows (requests_v2.json, note 101): how deep the row goes (Netflix's median for a reproduced name, else up to 150), the format its name claims
    /// ("movie", "tv" or "both"), the profile kind Netflix showed it to ("adult", "kids", "both" or "any"), the Netflix name it reproduces, and its template
    pub depth: Option<usize>,
    pub format: Option<String>,
    pub profile: Option<String>,
    pub netflix_name: Option<String>,
    pub template: Option<String>,
    /// The blocks the row is composed of (bucket and token), for page rules that look at what a row is about
    #[serde(default)]
    pub blocks: Vec<BlockRef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlockRef {
    pub bucket: String,
    pub token: String,
}

#[derive(Deserialize)]
struct RequestFile {
    version: Option<u32>,
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
    /// A composed row (version 2) runs on every page whose scope its titles fit instead of one owner page, as Netflix's rows do (note 97: the same row name on 49 to 79 pages)
    pub free: bool,
}

#[derive(Default)]
pub struct RequestStore {
    pub requests: Vec<Request>,
    /// 2 for composed rows (requests_v2.json), 1 for the older request file, 0 when empty
    pub version: u32,
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
    /// Reads `requests_v2.json` (composed rows), else `requests_v1.json`, next to `db_path`; an empty store when neither is readable (request rows are then off
    /// whatever the switch says).
    pub fn load(db_path: &str, catalog: &CatalogMap, vectors: &VectorStore, mean: &[f32]) -> Self {
        let Some(dir) = std::path::Path::new(db_path).parent() else { return Self::default() };
        let Some((path, text)) = ["requests_v2.json", "requests_v1.json"].iter().find_map(|n| { let p = dir.join(n); std::fs::read_to_string(&p).ok().map(|t| (p, t)) }) else { return Self::default() };
        let file: RequestFile = match serde_json::from_str(&text) {
            Ok(f) => f,
            Err(e) => {
                warn!("{} is not readable: {e}", path.display());
                return Self::default();
            }
        };
        let started = std::time::Instant::now();
        let version = file.version.unwrap_or(1);
        // every title once, in a fixed order, so members come out the same in every process
        let mut items: Vec<&crate::catalog::CatalogItem> = catalog.values().collect();
        items.sort_by_key(|i| i.id);
        let requests: Vec<Request> = file
            .requests
            .into_par_iter()
            .map(|def| {
                let format = def.format.clone().unwrap_or_else(|| "both".to_string());
                let mut members: Vec<&crate::catalog::CatalogItem> = items.iter().copied().filter(|i| (format == "both" || i.media_type == format) && def.rules.passes(i)).collect();
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
                // children's cartoons rarely reach the quality gate (few votes): a row with no quality member takes its centre from its first members instead
                if n == 0 {
                    for it in members.iter().take(400) {
                        if let Some(ix) = vectors.index_of(it.id) {
                            let mut c: Vec<f32> = vectors.get_f32(ix).iter().zip(mean).map(|(x, m)| x - m).collect();
                            l2_normalize(&mut c);
                            for (a, x) in anchor.iter_mut().zip(&c) {
                                *a += x;
                            }
                            n += 1;
                        }
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
                let free = version >= 2;
                let (owner_surface, owner_genre) = if free {
                    (match format.as_str() { "movie" => "movie", "tv" => "tv", _ => "both" }, None)
                } else {
                    choose_owner(&def, &members)
                };
                Request { members: members.iter().map(|i| i.id).collect(), anchor, season, def, appeal, kid_share, owner_surface, owner_genre, free }
            })
            .collect();
        info!("Requests loaded: {} (version {version}) from {} in {:.1} s", requests.len(), path.display(), started.elapsed().as_secs_f32());
        Self { requests, version }
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}
