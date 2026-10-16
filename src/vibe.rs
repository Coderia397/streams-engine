//! Vibe, mood and seasonal rows (notes/52). Three small tables of the database are read at start:
//!  vibe members   every title's vibe group (`titles.vf`, core members only), see groups.rs
//!  `mood_pointers`  per mood (late night, cozy, mind-bending ...) the vibe groups that fit it best, with a score
//!  `season_groups` and `season_members`  per calendar event the sub groups of its pool (named) and their titles
use crate::catalog::item_key;
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone)]
pub struct Mood {
    pub id: String,
    pub title: String,
    /// Shown first on a late night page
    pub late: bool,
    /// Vibe groups (fine) that fit the mood, best first, with their score
    pub groups: Vec<(i32, f32)>,
    /// The mood phrase embedded in query form (1,024 dimensions, not centred); empty when the table is missing
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SeasonSub {
    pub name: String,
    /// Item keys, best fit first
    pub items: Vec<u32>,
    /// Share of the members that carry the event's catalog keywords (the others joined by similarity to the event text)
    pub kw_share: f32,
}

#[derive(Debug, Default)]
pub struct VibeIndex {
    /// vibe group (fine) to the item keys of its core members
    pub members: HashMap<i32, Vec<u32>>,
    pub moods: Vec<Mood>,
    /// calendar event to its named sub groups, largest first
    pub seasons: HashMap<String, Vec<SeasonSub>>,
}

pub fn load(db_path: &str, members: HashMap<i32, Vec<u32>>) -> VibeIndex {
    let mut idx = VibeIndex { members, ..Default::default() };
    let Ok(conn) = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY) else { return idx };
    if let Ok(mut stmt) = conn.prepare("SELECT mood, title, late, group_id, score FROM mood_pointers ORDER BY mood, rank") {
        if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)? != 0, r.get::<_, i64>(3)? as i32, r.get::<_, f64>(4)? as f32))) {
            for (id, title, late, g, score) in rows.flatten() {
                match idx.moods.iter_mut().find(|m| m.id == id) {
                    Some(m) => m.groups.push((g, score)),
                    None => idx.moods.push(Mood { id, title, late, groups: vec![(g, score)], vector: Vec::new() }),
                }
            }
        }
    }
    if let Ok(mut stmt) = conn.prepare("SELECT mood, data FROM mood_vectors") {
        if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))) {
            for (id, blob) in rows.flatten() {
                if let Some(m) = idx.moods.iter_mut().find(|m| m.id == id) { m.vector = crate::vectors::le_f32s(&blob); }
            }
        }
    }
    let mut names: HashMap<(String, i32), String> = HashMap::new();
    if let Ok(mut stmt) = conn.prepare("SELECT event, sub, name FROM season_groups") {
        if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as i32, r.get::<_, String>(2)?))) {
            for (e, s, n) in rows.flatten() { names.insert((e, s), n); }
        }
    }
    if let Ok(mut stmt) = conn.prepare("SELECT event, sub, media_type, tmdb_id, kw FROM season_members ORDER BY event, sub, rank") {
        if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as i32, r.get::<_, String>(2)?, r.get::<_, i64>(3)? as u32, r.get::<_, i64>(4)? != 0))) {
            let mut subs: HashMap<(String, i32), (Vec<u32>, usize)> = HashMap::new();
            for (e, s, mt, id, kw) in rows.flatten() { let v = subs.entry((e, s)).or_default(); v.0.push(item_key(&mt, id)); v.1 += kw as usize; }
            for ((e, s), (items, kw)) in subs {
                let kw_share = kw as f32 / items.len().max(1) as f32;
                if let Some(name) = names.get(&(e.clone(), s)) { idx.seasons.entry(e).or_default().push(SeasonSub { name: name.clone(), items, kw_share }); }
            }
            for v in idx.seasons.values_mut() { v.sort_by(|a, b| b.items.len().cmp(&a.items.len()).then(a.name.cmp(&b.name))); }
        }
    }
    info!("Vibe index: {} vibe groups, {} moods, {} seasonal events", idx.members.len(), idx.moods.len(), idx.seasons.len());
    idx
}
