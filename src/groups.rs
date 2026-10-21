//! Group ids of every title from the single database (`titles` columns w1, w2, atom, th2, th3, from ai-engine/store/build_store.py): world level 1 and 2, atom (about 48 titles)
//! and bottom up theme groups. Used to boost recommendations inside the same group; empty when the columns are missing.
use crate::catalog::item_key;
use rusqlite::{Connection, OpenFlags};
use crate::det::HashMap;
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct GroupIds {
    pub w1: i32,
    pub w2: i32,
    pub atom: i32,
    pub theme: i32,
    pub th1: i32,
    pub th3: i32,
    /// Vibe group (fine) and mid vibe group of the title, -1 unless the title is a core member (reviewed, or placed with confidence 0.3 or more, in a group that is not an artifact)
    pub vf: i32,
}

pub type GroupMap = HashMap<u32, GroupIds>;

pub fn load(db_path: &str) -> GroupMap {
    let mut map = GroupMap::default();
    let Ok(conn) = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY) else { return map };
    let Ok(mut stmt) = conn.prepare("SELECT media_type, tmdb_id, w1, w2, atom, th2, th1, th3, CASE WHEN vcore = 1 THEN vf END FROM titles WHERE w1 IS NOT NULL") else { return map };
    let Ok(rows) = stmt.query_map([], |r| {
        let mt: String = r.get(0)?;
        let id: i64 = r.get(1)?;
        Ok((item_key(&mt, id as u32), GroupIds { w1: r.get::<_, Option<i64>>(2)?.unwrap_or(-1) as i32, w2: r.get::<_, Option<i64>>(3)?.unwrap_or(-1) as i32, atom: r.get::<_, Option<i64>>(4)?.unwrap_or(-1) as i32, theme: r.get::<_, Option<i64>>(5)?.unwrap_or(-1) as i32, th1: r.get::<_, Option<i64>>(6)?.unwrap_or(-1) as i32, th3: r.get::<_, Option<i64>>(7)?.unwrap_or(-1) as i32, vf: r.get::<_, Option<i64>>(8)?.unwrap_or(-1) as i32 }))
    }) else { return map };
    for r in rows.flatten() { map.insert(r.0, r.1); }
    info!("Group ids loaded for {} titles", map.len());
    map
}

/// Hierarchy levels from the finest to the coarsest: 0 world level 2 (3,138 groups), 1 themes level 3 (621), 2 themes level 2 (386), 3 themes level 1 (141), 4 world level 1 (47).
pub const N_LEVELS: u8 = 5;

impl GroupIds {
    pub fn at(&self, level: u8) -> i32 {
        match level { 0 => self.w2, 1 => self.th3, 2 => self.theme, 3 => self.th1, _ => self.w1 }
    }
}

pub type GroupNames = HashMap<(u8, i32), String>;

/// Names of the groups (table `groups`, written by the naming run), keyed by hierarchy level and group id.
pub fn load_names(db_path: &str) -> GroupNames {
    let mut map = GroupNames::default();
    let Ok(conn) = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY) else { return map };
    let Ok(mut stmt) = conn.prepare("SELECT kind, level, id, name FROM groups WHERE kind IN ('world','theme','vibe') AND name IS NOT NULL AND name != ''") else { return map };
    let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?))) else { return map };
    for (kind, level, id, name) in rows.flatten() {
        let code = match (kind.as_str(), level) { ("vibe", 1) => 5, ("vibe", 2) => 6, ("world", 2) => 0, ("theme", 3) => 1, ("theme", 2) => 2, ("theme", 1) => 3, ("world", 1) => 4, _ => continue };
        map.insert((code, id as i32), name);
    }
    info!("Group names loaded: {}", map.len());
    map
}

/// Members of every vibe group (fine level), for rows: group id to the item keys of its core members.
pub fn vibe_members(groups: &GroupMap) -> HashMap<i32, Vec<u32>> {
    let mut m: HashMap<i32, Vec<u32>> = HashMap::default();
    for (k, g) in groups { if g.vf >= 0 { m.entry(g.vf).or_default().push(*k); } }
    for v in m.values_mut() { v.sort_unstable(); }
    m
}
