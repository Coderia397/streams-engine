//! Group ids of every title from the single database (`titles` columns w1, w2, atom, th2, th3, from ai-engine/store/build_store.py): world level 1 and 2, atom (about 48 titles)
//! and bottom up theme groups. Used to boost recommendations inside the same group; empty when the columns are missing.
use crate::catalog::item_key;
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct GroupIds {
    pub w1: i32,
    pub w2: i32,
    pub atom: i32,
    pub theme: i32,
}

pub type GroupMap = HashMap<u32, GroupIds>;

pub fn load(db_path: &str) -> GroupMap {
    let mut map = GroupMap::new();
    let Ok(conn) = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY) else { return map };
    let Ok(mut stmt) = conn.prepare("SELECT media_type, tmdb_id, w1, w2, atom, th2 FROM titles WHERE w1 IS NOT NULL") else { return map };
    let Ok(rows) = stmt.query_map([], |r| {
        let mt: String = r.get(0)?;
        let id: i64 = r.get(1)?;
        Ok((item_key(&mt, id as u32), GroupIds { w1: r.get::<_, Option<i64>>(2)?.unwrap_or(-1) as i32, w2: r.get::<_, Option<i64>>(3)?.unwrap_or(-1) as i32, atom: r.get::<_, Option<i64>>(4)?.unwrap_or(-1) as i32, theme: r.get::<_, Option<i64>>(5)?.unwrap_or(-1) as i32 }))
    }) else { return map };
    for r in rows.flatten() { map.insert(r.0, r.1); }
    info!("Group ids loaded for {} titles", map.len());
    map
}
