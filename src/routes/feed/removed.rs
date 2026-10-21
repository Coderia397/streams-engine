//! Rows taken off every page by title (`blocked_rows.json` next to the database: a JSON list of row titles, compared without case).
//! The file is read again only when it changes, so rows can be removed while the engine runs. This is a stopgap for rows the generators should never have made;
//! the way out is to fix the generator, then delete the entry.
use crate::det::HashSet;
use crate::live::LiveFile;
use crate::page::Row;
use std::sync::Arc;

static BLOCKED: LiveFile<HashSet<String>> = LiveFile::new("blocked_rows.json", parse);

fn parse(text: &str) -> Option<HashSet<String>> {
    serde_json::from_str::<Vec<String>>(text).ok().map(|titles| titles.into_iter().map(|t| t.to_lowercase()).collect())
}

pub fn blocked_titles(db_path: &str) -> Arc<HashSet<String>> {
    BLOCKED.get(db_path)
}

/// The rows without the removed ones.
pub fn without_removed(rows: Vec<Row>, db_path: &str) -> Vec<Row> {
    let blocked = blocked_titles(db_path);
    if blocked.is_empty() {
        return rows;
    }
    rows.into_iter().filter(|r| !blocked.contains(&r.title.to_lowercase())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_compare_without_case_and_bad_text_blocks_nothing() {
        let set = parse(r#"["Halloween Collection", "News Programs"]"#).unwrap();
        assert!(set.contains("halloween collection"));
        assert!(parse("[").is_none());
    }
}
