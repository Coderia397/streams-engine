//! Search by person: an index from the names of directors and top billed cast to the titles they are in, built once from the catalog
//! database at start. The catalog's full text index covers title, overview, genres and director only, so a search for an actor would
//! otherwise find nothing.
use rusqlite::Connection;
use std::collections::HashMap;

/// Titles below this popularity are left out of the index (kept small; a person is found through the titles people watch).
const MIN_POPULARITY: f64 = 1.5;
const CAST_PER_TITLE: usize = 6;

#[derive(Default)]
pub struct PeopleIndex {
    /// normalised name -> (internal item key, popularity), unsorted
    names: HashMap<String, Vec<(u32, f32)>>,
}

/// Lower case, letters and digits only, single spaces: "Joseph Gordon-Levitt" becomes "joseph gordon levitt".
pub fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = true;
    for c in text.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
            space = false;
        } else if !space {
            out.push(' ');
            space = true;
        }
    }
    out.trim_end().to_string()
}

impl PeopleIndex {
    pub fn load(db_path: &str) -> Self {
        let Ok(conn) = Connection::open(db_path) else { return Self::default() };
        let Ok(mut stmt) = conn.prepare("SELECT id, popularity, director, cast_members FROM catalog WHERE popularity >= ?1 AND (cast_members IS NOT NULL OR director IS NOT NULL)") else {
            return Self::default();
        };
        let mut names: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        let rows = stmt.query_map([MIN_POPULARITY], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, Option<String>>(2)?, r.get::<_, Option<String>>(3)?))
        });
        if let Ok(rows) = rows {
            for (id, pop, director, cast) in rows.filter_map(|r| r.ok()) {
                let people = director.iter().flat_map(|d| d.split(',')).chain(cast.iter().flat_map(|c| c.split(',').take(CAST_PER_TITLE)));
                let mut seen: Vec<String> = Vec::new();
                for p in people {
                    let n = normalise(p);
                    if n.chars().count() >= 5 && !seen.contains(&n) {
                        names.entry(n.clone()).or_default().push((id as u32, pop as f32));
                        seen.push(n);
                    }
                }
            }
        }
        Self { names }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Titles of the people the query names, most popular first. A query matches a person when the person's name is the query or starts with it
    /// (a single word must be a whole name, so a first name alone does not pull in everyone). The first element tells whether the query was an exact name.
    pub fn titles_for(&self, query: &str, limit: usize) -> (bool, Vec<u32>) {
        let q = normalise(query);
        if q.chars().count() < 5 {
            return (false, Vec::new());
        }
        let exact = self.names.contains_key(&q);
        let mut best: HashMap<u32, f32> = HashMap::new();
        for (name, titles) in &self.names {
            if name == &q || (q.contains(' ') && name.starts_with(&q)) {
                for &(k, p) in titles {
                    let e = best.entry(k).or_insert(0.0);
                    *e = e.max(p);
                }
            }
        }
        let mut v: Vec<(u32, f32)> = best.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        (exact, v.into_iter().take(limit).map(|(k, _)| k).collect())
    }
}

/// True when `a` and `b` differ by at most `max` single letter edits (insert, delete, replace, or swap of two neighbours).
pub fn within_edits(a: &str, b: &str, max: usize) -> bool {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    if a.len().abs_diff(b.len()) > max {
        return false;
    }
    let mut prev2: Vec<usize> = vec![0; b.len() + 1];
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        let mut row_min = cur[0];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut v = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(prev2[j - 2] + 1);
            }
            cur[j] = v;
            row_min = row_min.min(v);
        }
        if row_min > max {
            return false;
        }
        prev2 = prev;
        prev = cur;
    }
    prev[b.len()] <= max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_normalised() {
        assert_eq!(normalise("Joseph Gordon-Levitt"), "joseph gordon levitt");
        assert_eq!(normalise("  Zoë  Saldaña "), "zoë saldaña");
    }

    #[test]
    fn small_typos_are_within_reach() {
        assert!(within_edits("stranger thngs", "stranger things", 1));
        assert!(within_edits("inceptoin", "inception", 1));
        assert!(!within_edits("avatar", "avenger", 1));
        assert!(within_edits("the offce", "the office", 1));
        assert!(!within_edits("a", "abcd", 2));
    }
}
