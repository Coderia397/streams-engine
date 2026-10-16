//! What the client knows about a profile and sends with a feed request, so the engine stays stateless: the keys of the titles it saved (`saved`), watched or is in the middle of
//! (`recent`, most recent first) and liked (`liked`). A key is `movie:<tmdb id>` or `tv:<tmdb id>`, keys are comma separated, at most `MAX_KEYS` of each list are read.
//! The same signals work for a local profile and for a synced account; nothing is stored here.
use crate::catalog::item_key;

pub const MAX_KEYS: usize = 40;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Signals {
    pub saved: Vec<u32>,
    pub recent: Vec<u32>,
    pub liked: Vec<u32>,
}

/// Item keys of a list like `movie:550,tv:1399`: unknown shapes are skipped, repeats removed (first place wins), at most `MAX_KEYS`.
pub fn parse_keys(list: &str) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for part in list.split(',') {
        let Some((media, id)) = part.trim().split_once(':') else { continue };
        let media = media.trim().to_ascii_lowercase();
        if media != "movie" && media != "tv" {
            continue;
        }
        let Ok(id) = id.trim().parse::<u32>() else { continue };
        let key = item_key(&media, id);
        if !out.contains(&key) {
            out.push(key);
        }
        if out.len() >= MAX_KEYS {
            break;
        }
    }
    out
}

impl Signals {
    pub fn parse(saved: Option<&str>, recent: Option<&str>, liked: Option<&str>) -> Self {
        Self { saved: parse_keys(saved.unwrap_or("")), recent: parse_keys(recent.unwrap_or("")), liked: parse_keys(liked.unwrap_or("")) }
    }

    pub fn is_empty(&self) -> bool {
        self.saved.is_empty() && self.recent.is_empty() && self.liked.is_empty()
    }

    /// Short text for the cache key: two profiles with the same signals share a page, any other profile does not.
    pub fn fingerprint(&self) -> String {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.saved.hash(&mut h);
        self.recent.hash(&mut h);
        self.liked.hash(&mut h);
        format!("{:x}", h.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::TV_BIT;

    #[test]
    fn keys_parse_in_order_without_repeats_or_junk() {
        let k = parse_keys("movie:550, tv:1399,movie:550,book:1,tv:x,,movie:603");
        assert_eq!(k, vec![550, 1399 | TV_BIT, 603]);
    }

    #[test]
    fn a_list_is_cut_at_the_limit() {
        let long: Vec<String> = (1..=100).map(|i| format!("movie:{i}")).collect();
        assert_eq!(parse_keys(&long.join(",")).len(), MAX_KEYS);
    }

    #[test]
    fn the_fingerprint_follows_the_signals() {
        let a = Signals::parse(Some("movie:1"), None, None);
        let b = Signals::parse(Some("movie:1"), None, None);
        let c = Signals::parse(Some("movie:2"), None, None);
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.fingerprint(), c.fingerprint());
        assert!(Signals::default().is_empty());
    }
}
