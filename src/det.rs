//! Deterministic hash collections for everything the engine builds from its own data (titles, groups, rows).
//!
//! `std::collections::HashMap` seeds its hasher at random for every map, so two maps with the same keys iterate in different orders, in every process and after every
//! catalog reload. A page that breaks a tie by taking the first item of an iteration then changes between requests and restarts: a session's page reshuffled when the catalog
//! reloaded, the pages of an infinite scroll did not line up, and a refactor could not be proven to change nothing. These aliases use a fixed hasher, so the same keys
//! inserted in the same order always iterate in the same order. Code that picks a winner among equals should still break the tie by an explicit key (see `page::text`).
//!
//! Not for maps keyed by what a client sends (session ids, tokens): those keep the standard hasher, which resists crafted collisions.
use rustc_hash::FxBuildHasher;

pub type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
pub type HashSet<K> = std::collections::HashSet<K, FxBuildHasher>;

/// A map with room for `n` entries.
pub fn map_with_capacity<K, V>(n: usize) -> HashMap<K, V> {
    HashMap::with_capacity_and_hasher(n, FxBuildHasher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_keys_iterate_in_the_same_order_in_separate_maps() {
        let build = || {
            let mut m: HashMap<u32, u32> = HashMap::default();
            for k in (0..2000).map(|i| i * 7919 % 10007) {
                m.insert(k, k);
            }
            m.keys().copied().collect::<Vec<u32>>()
        };
        assert_eq!(build(), build());
    }
}
