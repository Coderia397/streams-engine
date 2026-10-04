use crate::vectors::cosine_sim;

/// Greedy Maximal Marginal Relevance for intra-row diversity.
/// Selects `k` items from `candidates` that balance:
///   - Relevance to `query` (cosine similarity)
///   - Dissimilarity to already-selected items
///
/// λ = 1.0 → pure relevance (no diversity)
/// λ = 0.0 → pure diversity (no relevance)
/// λ = 0.72 → production default (Netflix-adjacent)
pub fn greedy_mmr(
    query: &[f32],
    candidates: &[(u32, Vec<f32>)], // (id, unit-vector)
    k: usize,
    lambda: f32,
) -> Vec<u32> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    // Relevance is computed once; the best similarity to the selected set is updated against the newest pick
    // only, so each pick costs n dot products instead of n x |selected| (same result, about 15x less work).
    let relevance: Vec<f32> = candidates.iter().map(|(_, v)| cosine_sim(query, v)).collect();
    let mut max_sim = vec![f32::NEG_INFINITY; n];
    let mut taken = vec![false; n];
    let mut selected: Vec<usize> = Vec::with_capacity(k);

    while selected.len() < k && selected.len() < n {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = None;
        for i in 0..n {
            if taken[i] {
                continue;
            }
            let score = if selected.is_empty() {
                relevance[i]
            } else {
                lambda * relevance[i] - (1.0 - lambda) * max_sim[i]
            };
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }
        let Some(idx) = best_idx else { break };
        taken[idx] = true;
        selected.push(idx);
        for i in 0..n {
            if !taken[i] {
                let s = cosine_sim(&candidates[i].1, &candidates[idx].1);
                if s > max_sim[i] {
                    max_sim[i] = s;
                }
            }
        }
    }

    selected.iter().map(|&i| candidates[i].0).collect()
}

/// Compute the medoid of a set of vectors: the item with highest average
/// cosine similarity to all other items. More stable than the mathematical
/// centroid because it's always an actual catalog item.
pub fn find_medoid_idx(vecs: &[Vec<f32>]) -> usize {
    let n = vecs.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }

    (0..n)
        .map(|i| {
            let avg_sim = (0..n)
                .filter(|&j| j != i)
                .map(|j| cosine_sim(&vecs[i], &vecs[j]))
                .sum::<f32>()
                / (n - 1) as f32;
            (i, avg_sim)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Filter cluster items: remove any whose cosine to the medoid is < threshold.
/// Returns (filtered_ids, filtered_vecs).
///
/// Threshold 0.62 is the empirically-derived minimum for coherent clusters
/// in 2560d Qwen3-Embedding space (concentration of measure in high dimensions
/// means random pairs already score ~0.45–0.65 — 0.30 is not a filter at all).
pub fn filter_outliers(
    ids: &[u32],
    vecs: &[Vec<f32>],
    threshold: f32,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let medoid_idx = find_medoid_idx(vecs);
    let medoid = &vecs[medoid_idx];

    ids.iter()
        .zip(vecs.iter())
        .filter(|(_, v)| cosine_sim(v, medoid) >= threshold)
        .map(|(&id, v)| (id, v.clone()))
        .unzip()
}

/// Mean pairwise cosine similarity of a cluster. Used as the coherence gate.
pub fn mean_pairwise_cosine(vecs: &[Vec<f32>]) -> f32 {
    let n = vecs.len();
    if n < 2 {
        return 1.0;
    }
    let mut total = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total += cosine_sim(&vecs[i], &vecs[j]);
            count += 1;
        }
    }
    total / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward version (recomputes similarity to every selected item each pick).
    fn reference_mmr(query: &[f32], candidates: &[(u32, Vec<f32>)], k: usize, lambda: f32) -> Vec<u32> {
        let mut selected: Vec<usize> = Vec::new();
        while selected.len() < k && selected.len() < candidates.len() {
            let mut best = (f32::NEG_INFINITY, None);
            for (i, (_, v)) in candidates.iter().enumerate() {
                if selected.contains(&i) { continue; }
                let rel = cosine_sim(query, v);
                let max_sim = selected.iter().map(|&j| cosine_sim(v, &candidates[j].1)).fold(f32::NEG_INFINITY, f32::max);
                let score = if selected.is_empty() { rel } else { lambda * rel - (1.0 - lambda) * max_sim };
                if score > best.0 { best = (score, Some(i)); }
            }
            match best.1 { Some(i) => selected.push(i), None => break }
        }
        selected.iter().map(|&i| candidates[i].0).collect()
    }

    #[test]
    fn incremental_mmr_matches_the_reference() {
        let mut state = 12345u64;
        let mut rnd = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5 };
        let unit = |v: Vec<f32>| { let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.into_iter().map(|x| x / n).collect::<Vec<f32>>() };
        for trial in 0..5 {
            let cands: Vec<(u32, Vec<f32>)> = (0..60).map(|i| (i as u32 + trial * 100, unit((0..32).map(|_| rnd()).collect()))).collect();
            let q = unit((0..32).map(|_| rnd()).collect());
            for &(k, lam) in &[(10usize, 0.72f32), (30, 0.72), (60, 0.5), (5, 1.0)] {
                assert_eq!(greedy_mmr(&q, &cands, k, lam), reference_mmr(&q, &cands, k, lam), "trial {trial} k {k} lambda {lam}");
            }
        }
        assert!(greedy_mmr(&[1.0], &[], 5, 0.7).is_empty());
    }
}
