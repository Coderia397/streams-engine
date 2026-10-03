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
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut selected_vecs: Vec<&Vec<f32>> = Vec::with_capacity(k);

    while selected.len() < k && selected.len() < candidates.len() {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = None;

        for (i, (_, vec)) in candidates.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }

            let relevance = cosine_sim(query, vec);

            // Max similarity to any already-selected item
            let max_sim_to_selected = selected_vecs
                .iter()
                .map(|s| cosine_sim(vec, s))
                .fold(f32::NEG_INFINITY, f32::max);

            let score = if selected.is_empty() {
                relevance
            } else {
                lambda * relevance - (1.0 - lambda) * max_sim_to_selected
            };

            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            selected_vecs.push(&candidates[idx].1);
            selected.push(idx);
        } else {
            break;
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
