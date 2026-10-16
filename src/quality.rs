use crate::catalog::CatalogItem;

/// Bayesian quality score constants (from streaming_algorithms_reference_manual.md)
/// CalibratedRating(i) = (pop × vote + C × μ₀) / (pop + C)
/// Quality(i) = 2.0 × CalibratedRating + 4.5 × ln(1 + pop)
const C: f32 = 25.0;
const MU_0: f32 = 6.0;

/// Dirichlet-Normal Bayesian calibrated rating.
pub fn calibrated_rating(vote_average: f32, popularity: f32) -> f32 {
    (popularity * vote_average + C * MU_0) / (popularity + C)
}

/// Full quality score. Combines calibrated vote with log-popularity confidence.
pub fn bayesian_quality_score(vote_average: f32, popularity: f32) -> f32 {
    2.0 * calibrated_rating(vote_average, popularity)
        + 4.5 * (1.0 + popularity).ln()
}

/// Popularity-adjusted score for ANN result reranking.
/// Penalises extremely popular items to counter embedding-space density bias.
/// The coefficient 0.08 is a tunable config — start here, adjust empirically.
pub fn rerank_score(cosine_sim: f32, popularity: f32) -> f32 {
    cosine_sim - 0.08 * popularity.ln_1p()
}

/// Quality score of a title for ranking members. Uses the vote count as the confidence behind the rating when it is
/// known (catalog_v2, pulled titles); falls back to popularity for titles that were never pulled.
pub fn item_quality(item: &CatalogItem) -> f32 {
    if item.vote_count > 0 {
        let vc = item.vote_count as f32;
        let calibrated = (vc * item.vote_average + C * MU_0) / (vc + C);
        2.0 * calibrated + 4.5 * (1.0 + item.popularity).ln()
    } else {
        bayesian_quality_score(item.vote_average, item.popularity)
    }
}

/// Sort items best quality first, ties by id (a page must not depend on the hash map order of the catalog). The score is computed once per item, not once per comparison.
pub fn sort_by_quality(items: &mut Vec<&CatalogItem>) {
    let mut keyed: Vec<(f32, &CatalogItem)> = items.iter().map(|i| (item_quality(i), *i)).collect();
    keyed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
    items.clear();
    items.extend(keyed.into_iter().map(|k| k.1));
}
