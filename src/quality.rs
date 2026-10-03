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

/// Full anchor quality gate. Must pass ALL conditions.
/// - vote_average >= 7.2
/// - popularity >= 40.0
/// - quality score >= 10.0
/// - overview >= 80 chars (structural gate — eliminates news, sports, game shows)
pub fn is_quality_anchor(item: &CatalogItem) -> bool {
    item.vote_average >= 7.2
        && item.popularity >= 40.0
        && bayesian_quality_score(item.vote_average, item.popularity) >= 10.0
        && item.overview_chars() >= 80
}

/// Popularity-adjusted score for ANN result reranking.
/// Penalises extremely popular items to counter embedding-space density bias.
/// The coefficient 0.08 is a tunable config — start here, adjust empirically.
pub fn rerank_score(cosine_sim: f32, popularity: f32) -> f32 {
    cosine_sim - 0.08 * popularity.ln_1p()
}
