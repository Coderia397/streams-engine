use crate::catalog::CatalogMap;
use crate::quality::bayesian_quality_score;
use crate::vectors::{l2_normalize, VectorStore, DIMS};
use anyhow::Result;
use crate::det::HashMap;
use std::sync::Arc;
use tracing::info;

/// Known TMDB genre names we compute centroids for.
/// Derived from the TMDB taxonomy — covers all major content categories.
pub const GENRE_LABELS: &[&str] = &[
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western", "Reality", "Talk",
    "News", "Soap", "Kids", "Action & Adventure", "Sci-Fi & Fantasy",
    "War & Politics", "Anime",
];

/// Vault of L2-normalised genre centroids computed at startup from the
/// top-quality catalog items. Used for:
/// - Taste-vector query blending (?taste= param → weighted centroid)
/// - Mood adjective projection (closest centroid name)
/// - BYW anchor domain gate (cosine ≥ 0.55 to any entertainment centroid)
pub struct CentroidVault {
    /// genre label → unit vector (2560d)
    centroids: HashMap<String, Vec<f32>>,
}

impl CentroidVault {
    /// Compute genre centroids via Bayesian Fréchet means.
    ///
    /// Algorithm:
    /// 1. Filter catalog to top-quality items (Bayesian score ≥ threshold)
    /// 2. For each genre, accumulate weighted unit vectors (weight = quality score)
    /// 3. Normalise the accumulated sum to the unit sphere
    ///
    /// Takes ~280ms at 100k catalog, single SQL pass already loaded in memory.
    pub fn compute(catalog: &CatalogMap, vectors: &VectorStore) -> Result<Arc<Self>> {
        use rayon::prelude::*;
        // the top-quality items with a vector: (row, score, indices into GENRE_LABELS)
        let eligible: Vec<(usize, f32, Vec<usize>)> = catalog
            .iter()
            .filter_map(|(catalog_id, item)| {
                let score = bayesian_quality_score(item.vote_average, item.popularity);
                if score < 8.0 {
                    return None; // only high-quality items define genre centroids
                }
                let genres: Vec<usize> = item.genres.iter().filter_map(|g| GENRE_LABELS.iter().position(|l| *l == g.as_str())).collect();
                if genres.is_empty() {
                    return None;
                }
                Some((vectors.index_of(*catalog_id)?, score, genres))
            })
            .collect();
        let g = GENRE_LABELS.len();
        // weighted sums per genre, accumulated in parallel over the f16 rows without materialising them
        let (mut sums, counts) = eligible
            .par_iter()
            .fold(
                || (vec![vec![0.0f32; DIMS]; g], vec![0usize; g]),
                |(mut sums, mut counts), (idx, score, genres)| {
                    let row = vectors.row(*idx);
                    for &gi in genres {
                        for (a, v) in sums[gi].iter_mut().zip(row) {
                            *a += score * v.to_f32();
                        }
                        counts[gi] += 1;
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![vec![0.0f32; DIMS]; g], vec![0usize; g]),
                |(mut sa, mut ca), (sb, cb)| {
                    for (a, b) in sa.iter_mut().zip(sb) {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x += y;
                        }
                    }
                    for (x, y) in ca.iter_mut().zip(cb) {
                        *x += y;
                    }
                    (sa, ca)
                },
            );

        // Normalise each centroid to unit sphere
        let mut centroids = HashMap::default();
        for (gi, label) in GENRE_LABELS.iter().enumerate() {
            if counts[gi] > 0 {
                let mut vec = std::mem::take(&mut sums[gi]);
                l2_normalize(&mut vec);
                centroids.insert(label.to_string(), vec);
            }
        }

        info!(
            "CentroidVault: computed {} genre centroids from top-quality items",
            centroids.len()
        );
        Ok(Arc::new(Self { centroids }))
    }

    /// Build a taste-query vector from a sparse genre-affinity map.
    /// query = normalise(Σ weight_i × centroid_i)
    /// This is mathematically valid for cosine ANN search on the unit sphere.
    pub fn blend_taste(&self, taste: &[(String, f32)]) -> Vec<f32> {
        let mut result = vec![0.0f32; DIMS];
        for (genre, weight) in taste {
            if let Some(centroid) = self.centroids.get(genre.as_str()) {
                for (r, c) in result.iter_mut().zip(centroid.iter()) {
                    *r += weight * c;
                }
            }
        }
        l2_normalize(&mut result);
        result
    }

}
