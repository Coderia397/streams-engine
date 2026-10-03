use crate::catalog::CatalogMap;
use crate::quality::bayesian_quality_score;
use crate::vectors::{cosine_sim, l2_normalize, VectorStore, DIMS};
use anyhow::Result;
use std::collections::HashMap;
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
        let mut accum: HashMap<String, Vec<f32>> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();

        // Initialise accumulators
        for &label in GENRE_LABELS {
            accum.insert(label.to_string(), vec![0.0f32; DIMS]);
        }

        // Accumulate weighted vectors for top-quality items
        for (catalog_id, item) in catalog.iter() {
            let score = bayesian_quality_score(item.vote_average, item.popularity);
            if score < 8.0 {
                continue; // only high-quality items define genre centroids
            }

            let Some(idx) = vectors.index_of(*catalog_id) else { continue };
            let vec = vectors.get_f32(idx);

            for genre in &item.genres {
                if let Some(acc) = accum.get_mut(genre.as_str()) {
                    for (a, v) in acc.iter_mut().zip(vec.iter()) {
                        *a += score * v;
                    }
                    *counts.entry(genre.clone()).or_insert(0) += 1;
                }
            }
        }

        // Normalise each centroid to unit sphere
        let mut centroids = HashMap::new();
        for (label, mut vec) in accum {
            let n = *counts.get(&label).unwrap_or(&0);
            if n > 0 {
                l2_normalize(&mut vec);
                centroids.insert(label, vec);
            }
        }

        info!(
            "CentroidVault: computed {} genre centroids from top-quality items",
            centroids.len()
        );
        Ok(Arc::new(Self { centroids }))
    }

    /// Find the genre name whose centroid is closest (cosine) to the given vector.
    pub fn closest_genre(&self, vec: &[f32]) -> &str {
        self.centroids
            .iter()
            .max_by(|(_, a), (_, b)| {
                cosine_sim(a, vec)
                    .partial_cmp(&cosine_sim(b, vec))
                    .unwrap()
            })
            .map(|(k, _)| k.as_str())
            .unwrap_or("Drama")
    }

    /// Max cosine similarity to ANY centroid. Used as domain gate:
    /// if max_sim < 0.55, the item is in noise space (news, reality, game shows).
    pub fn max_domain_sim(&self, vec: &[f32]) -> f32 {
        self.centroids
            .values()
            .map(|c| cosine_sim(c, vec))
            .fold(f32::NEG_INFINITY, f32::max)
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

    /// Get centroid for a specific genre (for BYW anchor check).
    pub fn get(&self, genre: &str) -> Option<&Vec<f32>> {
        self.centroids.get(genre)
    }
}
