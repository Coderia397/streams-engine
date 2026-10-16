use crate::vectors::cosine_sim;

/// Mood adjective vocabulary — platform-native, accessible.
/// These match the style of Netflix, Disney+, Apple TV+ row labels.
pub const MOOD_ADJECTIVES: &[&str] = &[
    "Dark",       "Epic",         "Mind-Bending",  "Feel-Good",
    "Gripping",   "Thrilling",    "Romantic",      "Hilarious",
    "Uplifting",  "Chilling",     "Suspenseful",   "Heartfelt",
    "Gritty",     "Cerebral",     "Cozy",          "Haunting",
    "Intense",    "Charming",     "Irreverent",     "Steamy",
];

/// Pre-computed mood centroids (2560d unit vectors).
/// In production these are loaded from `taxonomy_embeddings.bin` — the same
/// embedding pass that covers all taxonomy entries includes mood-only queries.
/// For the mood adjective projection we use the vibe_query of single-mood
/// taxonomy entries (entries with exactly one mood pill).
///
/// At startup: `MoodVault::from_taxonomy_store(store)`.
/// Fallback at index 0 = "Gripping" (safe neutral default).
pub struct MoodVault {
    /// parallel to MOOD_ADJECTIVES: mood_centroids[i] is the embedding for MOOD_ADJECTIVES[i]
    pub mood_centroids: Vec<Vec<f32>>,
}

impl MoodVault {
    /// Build from a flat list of mood embeddings (indexed by MOOD_ADJECTIVES order).
    pub fn new(mood_centroids: Vec<Vec<f32>>) -> Self {
        Self { mood_centroids }
    }

    /// Project a cluster centroid to the closest mood adjective.
    /// Returns the adjective string.
    pub fn closest(&self, centroid: &[f32]) -> &'static str {
        if self.mood_centroids.is_empty() {
            return "Gripping";
        }
        let best_idx = self
            .mood_centroids
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                cosine_sim(a, centroid)
                    .partial_cmp(&cosine_sim(b, centroid))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        MOOD_ADJECTIVES[best_idx.min(MOOD_ADJECTIVES.len() - 1)]
    }
}

/// Load `mood_embeddings.bin`: u32 LE count, then count x 2560 f32 LE, in MOOD_ADJECTIVES order.
pub fn load_mood_embeddings(path: &str) -> anyhow::Result<Vec<Vec<f32>>> {
    use crate::vectors::{l2_normalize, DIMS};
    let bytes = std::fs::read(path)?;
    let n = u32::from_le_bytes(bytes.get(0..4).ok_or_else(|| anyhow::anyhow!("file too short"))?.try_into()?) as usize;
    if n != MOOD_ADJECTIVES.len() || bytes.len() != 4 + n * DIMS * 4 {
        anyhow::bail!("mood_embeddings.bin has {n} entries, expected {}", MOOD_ADJECTIVES.len());
    }
    Ok((0..n)
        .map(|i| {
            let off = 4 + i * DIMS * 4;
            let mut v: Vec<f32> = crate::vectors::le_f32s(&bytes[off..off + DIMS * 4]);
            l2_normalize(&mut v);
            v
        })
        .collect())
}
