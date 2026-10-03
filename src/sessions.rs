use moka::sync::Cache;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::catalog::CatalogItem;

const MAX_SEEN: usize = 500;
const DECAY: f32 = 0.95;
const MAX_TASTE_GENRES: usize = 5;

/// Implicit signal reported by the client for one title.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackEvent {
    Click,
    Start,
    Complete,
    Like,
    Skip,
}

impl FeedbackEvent {
    fn weight(self) -> f32 {
        match self {
            Self::Click => 0.3,
            Self::Start => 0.6,
            Self::Complete => 1.0,
            Self::Like => 1.5,
            Self::Skip => -0.5,
        }
    }

    /// Events that mean the person has engaged with the title, so it should
    /// not be pushed again as a fresh recommendation.
    fn marks_seen(self) -> bool {
        matches!(self, Self::Start | Self::Complete | Self::Like)
    }
}

#[derive(Default)]
pub struct SessionState {
    /// Genre name to affinity, decayed on every event.
    taste: HashMap<String, f32>,
    seen: HashSet<u32>,
    seen_order: VecDeque<u32>,
    /// Bumped on every change so cached feeds can be keyed on it.
    pub version: u64,
}

impl SessionState {
    pub fn record(&mut self, item: &CatalogItem, event: FeedbackEvent) {
        for v in self.taste.values_mut() {
            *v *= DECAY;
        }
        self.taste.retain(|_, v| v.abs() > 0.01);

        let share = event.weight() / item.genres.len().max(1) as f32;
        for g in &item.genres {
            *self.taste.entry(g.clone()).or_insert(0.0) += share;
        }

        if event.marks_seen() && self.seen.insert(item.id) {
            self.seen_order.push_back(item.id);
            if self.seen_order.len() > MAX_SEEN {
                if let Some(old) = self.seen_order.pop_front() {
                    self.seen.remove(&old);
                }
            }
        }
        self.version += 1;
    }

    pub fn has_seen(&self, id: u32) -> bool {
        self.seen.contains(&id)
    }

    pub fn seen_ids(&self) -> &HashSet<u32> {
        &self.seen
    }

    /// Top positive genre affinities, normalised to sum to 1.
    pub fn taste(&self) -> Vec<(String, f32)> {
        let mut t: Vec<(String, f32)> = self
            .taste
            .iter()
            .filter(|(_, &w)| w > 0.0)
            .map(|(g, &w)| (g.clone(), w))
            .collect();
        t.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        t.truncate(MAX_TASTE_GENRES);
        let total: f32 = t.iter().map(|(_, w)| w).sum();
        if total > 0.0 {
            t.iter_mut().for_each(|(_, w)| *w /= total);
        }
        t
    }
}

/// In-memory per-session store. Sessions expire after 24h without activity.
#[derive(Clone)]
pub struct SessionStore {
    inner: Cache<String, Arc<Mutex<SessionState>>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            inner: Cache::builder()
                .max_capacity(100_000)
                .time_to_idle(Duration::from_secs(24 * 3600))
                .build(),
        }
    }

    pub fn get_or_create(&self, session_id: &str) -> Arc<Mutex<SessionState>> {
        self.inner
            .get_with(session_id.to_string(), || Arc::new(Mutex::new(SessionState::default())))
    }

    pub fn get(&self, session_id: &str) -> Option<Arc<Mutex<SessionState>>> {
        self.inner.get(session_id)
    }
}
