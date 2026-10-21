//! What the engine remembers about the caller's session: the taste it learned and the titles already started.
use crate::app::AppState;
use crate::det::HashSet;

pub struct SessionView {
    /// Genre affinities learned from feedback (used when the client sends no taste)
    pub taste: Vec<(String, f32)>,
    /// Titles the session has started (not offered again)
    pub seen: HashSet<u32>,
    /// Counts the feedback events; part of the page key, so new feedback gives a new page
    pub version: u64,
}

impl SessionView {
    pub fn of(state: &AppState, session_id: Option<&str>) -> Self {
        let state_of = session_id.filter(|s| !s.is_empty()).and_then(|sid| state.sessions.get(sid));
        match state_of {
            Some(s) => {
                let s = s.lock().unwrap_or_else(|e| e.into_inner());
                Self { taste: s.taste(), seen: s.seen_ids().iter().copied().collect(), version: s.version }
            }
            None => Self { taste: Vec::new(), seen: HashSet::default(), version: 0 },
        }
    }
}
