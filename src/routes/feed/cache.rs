//! The key a built page is cached under: every input that changes the page is in it, nothing else.
use super::query::FeedRequest;
use super::session::SessionView;

/// The choice of rows rotates every six hours for the same session, so a returning visitor does not meet the same page all week.
pub fn slot() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / (6 * 3600))
}

/// The session string that seeds the page's exploration noise: stable for one session inside one slot.
pub fn noise_session(req: &FeedRequest, slot: u64) -> String {
    format!("{}|{slot}", req.session_id.as_deref().unwrap_or(""))
}

pub fn page_key(req: &FeedRequest, session: &SessionView, slot: u64) -> String {
    let sid = req.session_id.as_deref().unwrap_or("");
    let mut key = taste_key(req.surface, &req.genre.as_deref().unwrap_or("").to_lowercase(), req.offset + req.per_page, &req.taste);
    // Entry driven pages explore per session, so the session is part of the key
    key = format!("{key}|t:{slot}|s:{sid}");
    if session.version > 0 {
        key = format!("{key}|{sid}|v{}", session.version);
    }
    key = format!("{key}|n:{}", req.late_night as u8);
    if let Some(c) = req.collection.as_deref() {
        key = format!("{key}|c:{c}");
    }
    if !req.signals.is_empty() {
        key = format!("{key}|g:{}", req.signals.fingerprint());
    }
    // pages differ per maturity
    format!("{key}|m:{}|l:{}", req.maturity.map_or("-", |m| m.as_str()), req.language.as_deref().unwrap_or(""))
}

/// Normalise taste weights to 1 decimal, sort by genre id, cap at 5: "action:0.80" and "action:0.8001" share one key.
fn taste_key(surface: &str, genre: &str, rows: usize, taste: &str) -> String {
    let mut pairs: Vec<(String, String)> = taste
        .split(',')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, ':');
            let genre = parts.next()?.trim().to_string();
            let weight: f32 = parts.next().and_then(|w| w.trim().parse().ok()).unwrap_or(1.0);
            let rounded = format!("{:.1}", weight);
            Some((genre, rounded))
        })
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs.truncate(5);
    let taste_bucket = pairs.iter().map(|(g, w)| format!("{g}:{w}")).collect::<Vec<_>>().join(",");
    format!("{surface}|{genre}|{rows}|{taste_bucket}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_taste_weights_share_a_key() {
        assert_eq!(taste_key("both", "", 12, "action:0.80,drama:0.3"), taste_key("both", "", 12, "drama:0.31,action:0.8001"));
        assert_ne!(taste_key("both", "", 12, "action:0.8"), taste_key("both", "", 12, "action:0.5"));
    }
}
