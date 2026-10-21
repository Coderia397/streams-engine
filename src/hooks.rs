//! The two short lines on the hero banner of a title (the badges at the bottom right). They are not a fixed pair of facts: each title gets a mix of kinds that
//! fits it and the moment: what is new about it, where it stands in the charts, who made it, what it is known for, what it feels like, a season or mood it
//! suits. Every line carries a translation key with its arguments and the English text as a fallback. The choice rotates with the session and every six hours.
use crate::catalog::{Catalog, CatalogItem};
use crate::page::entry_noise;
use serde_json::{json, Value};

pub struct HookEnv<'a> {
    pub catalog: &'a Catalog,
    pub vibe: &'a crate::vibe::VibeIndex,
    pub taxonomy: &'a crate::taxonomy::TaxonomyStore,
    pub profiles: &'a crate::profiles::ProfileStore,
    pub fresh: &'a crate::fresh::FreshStore,
    pub today_md: (u8, u8),
    pub today_days: i64,
    pub late_night: bool,
    pub session: &'a str,
    pub slot: usize,
    /// The banner is for a Kids profile: charts are the Kids charts and the lines are the ones that suit children
    pub kids: bool,
}

/// A line the banner can show: its group (two lines on one banner come from two groups), a weight and the value sent to the client.
struct Cand {
    group: u8,
    weight: f32,
    value: Value,
}

const NEWS: u8 = 0; // what is new, where it stands
const CREDIT: u8 = 1; // who made it
const FAME: u8 = 2; // what it is known for
const FEEL: u8 = 3; // how it feels, what mood it suits
const SEASON: u8 = 4; // the time of year

fn line(kind: &str, icon: &str, key: &str, args: Value, text: String) -> Value {
    json!({ "kind": kind, "icon": icon, "key": key, "args": args, "text": text })
}

/// "A, B, C" becomes "A and B" when that is short, "A" otherwise: a banner line is read at a glance.
fn short_names(names: &str) -> String {
    let parts: Vec<&str> = names.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
    match parts.as_slice() {
        [] => String::new(),
        [one] => (*one).to_string(),
        [a, b, ..] if a.chars().count() + b.chars().count() <= 30 => format!("{a} and {b}"),
        [a, ..] => (*a).to_string(),
    }
}

fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// Rank of a title among the chart titles of its own kind (1 to 10), when it is in the Top 10 of its kind.
fn chart_rank(env: &HookEnv, item: &CatalogItem) -> Option<usize> {
    let mut n = 0;
    for k in env.catalog.chart_keys.iter().take(80) {
        let Some(c) = env.catalog.get(k) else { continue };
        if c.media_type != item.media_type { continue; }
        n += 1;
        if c.id == item.id { return (n <= 10).then_some(n); }
        if n >= 10 { return None; }
    }
    None
}

/// Rank (1 to 10) of a title in the Kids chart of its own kind: the chart titles made for children (the same ones the Kids Top 10 row holds).
fn kids_chart_rank(env: &HookEnv, item: &CatalogItem) -> Option<usize> {
    let mut n = 0;
    for k in env.catalog.chart_keys.iter().take(400) {
        let Some(c) = env.catalog.get(k) else { continue };
        if c.media_type != item.media_type || !crate::page::kid_interest(env.profiles, c) || c.cert > 2 { continue; }
        n += 1;
        if c.id == item.id { return (n <= 10).then_some(n); }
        if n >= 10 { return None; }
    }
    None
}

/// How many well known titles (rated 7 or more by a thousand voters) each director or creator has in the catalog, by lower case name. A credit line is shown only for people with a real body of work.
fn notable_credits(catalog: &Catalog) -> &'static crate::det::HashMap<String, u16> {
    static MAP: std::sync::OnceLock<crate::det::HashMap<String, u16>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let mut m: crate::det::HashMap<String, u16> = crate::det::HashMap::default();
        for it in catalog.values() {
            if it.vote_average < 7.0 || it.vote_count < 1000 { continue; }
            if let Some(d) = it.director.as_deref() {
                for name in d.split(',').map(|x| x.trim().to_lowercase()).filter(|x| !x.is_empty()).take(2) {
                    *m.entry(name).or_insert(0) += 1;
                }
            }
        }
        m
    })
}

fn candidates(item: &CatalogItem, env: &HookEnv) -> Vec<Cand> {
    let is_tv = item.media_type == "tv";
    let mut out: Vec<Cand> = Vec::new();
    // news and charts
    let release = env.fresh.get(&item.media_type, item.tmdb_id).map(|f| f.release_date.as_str()).and_then(crate::fresh::parse_days);
    if release.is_some_and(|d| (0..=crate::contract::NEW_DAYS).contains(&(env.today_days - d))) {
        out.push(Cand { group: NEWS, weight: 3.0, value: line("new", "megaphone", "hook.recently_added", json!({}), "Recently added".into()) });
    }
    if env.kids {
        if let Some(rank) = kids_chart_rank(env, item) {
            let (key, text) = if is_tv { ("hook.kids_top10_series", format!("No. {rank} in Kids' Series")) } else { ("hook.kids_top10_films", format!("No. {rank} in Kids' Films")) };
            out.push(Cand { group: NEWS, weight: 4.0, value: line("top10", "fire", key, json!({ "rank": rank }), text) });
        }
    } else if let Some(rank) = chart_rank(env, item) {
        let (key, text) = if is_tv { ("hook.top10_series", format!("#{rank} in Series today")) } else { ("hook.top10_films", format!("#{rank} in Films today")) };
        out.push(Cand { group: NEWS, weight: 3.5, value: line("top10", "fire", key, json!({ "rank": rank }), text) });
    }
    // credits
    // only for someone with a body of work (at least three well known titles), and the more they have made the likelier the line
    let known = item.director.as_deref().map(|d| {
        let m = notable_credits(env.catalog);
        d.split(',').map(|x| x.trim().to_lowercase()).filter(|x| !x.is_empty()).take(2).map(|n| *m.get(&n).unwrap_or(&0)).max().unwrap_or(0)
    }).unwrap_or(0);
    if let (true, Some(name)) = (known >= 3, item.director.as_deref().filter(|d| !d.trim().is_empty()).map(short_names)) {
        let name = name.as_str();
        let (key, text) = if is_tv { ("hook.created_by", format!("Created by {name}")) } else { ("hook.directed_by", format!("Directed by {name}")) };
        out.push(Cand { group: CREDIT, weight: if known >= 8 { 2.6 } else { 1.8 }, value: line("credit", "clapperboard", key, json!({ "name": name }), text) });
    }
    // what it is known for
    for (fam, name) in [("emmy", "Emmy"), ("oscar", "Oscar"), ("bafta", "BAFTA"), ("globe", "Golden Globe")] {
        let won = item.keywords.iter().any(|k| k == &format!("award:{fam}:won"));
        let nominated = item.keywords.iter().any(|k| k == &format!("award:{fam}:nominated"));
        if won {
            out.push(Cand { group: FAME, weight: 3.5, value: line("award", "trophy", &format!("hook.{fam}_winner"), json!({}), format!("{name} winner")) });
        } else if nominated {
            out.push(Cand { group: FAME, weight: 2.8, value: line("award", "medal", &format!("hook.{fam}_nominee"), json!({}), format!("{name} nominee")) });
        }
    }
    if item.vote_average >= 8.0 && item.vote_count >= 1000 {
        out.push(Cand { group: FAME, weight: 2.0, value: line("acclaimed", "trophy", "hook.acclaimed", json!({}), "Critically acclaimed".into()) });
    }
    if item.popularity >= 60.0 && item.vote_average >= 7.4 {
        out.push(Cand { group: FAME, weight: 1.5, value: line("favourite", "heart", "hook.fan_favourite", json!({}), "Fan favourite".into()) });
    }
    if crate::personal::is_gem(item) {
        out.push(Cand { group: FAME, weight: 2.5, value: line("gem", "sparkle", "hook.hidden_gem", json!({}), "A hidden gem".into()) });
    }
    if item.release_year.is_some_and(|y| y <= 1995) && item.vote_average >= 7.5 {
        out.push(Cand { group: FAME, weight: 2.0, value: line("classic", "film-reel", "hook.timeless_classic", json!({}), "A timeless classic".into()) });
    }
    if !is_tv && item.runtime > 0 && item.runtime <= 95 {
        out.push(Cand { group: FAME, weight: 1.0, value: line("short", "clock", "hook.quick_watch", json!({ "min": item.runtime }), format!("A quick watch: {} min", item.runtime)) });
    }
    if !is_tv && item.runtime >= 150 {
        out.push(Cand { group: FAME, weight: 1.0, value: line("epic", "mountains", "hook.epic", json!({ "h": item.runtime / 60, "m": item.runtime % 60 }), format!("An epic: {}h {}m", item.runtime / 60, item.runtime % 60)) });
    }
    if item.keywords.iter().any(|k| k.contains("true story") || k.contains("biography")) {
        out.push(Cand { group: FAME, weight: 1.5, value: line("true_story", "book-open", "hook.true_story", json!({}), "Based on a true story".into()) });
    } else if item.keywords.iter().any(|k| k.contains("based on novel") || k.contains("based on book")) {
        out.push(Cand { group: FAME, weight: 1.5, value: line("book", "book-open", "hook.from_a_book", json!({}), "Based on a book".into()) });
    }
    // more things a title can be known for
    if let Some(y) = item.release_year.filter(|y| (1980..=2009).contains(y)) {
        if item.vote_average >= 7.2 && item.vote_count >= 800 {
            let label = format!("{}{}0s", if y >= 2000 { "20" } else { "19" }, (y % 100) / 10);
            out.push(Cand { group: FAME, weight: 1.6, value: line("decade", "film-reel", "hook.decade_favourite", json!({ "decade": label }), format!("A {label} favourite")) });
        }
    }
    if item.original_language != "en" && item.popularity >= 40.0 && item.vote_average >= 7.2 {
        out.push(Cand { group: FAME, weight: 1.8, value: line("global", "globe", "hook.global_hit", json!({}), "A global hit".into()) });
    }
    if item.genres.iter().any(|g| g == "Family") && item.vote_average >= 7.4 && item.vote_count >= 500 {
        out.push(Cand { group: FAME, weight: 1.6, value: line("family_fav", "users-three", "hook.family_favourite", json!({}), "A family favourite".into()) });
    }
    if env.kids {
        if item.cert <= 1 && item.genres.iter().any(|g| g == "Animation" || g == "Family") {
            out.push(Cand { group: FEEL, weight: 2.0, value: line("little", "baby", "hook.little_ones", json!({}), "Great for little ones".into()) });
        } else if item.cert == 2 {
            out.push(Cand { group: FEEL, weight: 1.6, value: line("big_kids", "rocket", "hook.big_kids", json!({}), "Made for big kids".into()) });
        }
    }
    // how it feels
    let tone = env.profiles.get(item.id).and_then(|p| p.tone.iter().find_map(|w| crate::profiles::tone_label(w)));
    let genre = item.genres.first().map(|g| lower(g));
    if let (Some(t), Some(g)) = (tone, genre.as_deref()) {
        let (tl, kind) = (lower(t), if entry_noise(env.session, item.id as usize + env.slot) < 0.5 { 0 } else { 1 });
        let (key, text) = if kind == 0 { ("hook.craving", format!("Craving {tl} {g}?")) } else { ("hook.in_the_mood", format!("In the mood for {tl} {g}?")) };
        out.push(Cand { group: FEEL, weight: 3.0, value: line("feel", "popcorn", key, json!({ "tone": tl, "genre": g }), text) });
    }
    // a mood it suits (a late night mood first on a night page)
    let mut best_mood: Option<&crate::vibe::Mood> = None;
    for m in &env.vibe.moods {
        let fits = m.groups.iter().take(6).any(|(g, _)| env.vibe.members.get(g).is_some_and(|v| v.contains(&item.id)));
        if fits && (best_mood.is_none() || (env.late_night && m.late && !best_mood.is_some_and(|b| b.late)) || entry_noise(env.session, 5_000 + m.id.len() + item.id as usize) > 0.7) {
            best_mood = Some(m);
        }
    }
    if let Some(m) = best_mood {
        if env.late_night && m.late {
            out.push(Cand { group: FEEL, weight: 4.0, value: line("late", "moon", "hook.late_night", json!({}), "Made for late nights".into()) });
        } else {
            let phrase = lower(&m.title);
            out.push(Cand { group: FEEL, weight: 2.5, value: line("mood", "smiley", "hook.in_the_mood_for", json!({ "mood": phrase }), format!("In the mood for {phrase}?")) });
        }
    }
    // a season it suits
    for ev in crate::personal::events_in_season(env.taxonomy, env.vibe, env.today_md) {
        if env.vibe.seasons[ev].iter().any(|s| s.items.contains(&item.id)) {
            let label = crate::personal::prettify(ev);
            out.push(Cand { group: SEASON, weight: 4.0, value: line("season", "leaf", "hook.for_season", json!({ "event": ev, "label": label }), format!("Perfect for {label}")) });
            break;
        }
    }
    // a family title
    if item.genres.iter().any(|g| g == "Family") && item.cert <= 2 && item.cert > 0 {
        out.push(Cand { group: FEEL, weight: 1.5, value: line("family", "users-three", "hook.family", json!({}), "Watch with the whole family".into()) });
    }
    out
}

/// Up to two lines for the hero banner of this title, from different groups, drawn by weight with the session and the six hour slot as the seed.
pub fn hero_hooks(item: &CatalogItem, env: &HookEnv) -> Vec<Value> {
    let mut cands = candidates(item, env);
    if cands.is_empty() { return Vec::new(); }
    let seed = format!("{}|{}", env.session, env.slot);
    let mut keyed: Vec<(f32, u8, Value)> = cands
        .drain(..)
        .enumerate()
        .map(|(i, c)| (c.weight * (0.4 + entry_noise(&seed, item.id as usize * 37 + i)), c.group, c.value))
        .collect();
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<Value> = Vec::new();
    let mut first_group = None;
    // a line is shown only when it is worth a place: the first needs a score of 1.3, the second 1.9; a title with nothing to say gets no badge (as on Netflix)
    for (score, g, v) in keyed {
        let need = if out.is_empty() { 1.3 } else { 1.9 };
        if score < need { break; }
        if first_group.is_some_and(|f| f == g) { continue; }
        first_group.get_or_insert(g);
        out.push(v);
        if out.len() == 2 { break; }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::short_names;

    #[test]
    fn credit_lines_stay_short() {
        assert_eq!(short_names("Dick Wolf"), "Dick Wolf");
        assert_eq!(short_names("Derek Haas, Michael Brandt"), "Derek Haas and Michael Brandt");
        assert_eq!(short_names("Aaron Horvath, Michael Jelenic, Someone Else"), "Aaron Horvath and Michael Jelenic");
        assert_eq!(short_names("Christopher Miller Smithson, Phil Lord Anderson"), "Christopher Miller Smithson");
    }
}
