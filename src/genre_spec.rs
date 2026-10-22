//! Genre pages: turn what a client sends for `genre` into a rule over catalog items.
//!
//! The frontend sends display names from its pickers ("Comedies", "Sci-Fi", "Black Stories", "Stand-up & Chat Shows"),
//! sometimes TMDB names ("Comedy"), and the hero feed sends numeric ids (28, 10003, 10765). TMDB genres alone cannot
//! express collections such as British, European, Cult or Shorts, so those use catalog_v2 metadata: origin countries,
//! keywords, runtime, original language, TV type, release year and vote counts.
//!
//! Movies and TV use different TMDB genre names ("Action" and "Action & Adventure", "Science Fiction" and
//! "Sci-Fi & Fantasy"); every rule matches both, so one name works on the films page, the series page and home.

use crate::catalog::CatalogItem;

#[derive(Clone, Debug, PartialEq)]
pub enum GenreSpec {
    /// Matches everything (a picker that is not a filter, for example Moods)
    Any,
    /// A collection the data cannot express (about 30 titles for Astrology)
    Unsupported,
    /// Titles that won or were nominated for an award family (`emmy`, `oscar`, `bafta`, `globe`), from the awards file
    Awards(&'static str),
    /// Any of these TMDB genre names, movie and TV naming
    Tmdb(&'static [&'static str]),
    Thriller,
    Anime,
    BlackStories,
    BookAdaptations,
    British,
    Classics,
    Cult,
    European,
    Halloween,
    Horror,
    Independent,
    International,
    Lgbtq,
    Musical,
    Reality,
    Romance,
    ScienceNature,
    Shorts,
    Sports,
    StandUp,
    ChatShows,
    Teen,
    /// The page of a calendar event (`events_v1.json`: Christmas, Valentine's Day ...), offered in its window; a title belongs by the event's keywords
    Event(String),
    /// Hollywood films (US, widely rated) and US series
    Us,
    /// A page of the Kids menus (genre_pages::KIDS_*): the rule is read from `kids_rule` by the page's lower case name
    Kids(&'static str),
    /// Not in the table: exact TMDB genre name, case insensitive (the earlier behaviour)
    Name(String),
}

/// The rules of the Kids pages that have no genre of their own: (page name, any of these keywords, any of these genres). Netflix's own per title tags count as keywords
/// (`nf:<tag>`, note 102). A children's profile's age limit keeps every page to children's titles; the rule says what the page is about.
const KIDS_RULES: &[(&str, &[&str], &[&str])] = &[
    ("animal time", &["animal", "animals", "talking animal", "anthropomorphic animal", "dog", "cat", "horse", "pets", "wildlife", "nf:animals", "nf:animal tales", "zoo", "farm"], &[]),
    ("animals", &["animal", "animals", "talking animal", "anthropomorphic animal", "dog", "cat", "horse", "pets", "wildlife", "nf:animals", "nf:animal tales", "zoo", "farm"], &[]),
    ("cars, trucks & trains", &["car", "cars", "truck", "trucks", "train", "trains", "racing", "race car", "monster truck", "vehicle", "vehicles", "construction vehicle", "nf:cars & trucks"], &[]),
    ("dinosaurs", &["dinosaur", "dinosaurs", "prehistoric", "nf:dinosaurs"], &[]),
    ("feel-good", &["nf:feel-good", "nf:heartfelt", "nf:charming", "nf:fun", "feelgood", "feel-good", "heartwarming", "cheerful", "lighthearted"], &[]),
    ("for little ones", &["preschool", "pre-school", "toddler", "nursery", "educational", "nf:preschool", "nf:educational", "learning", "alphabet", "counting"], &[]),
    ("friends", &["friendship", "friends", "best friends", "buddies", "nf:friendship", "nf:besties", "friendship group"], &[]),
    ("funny", &["nf:goofy", "nf:witty", "nf:fun", "nf:irreverent", "nf:absurd", "slapstick comedy", "silly"], &["Comedy"]),
    ("girls take the lead", &["female protagonist", "girl", "girls", "heroine", "strong female lead", "girl power", "nf:girl power", "nf:female protagonist", "princess", "sisters"], &[]),
    ("science & arts", &["science", "educational", "art", "arts", "crafts", "mathematics", "stem", "experiment", "inventions", "music education", "nf:educational", "nf:science"], &[]),
    ("singing & dancing", &["musical", "singing", "song", "songs", "dance", "dancing", "sing-along", "nf:musical", "nf:music"], &["Music"]),
    ("spooky stuff", &["halloween", "ghost", "ghosts", "monster", "monsters", "witch", "spooky", "haunted house", "haunting", "vampire", "nf:spooky", "nf:halloween"], &[]),
    ("cool special effects", &["superhero", "robot", "giant robot", "giant monster", "visual effects", "space", "spaceship", "alien", "superpower", "nf:superhero"], &["Science Fiction", "Sci-Fi & Fantasy"]),
    ("princess tales", &["princess", "prince", "fairy tale", "fairytale", "royalty", "nf:princess", "nf:fairy tale"], &[]),
];

fn kids_rule(name: &str) -> Option<&'static (&'static str, &'static [&'static str], &'static [&'static str])> {
    KIDS_RULES.iter().find(|(n, _, _)| *n == name)
}

const EUROPE: &[&str] = &[
    "FR", "DE", "IT", "ES", "SE", "DK", "NO", "FI", "PL", "NL", "BE", "AT", "CH", "IE", "PT", "GR", "CZ", "HU", "RO", "BG",
    "HR", "RS", "SK", "SI", "LT", "LV", "EE", "IS", "LU", "UA", "AL", "BA", "MK", "ME", "CY", "MT",
];

fn has_genre(it: &CatalogItem, names: &[&str]) -> bool {
    it.genres.iter().any(|g| names.iter().any(|n| g.eq_ignore_ascii_case(n)))
}
fn kw_is(it: &CatalogItem, words: &[&str]) -> bool {
    it.keywords.iter().any(|k| words.iter().any(|w| k == w))
}
/// Membership judged for this page category (`cat:<slug>`); None when the judged file was not loaded, so the page falls back to its keyword rule.
fn judged(it: &CatalogItem, slug: &str) -> Option<bool> {
    crate::catalog::CATEGORIES_LOADED.load(std::sync::atomic::Ordering::Relaxed).then(|| it.keywords.iter().any(|k| k.strip_prefix("cat:") == Some(slug)))
}
fn kw_has(it: &CatalogItem, parts: &[&str]) -> bool {
    it.keywords.iter().any(|k| parts.iter().any(|p| k.contains(p)))
}
fn from_country(it: &CatalogItem, codes: &[&str]) -> bool {
    it.countries.iter().any(|c| codes.contains(&c.as_str()))
}

impl GenreSpec {
    /// The calendar event this page is the page of (the collection banner of Home leads here), if it is one.
    pub fn event_id(&self) -> Option<&str> {
        match self { GenreSpec::Halloween => Some("halloween"), GenreSpec::Lgbtq => Some("pride"), GenreSpec::Event(e) => Some(e.as_str()), _ => None }
    }

    /// Whether the page is offered today: an event page in its window only, every other page always.
    pub fn in_season(&self) -> bool {
        match self { GenreSpec::Event(e) => crate::events::in_season(e), _ => true }
    }
}

impl GenreSpec {
    /// Whether children's titles (TMDB Kids or Family) belong on a page of this genre. Only pages about animation, anime,
    /// kids, family or teens keep them; a page for any other genre (Action, Thriller, Pride) is for adults and teens.
    pub fn keeps_children(&self) -> bool {
        match self {
            GenreSpec::Any | GenreSpec::Anime | GenreSpec::Teen | GenreSpec::Unsupported | GenreSpec::Event(_) => true,
            GenreSpec::Tmdb(names) => names.iter().any(|n| matches!(n.to_ascii_lowercase().as_str(), "kids" | "family" | "animation")),
            GenreSpec::Name(n) => matches!(n.to_ascii_lowercase().as_str(), "kids" | "family" | "animation" | "kids & family"),
            _ => false,
        }
    }

    /// Whether the page's members come from the judged lists (`categories_v1.json` is loaded and this page is one of its categories).
    pub fn is_judged(&self) -> bool {
        use GenreSpec::*;
        crate::catalog::CATEGORIES_LOADED.load(std::sync::atomic::Ordering::Relaxed) && matches!(self, BlackStories | Cult | Halloween | Independent | Lgbtq | ScienceNature | Sports | StandUp | ChatShows)
    }

    pub fn matches(&self, it: &CatalogItem) -> bool {
        use GenreSpec::*;
        // the pages whose membership was judged on Colab use the judged list alone; the keyword rules below only propose candidates
        let slug = match self {
            BlackStories => Some("black_stories"), Cult => Some("cult"), Halloween => Some("halloween"), Independent => Some("independent"), Lgbtq => Some("lgbtq"),
            ScienceNature => Some("science_nature"), Sports => Some("sport"), StandUp => Some("stand_up"), ChatShows => Some("chat_shows"), _ => None,
        };
        if let Some(slug) = slug {
            if let Some(j) = judged(it, slug) {
                return j || (matches!(self, ChatShows) && judged(it, "stand_up") == Some(true));
            }
        }
        match self {
            Any => true,
            Unsupported => false,
            Awards(family) => it.keywords.iter().any(|k| k.strip_prefix("award:").is_some_and(|r| r.split(':').next() == Some(*family))),
            Tmdb(names) => has_genre(it, names),
            // For Little Kids (ages about 5 to 8): U rated animation or family titles that are not preschool ones (those are For Little Ones)
            // For Little Ones: preschool television: TV-Y, or animation rated TV-G or U that runs 12 minutes or less, or the preschool words
            Kids("for little ones") => it.cert_us == 6
                || ((it.cert_us == 8 || it.cert == 1) && (1..=12).contains(&it.runtime) && has_genre(it, &["Animation"]))
                || kids_rule("for little ones").is_some_and(|(_, kws, _)| kw_is(it, kws)),
            // For Little Kids: the preschool titles (41% of Netflix's page are also on For Little Ones) and short U rated animation or family films and specials
            Kids("little kids") => kids_rule("for little ones").is_some_and(|(_, kws, _)| kw_is(it, kws)) || (it.cert == 1 && has_genre(it, &["Animation", "Family", "Kids"]) && it.runtime > 0 && it.runtime <= 75),
            Kids(name) => kids_rule(name).is_some_and(|(_, kws, genres)| kw_is(it, kws) || (!genres.is_empty() && has_genre(it, genres))),
            Thriller => has_genre(it, &["Thriller"]) || (it.media_type == "tv" && has_genre(it, &["Mystery"])),
            Anime => it.is_anime,
            BlackStories => kw_is(
                it,
                &["black cinema", "african american", "blaxploitation", "black history", "civil rights", "civil rights movement",
                  "slavery", "racism", "hip-hop"],
            ),
            BookAdaptations => kw_is(it, &["based on novel or book", "based on young adult novel"]),
            British => from_country(it, &["GB"]),
            Classics => it.media_type == "movie" && it.release_year.is_some_and(|y| y <= 1979) && it.vote_average >= 7.0,
            Cult => kw_is(
                it,
                &["cult film", "cult classic", "cult", "midnight movie", "cult tv", "cult favorite", "grindhouse", "exploitation",
                  "b movie", "so bad it's good", "trash cinema", "splatter", "campy"],
            ),
            European => from_country(it, EUROPE) && !from_country(it, &["US", "GB"]),
            Halloween => kw_has(it, &["halloween"]),
            // TMDB has no Horror or Romance genre for TV, so series are found through keywords
            Horror => has_genre(it, &["Horror"])
                || (it.media_type == "tv"
                    && kw_has(it, &["horror", "slasher", "zombie", "haunted", "ghost", "vampire", "demon"])),
            Romance => has_genre(it, &["Romance"]) || (it.media_type == "tv" && kw_has(it, &["romance", "romantic"])),
            Independent => kw_is(
                it,
                &["independent film", "indie film", "indie", "independent cinema", "low budget", "micro budget", "no budget",
                  "arthouse", "art house", "experimental film", "experimental cinema"],
            ),
            International => !it.original_language.is_empty() && it.original_language != "en" && it.original_language != "xx",
            Lgbtq => it.keywords.iter().any(|k| {
                k == "gay" || k.starts_with("gay ") || k == "coming out"
                    || ["lgbt", "homosexual", "lesbian", "queer", "transgender", "bisexual", "drag queen", "drag race"]
                        .iter().any(|p| k.contains(p))
            }),
            Musical => has_genre(it, &["Music"]) || kw_has(it, &["musical"]),
            Reality => has_genre(it, &["Reality"]) || it.tv_type.as_deref() == Some("Reality"),
            Event(e) => crate::events::title_belongs(e, it),
            ScienceNature => has_genre(it, &["Documentary"])
                && kw_has(it, &["nature", "wildlife", "animal", "science", "space", "ocean", "planet", "universe", "dinosaur",
                                "environment", "biology", "physics", "climate"]),
            Shorts => it.media_type == "movie" && it.runtime > 0 && it.runtime <= 40,
            Sports => kw_has(
                it,
                &["sport", "football", "basketball", "boxing", "wrestling", "baseball", "olympic", "soccer", "tennis", "golf",
                  "cycling", "formula", "motorsport", "athlet", "marathon", "swimming", "hockey", "rugby", "cricket", "skiing",
                  "skateboard", "surf"],
            ),
            StandUp => kw_has(it, &["stand-up comedy", "stand-up comedian", "stand-up special", "stand up comedy"]),
            ChatShows => StandUp.matches(it) || it.tv_type.as_deref() == Some("Talk Show"),
            Teen => kw_has(it, &["high school", "teen", "coming of age"]),
            Us => if it.media_type == "tv" {
                from_country(it, &["US"])
            } else {
                from_country(it, &["US"]) && (it.vote_count >= 1000 || it.popularity >= 30.0)
            },
            Name(n) => it.genres.iter().any(|g| g.eq_ignore_ascii_case(n)),
        }
    }

    /// True for collections whose own content the hero feed would otherwise exclude (documentary, reality, talk).
    pub fn includes_nonscripted(&self) -> bool {
        use GenreSpec::*;
        match self {
            Reality | ChatShows | ScienceNature => true,
            Tmdb(n) => n.iter().any(|g| ["Documentary", "News", "Talk", "Soap", "Reality"].contains(g)),
            Name(n) => ["documentary", "news", "talk", "soap", "reality"].contains(&n.to_lowercase().as_str()),
            _ => false,
        }
    }
}

fn by_name(key: &str) -> Option<GenreSpec> {
    use GenreSpec::*;
    Some(match key {
        "action" => Tmdb(&["Action", "Action & Adventure"]),
        "adventure" => Tmdb(&["Adventure", "Action & Adventure"]),
        "animation" => Tmdb(&["Animation"]),
        "anime" => Anime,
        "astrology" => Unsupported,
        "emmys" => Awards("emmy"),
        "black stories" => BlackStories,
        "book adaptations" => BookAdaptations,
        "british" => British,
        "classics" => Classics,
        "comedies" | "comedy" => Tmdb(&["Comedy"]),
        "crime" => Tmdb(&["Crime"]),
        "cult" => Cult,
        "documentaries" | "documentary" | "documentary series" => Tmdb(&["Documentary"]),
        "dramas" | "drama" => Tmdb(&["Drama"]),
        "european" => European,
        "fantasy" => Tmdb(&["Fantasy", "Sci-Fi & Fantasy"]),
        "halloween" => Halloween,
        "history" => Tmdb(&["History"]),
        "hollywood" | "us" => Us,
        "horror" => Horror,
        "independent" => Independent,
        "international" => International,
        "kids & family" | "kids" | "family" => Tmdb(&["Family", "Kids"]),
        "lgbtq" | "pride" => Lgbtq,
        "moods" => Any,
        "music & musicals" | "music" | "musicals" => Musical,
        "mystery" | "mysteries" => Tmdb(&["Mystery"]),
        "news" => Tmdb(&["News"]),
        "reality" => Reality,
        "romance" => Romance,
        "sci-fi" | "science fiction" | "sci-fi & fantasy" => Tmdb(&["Science Fiction", "Sci-Fi & Fantasy"]),
        "science & nature" => ScienceNature,
        "shorts" => Shorts,
        "soap" => Tmdb(&["Soap"]),
        "sports" | "sport" => Sports,
        "stand-up comedy" => StandUp,
        "stand-up & chat shows" => ChatShows,
        "talk" => Tmdb(&["Talk"]),
        "teen" => Teen,
        "thriller" | "thrillers" => Thriller,
        "tv movie" => Tmdb(&["TV Movie"]),
        "war" | "war & politics" => Tmdb(&["War", "War & Politics"]),
        "western" => Western,
        "animated" => Tmdb(&["Animation"]),
        "watch with the family" => Tmdb(&["Family", "Kids"]),
        // the Kids menus' own pages
        "animal time" => Kids("animal time"),
        "animals" => Kids("animals"),
        "cars, trucks & trains" | "cars trucks & trains" => Kids("cars, trucks & trains"),
        "dinosaurs" => Kids("dinosaurs"),
        "feel-good" | "feel good" => Kids("feel-good"),
        "for little ones" => Kids("for little ones"),
        "for little kids" | "little kids" => Kids("little kids"),
        "friends" => Kids("friends"),
        "funny" => Kids("funny"),
        "girls take the lead" => Kids("girls take the lead"),
        "science & arts" => Kids("science & arts"),
        "singing & dancing" => Kids("singing & dancing"),
        "spooky stuff" => Kids("spooky stuff"),
        "cool special effects" => Kids("cool special effects"),
        "princess tales" => Kids("princess tales"),
        _ => return None,
    })
}
#[allow(non_upper_case_globals)]
const Western: GenreSpec = GenreSpec::Tmdb(&["Western"]);

/// Ids the frontend's pickers use (page_genres.rs and the TMDB genre ids). 10008 and 16 depend on the surface.
fn by_id(id: u32, surface: &str) -> Option<&'static str> {
    Some(match id {
        28 | 10759 => "action",
        12 => "adventure",
        16 => "anime",
        10002 => "astrology",
        10003 => "black stories",
        10004 => "book adaptations",
        10005 => "british",
        10007 => "moods",
        10009 => "classics",
        35 => "comedies",
        80 => "crime",
        10010 => "cult",
        99 => "documentaries",
        18 => "dramas",
        10018 => "emmys",
        10006 => "european",
        14 => "fantasy",
        10016 => "halloween",
        36 => "history",
        10008 => if surface == "tv" { "us" } else { "hollywood" },
        27 => "horror",
        10011 => "independent",
        10012 => "international",
        10751 | 10762 => "kids & family",
        10001 => "lgbtq",
        10402 => "music & musicals",
        9648 => "mystery",
        10763 => "news",
        10764 => "reality",
        10749 => "romance",
        878 | 10765 => "sci-fi",
        10019 => "science & nature",
        10013 => "shorts",
        10766 => "soap",
        10014 => "sports",
        10017 => "stand-up comedy",
        10020 => "stand-up & chat shows",
        10767 => "talk",
        10015 => "teen",
        53 => "thriller",
        10770 => "tv movie",
        10752 | 10768 => "war",
        37 => "western",
        _ => return None,
    })
}

/// Resolve a client value (name or numeric id) for a surface ("movie", "tv" or "both").
pub fn resolve(raw: &str, surface: &str) -> GenreSpec {
    let t = raw.trim();
    let key = if let Ok(id) = t.parse::<u32>() {
        match by_id(id, surface) {
            Some(k) => k.to_string(),
            None => {
                // the page of an event: its id (10101 up); Halloween and Pride are genre pages too and resolve above
                if let Some(ev) = crate::genre_pages::event_of_page(id).filter(|ev| crate::events::get(ev).is_some()) {
                    return GenreSpec::Event(ev.to_string());
                }
                return GenreSpec::Name(t.to_string());
            }
        }
    } else {
        t.to_lowercase().replace(" and ", " & ").split_whitespace().collect::<Vec<_>>().join(" ")
    };
    if let Some(spec) = by_name(&key) {
        return spec;
    }
    // an event by its id (`christmas`) or its label ("Christmas", "Valentine's Day")
    if let Some(e) = crate::events::get(&key).or_else(|| crate::events::by_label(&key)) {
        return GenreSpec::Event(e.id.clone());
    }
    GenreSpec::Name(t.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(mt: &str) -> CatalogItem {
        CatalogItem {
            id: 1, tmdb_id: 1, media_type: mt.to_string(), title: "t".into(), release_year: Some(2000), genres: vec![],
            overview: String::new(), tagline: None, director: None, poster_path: None, backdrop_path: None,
            vote_average: 7.5, popularity: 10.0, is_anime: false, vote_count: 0, adult: false,
            original_language: "en".into(), tv_type: None, countries: vec![], keywords: vec![], runtime: 0, networks: vec![], flags: 0, cert: 0, cert_gb: 0, cert_us: 0,
        }
    }

    #[test]
    fn names_and_ids_resolve_alike() {
        assert_eq!(resolve("Comedies", "movie"), resolve("Comedy", "tv"));
        assert_eq!(resolve("35", "movie"), resolve("comedies", "movie"));
        assert_eq!(resolve("Sci-Fi", "both"), resolve("Sci-Fi & Fantasy", "tv"));
        assert_eq!(resolve("Stand-up and Chat Shows", "tv"), GenreSpec::ChatShows);
        assert_eq!(resolve("10008", "tv"), GenreSpec::Us);
        assert_eq!(resolve("10008", "movie"), GenreSpec::Us);
        assert_eq!(resolve("Emmys", "tv"), GenreSpec::Awards("emmy"));
        let mut c = item("tv");
        assert!(!resolve("Emmys", "tv").matches(&c));
        c.keywords = vec!["award:emmy:nominated".into()];
        assert!(resolve("Emmys", "tv").matches(&c));
        assert_eq!(resolve("Noir", "movie"), GenreSpec::Name("Noir".into()));
    }

    #[test]
    fn genre_names_cross_media() {
        let mut m = item("movie"); m.genres = vec!["Science Fiction".into()];
        let mut t = item("tv"); t.genres = vec!["Sci-Fi & Fantasy".into()];
        let s = resolve("Sci-Fi", "both");
        assert!(s.matches(&m) && s.matches(&t));
        let mut mys = item("tv"); mys.genres = vec!["Mystery".into()];
        assert!(resolve("Thriller", "tv").matches(&mys));
        let mut mm = item("movie"); mm.genres = vec!["Mystery".into()];
        assert!(!resolve("Thriller", "movie").matches(&mm));
    }

    #[test]
    fn collections_use_metadata() {
        let mut a = item("movie"); a.countries = vec!["GB".into()]; assert!(resolve("British", "movie").matches(&a));
        let mut e = item("tv"); e.countries = vec!["FR".into()]; assert!(resolve("European", "tv").matches(&e));
        e.countries = vec!["FR".into(), "US".into()]; assert!(!resolve("European", "tv").matches(&e));
        let mut h = item("tv"); h.keywords = vec!["zombie".into()]; assert!(resolve("Horror", "tv").matches(&h));
        let mut hm = item("movie"); hm.keywords = vec!["zombie".into()]; assert!(!resolve("Horror", "movie").matches(&hm), "movies use the genre");
        let mut r = item("tv"); r.keywords = vec!["romance".into()]; assert!(resolve("Romance", "tv").matches(&r));
        let mut s = item("movie"); s.runtime = 25; assert!(resolve("Shorts", "movie").matches(&s));
        s.runtime = 120; assert!(!resolve("Shorts", "movie").matches(&s));
        let mut l = item("movie"); l.keywords = vec!["gay theme".into()]; assert!(resolve("LGBTQ", "movie").matches(&l));
        l.keywords = vec!["dragon".into()]; assert!(!resolve("Pride", "movie").matches(&l));
        let mut c = item("movie"); c.keywords = vec!["occult".into()]; assert!(!resolve("Cult", "movie").matches(&c));
        c.keywords = vec!["cult".into()]; assert!(resolve("Cult", "movie").matches(&c));
        let mut cl = item("movie"); cl.release_year = Some(1960); assert!(resolve("Classics", "movie").matches(&cl));
        cl.release_year = Some(1990); assert!(!resolve("Classics", "movie").matches(&cl));
        let mut i = item("tv"); i.original_language = "ko".into(); assert!(resolve("International", "tv").matches(&i));
        let mut talk = item("tv"); talk.tv_type = Some("Talk Show".into()); assert!(resolve("Stand-up & Chat Shows", "tv").matches(&talk));
    }
}
