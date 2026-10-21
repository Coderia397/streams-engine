//! One title of the catalog and the flags computed for it once at load.

/// A single item in the streaming catalog.
#[derive(Debug, Clone)]
pub struct CatalogItem {
    /// Internal key, unique across movies and TV (see `item_key`). Use `tmdb_id` for anything shown to clients.
    pub id: u32,
    /// TMDB id. Movies and TV series share the same id space, so it is only unique together with `media_type`.
    pub tmdb_id: u32,
    pub media_type: String,
    pub title: String,
    pub release_year: Option<i32>,
    /// TMDB genre names, e.g. ["Action", "Science Fiction"]
    pub genres: Vec<String>,
    pub overview: String,
    pub tagline: Option<String>,
    pub director: Option<String>,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
    pub vote_average: f32,
    pub popularity: f32,
    pub is_anime: bool,
    /// Present only in a catalog built from the TMDB pull (catalog_v2); defaults otherwise.
    pub vote_count: u32,
    pub adult: bool,
    pub original_language: String,
    /// TMDB tv type: Scripted, Reality, Talk Show, News, Documentary, Miniseries, Video
    pub tv_type: Option<String>,
    /// ISO 3166 origin or production countries (catalog_v2 only)
    pub countries: Vec<String>,
    /// TMDB keywords, lower case (catalog_v2 only, pulled titles)
    pub keywords: Vec<String>,
    /// British age rating of the title as a rank: 0 unknown, 1 U, 2 PG, 3 12, 4 15, 5 18. Taken from the GB certificate, else the US one.
    pub cert: u8,
    /// The real certificates, kept so a client can show the system its user reads: GB rank 1 to 5 (U, PG, 12, 15, 18), and the US code (see `us_label`). 0 when absent.
    pub cert_gb: u8,
    pub cert_us: u8,
    /// Minutes: movie runtime or median episode runtime, 0 when unknown
    pub runtime: u32,
    /// Bit flags computed once at load (see FLAG_*), so page builders never scan titles or keywords per request
    pub flags: u8,
}

pub const FLAG_CHRISTMAS: u8 = 1;
pub const FLAG_HALLOWEEN: u8 = 2;
pub const FLAG_WRESTLING_EVENT: u8 = 4;
pub const FLAG_NONSCRIPTED: u8 = 8; // news, talk, soap
pub const FLAG_SUGGESTIVE: u8 = 16; // ecchi, harem, softcore, erotic: not for billboards, Top 10 or default rows
/// The flags above are the data defect rules; a page asks `flags & FLAGS_RULES` to know whether any applies
pub const FLAGS_RULES: u8 = 31;
pub const FLAG_OVERVIEW: u8 = 32; // overview of 60 characters or more
pub const FLAG_POSTER: u8 = 64; // has a poster
pub const FLAG_BACKDROP: u8 = 128; // has a backdrop

pub(crate) fn compute_flags(title: &str, keywords: &[String], tv_type: Option<&str>, overview: &str, poster: Option<&str>, backdrop: Option<&str>) -> u8 {
    let t = title.to_lowercase();
    let mut f = 0u8;
    if overview.chars().take(60).count() >= 60 {
        f |= FLAG_OVERVIEW;
    }
    if poster.is_some_and(|p| !p.is_empty()) {
        f |= FLAG_POSTER;
    }
    if backdrop.is_some_and(|b| !b.is_empty()) {
        f |= FLAG_BACKDROP;
    }
    if keywords.iter().any(|k| k.contains("christmas")) || t.contains("christmas") || t.contains("xmas") || t.contains("santa claus") {
        f |= FLAG_CHRISTMAS;
    }
    if keywords.iter().any(|k| k.contains("halloween")) || t.contains("halloween") {
        f |= FLAG_HALLOWEEN;
    }
    if keywords.iter().any(|k| k == "pro wrestling" || k == "professional wrestling")
        || ["wwe ", "wwf ", "wcw ", "wrestlemania", "summerslam", "royal rumble", "survivor series"].iter().any(|w| t.contains(w))
    {
        f |= FLAG_WRESTLING_EVENT;
    }
    if matches!(tv_type, Some("News") | Some("Talk Show") | Some("Soap")) {
        f |= FLAG_NONSCRIPTED;
    }
    const SUGGESTIVE: &[&str] = &[
        "ecchi", "harem", "reverse harem", "softcore", "eroticism", "erotic", "erotic movie", "erotic thriller", "sex", "sex comedy",
        "nudity", "sexy", "pornography", "stripper", "brothel", "hentai", "sexual humor",
    ];
    if keywords.iter().any(|k| SUGGESTIVE.contains(&k.as_str())) {
        f |= FLAG_SUGGESTIVE;
    }
    f
}

impl CatalogItem {
    /// Overview of 60 characters or more, poster and backdrop present: flags set once at load, so a scan of the catalog never counts characters or compares strings
    #[inline]
    pub fn has_overview(&self) -> bool {
        self.flags & FLAG_OVERVIEW != 0
    }

    #[inline]
    pub fn has_poster(&self) -> bool {
        self.flags & FLAG_POSTER != 0
    }

    #[inline]
    pub fn has_backdrop(&self) -> bool {
        self.flags & FLAG_BACKDROP != 0
    }

}
