//! Everything a page builder knows about the request and the catalog.
use super::*;

// ── PageContext ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct PageContext<'a> {
    pub catalog:     &'a CatalogMap,
    pub vectors:     &'a VectorStore,
    pub hnsw:        &'a RwLock<Option<HnswIndex>>,
    /// Index over the title profile vectors; row definitions are matched against it when it is built
    pub seeds:       &'a ClusterSeeds,
    /// World, atom and theme group of every title (single database)
    pub groups:      &'a crate::groups::GroupMap,
    pub group_names: &'a crate::groups::GroupNames,
    /// Vibe groups, moods and seasonal sub groups
    pub vibe:        &'a crate::vibe::VibeIndex,
    /// The client's local time is between 22:00 and 04:59: late night rows come first
    pub late_night:  bool,
    pub vault:       &'a CentroidVault,
    pub taxonomy:    &'a TaxonomyStore,
    pub profiles:    &'a crate::profiles::ProfileStore,
    pub topics:      &'a crate::topics::TopicStore,
    /// Release dates of new and upcoming titles, and today's date (YYYY-MM-DD) to compare them with
    pub fresh:       &'a crate::fresh::FreshStore,
    pub today_ymd:   String,
    /// Original language code (a language page); every row holds only titles in that language
    pub language:    Option<&'a str>,
    /// Children's home only: build the page from the animated titles (`Some(true)`) or from the live action ones (`Some(false)`)
    pub animation:   Option<bool>,
    /// The Kids home is built from the Kids row vocabulary (`taxonomy` is then that vocabulary): rows chosen by value and variety, none at a fixed place
    pub kids_entries: bool,
    pub surface:     &'a str,   // "movie" | "tv" | "both"
    /// Optional genre as sent by the client (kept for row titles).
    pub genre:       Option<&'a str>,
    /// The rule that genre resolves to, applied to every row (see genre_spec.rs).
    pub genre_spec:  Option<&'a crate::genre_spec::GenreSpec>,
    pub taste:       Vec<f32>,  // blended query vector on unit sphere
    pub target_rows: usize,
    /// Corpus mean vector for centred scoring
    pub mean:        &'a [f32],
    /// Today's (month, day) for calendar rows
    pub today:       (u8, u8),
    /// Session id: seeds the exploration noise so a session sees a stable page
    pub session:     &'a str,
    /// Maturity of the verified caller's profile; none for anonymous requests (see identity.rs)
    pub maturity:    Option<crate::identity::Maturity>,
    /// Fill pass of a thin genre page: the pool is these titles (neighbours of what the page shows) instead of the whole catalog
    pub fill_pool:   Option<&'a [u32]>,
    /// Titles a fill pass must not use again (the ones the page already shows)
    pub skip:        Option<&'a HashSet<u32>>,
    /// What the client sent about the profile: saved, recent and liked titles (personal rows)
    pub signals:     &'a crate::signals::Signals,
}
