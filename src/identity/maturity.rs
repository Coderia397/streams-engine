//! The age tiers of a profile and what each tier may see.
use crate::catalog::{CatalogItem, FLAG_SUGGESTIVE};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Maturity {
    LittleKids,
    OlderKids,
    Twelve,
    Fifteen,
    Adult,
}

/// Genres that keep a title out of both kids tiers (TMDB movie and TV names).
const KIDS_BLOCKED_GENRES: [&str; 11] = [
    "Horror", "Crime", "Thriller", "Mystery", "War", "War & Politics", "Reality", "Talk", "News", "Soap", "Western",
];
/// Keywords that mark violent or sexual content. The catalogue has no certifications, so this is a conservative stand in;
/// it only applies to titles that carry keywords (pulled titles).
const KIDS_BLOCKED_KEYWORDS: [&str; 14] = [
    "gore", "violence", "blood", "murder", "serial killer", "torture", "rape", "sex", "nudity", "drugs", "suicide", "zombie",
    "slasher", "erotic",
];

impl Maturity {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "little_kids" => Some(Self::LittleKids),
            "older_kids" => Some(Self::OlderKids),
            "teens" | "twelve" => Some(Self::Twelve),
            "fifteen" => Some(Self::Fifteen),
            "adult" => Some(Self::Adult),
            _ => None,
        }
    }

    /// From a British age rating as the profile setting saves it: U, PG, 12, 15 or 18.
    pub fn from_rating(s: &str) -> Option<Self> {
        match s.trim().to_ascii_uppercase().as_str() {
            "U" => Some(Self::LittleKids),
            "PG" => Some(Self::OlderKids),
            "12" | "12A" => Some(Self::Twelve),
            "15" => Some(Self::Fifteen),
            "18" => Some(Self::Adult),
            _ => None,
        }
    }

    /// The highest certificate rank this tier may see (1 U, 2 PG, 3 12, 4 15, 5 18).
    pub fn max_cert(self) -> u8 {
        match self {
            Self::LittleKids => 1,
            Self::OlderKids => 2,
            Self::Twelve => 3,
            Self::Fifteen => 4,
            Self::Adult => 5,
        }
    }

    /// Kids tiers (U and PG) are the ones for children's profiles.
    pub fn is_kids(self) -> bool {
        matches!(self, Self::LittleKids | Self::OlderKids)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LittleKids => "little_kids",
            Self::OlderKids => "older_kids",
            Self::Twelve => "twelve",
            Self::Fifteen => "fifteen",
            Self::Adult => "adult",
        }
    }

    /// May a profile with this maturity see this title?
    ///  * adult: everything (the page builders' own safety nets still apply)
    ///  * teens: no adult flagged and no suggestive titles
    ///  * older kids: a known rating up to PG, no blocked genre or keyword, and the title must be kids, family or animation (never unrated)
    ///  * little kids: a known rating of U, as older kids, but animation alone is not enough (it must be kids or family)
    pub fn allows(self, item: &CatalogItem) -> bool {
        if self == Self::Adult {
            return true;
        }
        if item.adult || item.flags & FLAG_SUGGESTIVE != 0 {
            return false;
        }
        if self.is_kids() {
            // a title for a Kids profile needs a known rating within the tier AND to be made for children: a PG murder mystery or a rated sitcom is not (the rating alone
            // never decides for Kids), and an unrated title is never shown
            if item.cert == 0 || item.cert > self.max_cert() {
                return false;
            }
            let has = |name: &str| item.genres.iter().any(|g| g.eq_ignore_ascii_case(name));
            if KIDS_BLOCKED_GENRES.iter().any(|g| has(g)) {
                return false;
            }
            if item.keywords.iter().any(|k| KIDS_BLOCKED_KEYWORDS.iter().any(|b| k.contains(b))) {
                return false;
            }
            let kids_or_family = has("Kids") || has("Family");
            return match self {
                Self::LittleKids => kids_or_family,
                _ => kids_or_family || has("Animation"),
            };
        }
        // a known age rating decides (12 or 15 tiers); an unrated title is allowed there
        if item.cert != 0 {
            return item.cert <= self.max_cert();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rated(cert: u8) -> CatalogItem {
        let mut it = item(&["Drama"], &[], false, 0);
        it.cert = cert;
        it
    }

    /// The same title with a known rating (U unless said otherwise).
    fn with_cert(mut it: CatalogItem, cert: u8) -> CatalogItem {
        it.cert = cert;
        it
    }

    #[test]
    fn a_known_age_rating_decides_what_each_tier_sees() {
        // little kids see U only, older kids up to PG (and only titles made for children), teens up to 15, adults everything
        let kids_title = |c| with_cert(item(&["Kids", "Family"], &[], false, 0), c);
        assert!(Maturity::LittleKids.allows(&kids_title(1)) && !Maturity::LittleKids.allows(&kids_title(2)));
        assert!(Maturity::OlderKids.allows(&kids_title(2)) && !Maturity::OlderKids.allows(&kids_title(3)));
        // a PG drama or murder mystery is not for children whatever its rating says, and an unrated title is never shown to them
        assert!(!Maturity::OlderKids.allows(&rated(2)));
        assert!(!Maturity::OlderKids.allows(&with_cert(item(&["Mystery", "Drama", "Crime"], &[], false, 0), 2)));
        assert!(!Maturity::OlderKids.allows(&with_cert(item(&["Family"], &[], false, 0), 0)));
        assert!(Maturity::Twelve.allows(&rated(3)) && !Maturity::Twelve.allows(&rated(4)));
        assert!(Maturity::Fifteen.allows(&rated(4)) && !Maturity::Fifteen.allows(&rated(5)));
        assert!(Maturity::Adult.allows(&rated(5)));
    }

    fn item(genres: &[&str], keywords: &[&str], adult: bool, flags: u8) -> CatalogItem {
        CatalogItem {
            id: 1,
            tmdb_id: 1,
            media_type: "tv".into(),
            title: "t".into(),
            release_year: None,
            genres: genres.iter().map(|s| s.to_string()).collect(),
            overview: String::new(),
            tagline: None,
            director: None,
            poster_path: None,
            backdrop_path: None,
            vote_average: 7.0,
            popularity: 10.0,
            is_anime: false,
            vote_count: 100,
            adult,
            original_language: "en".into(),
            tv_type: None,
            countries: vec![],
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            runtime: 0,
            networks: vec![],
            flags,
            cert: 0, cert_gb: 0, cert_us: 0,
        }
    }

    #[test]
    fn maturity_policy() {
        let kids_show = with_cert(item(&["Kids", "Animation"], &[], false, 0), 1);
        let family_film = with_cert(item(&["Family", "Comedy"], &["friendship"], false, 0), 1);
        let adult_cartoon = with_cert(item(&["Animation", "Comedy"], &[], false, 0), 2);
        let horror = with_cert(item(&["Horror", "Family"], &[], false, 0), 1);
        let violent_anime = with_cert(item(&["Animation", "Action & Adventure"], &["gore", "war"], false, 0), 2);
        let drama = with_cert(item(&["Drama"], &[], false, 0), 2);
        let adult_flag = item(&["Drama"], &[], true, 0);
        let suggestive = item(&["Animation"], &[], false, FLAG_SUGGESTIVE);

        for m in [Maturity::LittleKids, Maturity::OlderKids, Maturity::Twelve, Maturity::Fifteen, Maturity::Adult] {
            assert!(m.allows(&kids_show), "{m:?} kids show");
        }
        assert!(Maturity::LittleKids.allows(&family_film));
        assert!(!Maturity::LittleKids.allows(&adult_cartoon), "animation alone is not enough for little kids");
        assert!(Maturity::OlderKids.allows(&adult_cartoon), "older kids may see unlabelled animation");
        for m in [Maturity::LittleKids, Maturity::OlderKids] {
            assert!(!m.allows(&horror), "{m:?} horror");
            assert!(!m.allows(&violent_anime), "{m:?} violent keywords");
            assert!(!m.allows(&drama), "{m:?} plain drama");
        }
        assert!(Maturity::Fifteen.allows(&drama) && Maturity::Fifteen.allows(&horror));
        assert!(!Maturity::Fifteen.allows(&adult_flag) && !Maturity::Fifteen.allows(&suggestive));
        assert!(Maturity::Adult.allows(&adult_flag), "adult profiles keep the existing page safety nets only");
        assert_eq!(Maturity::parse(Maturity::OlderKids.as_str()), Some(Maturity::OlderKids));
    }
}
