//! Who may see which title on which page: genre rule, age limit, audience, news and seasonal nets, vote and quality gates.
use super::*;

/// The page's genre rule. The Pride page trusts the model's profile when the title has one: LGBTQ+ must be central to the story (measured
/// 51 of 53 correct on titles graded by hand, against 18 of 50 for "present"); a title without a profile keeps the keyword rule.
pub fn genre_ok_with(spec: Option<&crate::genre_spec::GenreSpec>, profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    let Some(spec) = spec else { return true };
    if matches!(spec, crate::genre_spec::GenreSpec::Lgbtq) {
        if let Some(p) = profiles.get(item.id) {
            return p.lgbtq == crate::profiles::Lgbtq::Central;
        }
    }
    spec.matches(item)
}

/// May this tier see the title? A real certificate decides (see `Maturity::allows`). A title with no certificate but a profile is judged by
/// the profile (strict rule for children, no sexual content for teenagers); without either the older genre and keyword rules apply.
pub fn maturity_allows_with(m: crate::identity::Maturity, profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    use crate::identity::Maturity;
    if item.cert == 0 && !matches!(m, Maturity::Adult) {
        if let Some(p) = profiles.get(item.id) {
            if item.adult || item.flags & crate::catalog::FLAG_SUGGESTIVE != 0 {
                return false;
            }
            return match m {
                Maturity::LittleKids | Maturity::OlderKids => p.suits_children(),
                Maturity::Twelve => p.suits_teens(),
                _ => !p.content.iter().any(|c| matches!(c.as_str(), "sex" | "nudity")),
            };
        }
    }
    m.allows(item)
}

/// A page of an adult genre (Action, Thriller, Pride...) does not carry children's titles for an adult or an anonymous caller. Kids profiles are
/// limited by their maturity instead, and pages about kids, family, animation or anime keep them.
pub fn audience_fits_with(spec: Option<&crate::genre_spec::GenreSpec>, maturity: Option<crate::identity::Maturity>, profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    let Some(spec) = spec else { return true };
    if spec.keeps_children() || maturity.is_some_and(|m| m.is_kids()) {
        return true;
    }
    if item.cert != 0 {
        return item.cert >= 3; // U and PG are children's and family ratings
    }
    if let Some(p) = profiles.get(item.id) {
        return p.audience & (crate::profiles::TEEN | crate::profiles::ADULT) != 0;
    }
    !item.genres.iter().any(|g| g.eq_ignore_ascii_case("Kids") || g.eq_ignore_ascii_case("Family"))
}

/// Whether a title is made for children, for a children's home page: the title's profile says kids or family, or (no profile) it is a Kids, Family
/// or Animation title. A U or PG certificate alone is not enough: an old U-rated Western is safe for a child and still not a child's title.
pub fn kid_interest(profiles: &crate::profiles::ProfileStore, item: &CatalogItem) -> bool {
    match profiles.get(item.id) {
        Some(p) => p.audience & (crate::profiles::KIDS | crate::profiles::FAMILY) != 0,
        None => item.genres.iter().any(|g| matches!(g.as_str(), "Kids" | "Family" | "Animation")),
    }
}

/// Whether a title's original language belongs to a language page: TMDB writes Chinese as `zh` or `cn` and Norwegian as `no` or `nb`.
pub fn language_matches(page: &str, original: &str) -> bool {
    page == original || matches!((page, original), ("zh", "cn") | ("no", "nb"))
}

/// A Top 10 needs real titles: art, a decent rating, scripted or documentary TV.
pub fn chart_ok(item: &CatalogItem) -> bool {
    item.vote_average >= 6.0
        && item.has_poster()
        && item.has_backdrop()
        && item.has_overview()
        && !matches!(item.tv_type.as_deref(), Some("Reality") | Some("Video") | Some("News") | Some("Talk Show") | Some("Soap"))
}

/// Vote count floor (see `PageContext::reliable`): 10 votes on default pages, 5 on genre pages.
pub fn meets_vote_floor(item: &CatalogItem, genre_page: bool) -> bool {
    !safety_nets() || item.vote_count == 0 || item.vote_count >= if genre_page { 5 } else { 10 }
}

/// Cheap quality gate for cluster members: drops obscure, unrated and stub entries.
pub fn passes_member_quality(item: &CatalogItem) -> bool {
    // The catalog has no vote_count, so popularity is the proxy for a reliable rating.
    // Measured pool: popularity >= 2 with rating >= 6.0 keeps about 38k titles of 700k.
    item.vote_average >= 6.0 && item.popularity >= 2.0 && item.has_overview() && item.has_poster() && (!safety_nets() || item.has_backdrop())
}

/// The quality gate of a page: a page whose members the Colab judge vetted (Stand-up, Cult, Independent ...) accepts less known titles (rating 5.5, popularity 0.5), since
/// the niche is made of them; every other page uses `passes_member_quality`.
pub fn member_quality_for(spec: Option<&crate::genre_spec::GenreSpec>, item: &CatalogItem) -> bool {
    if spec.is_some_and(|s| s.is_judged()) {
        item.vote_average >= 5.5 && item.popularity >= 0.5 && item.has_overview() && item.has_poster() && (!safety_nets() || item.has_backdrop())
    } else {
        passes_member_quality(item)
    }
}

impl<'a> PageContext<'a> {
    /// A title this page may show in a row made from a request or a group: the page's filters, its quality gate and its vote floor (also served by /api/check/page-pool).
    pub fn page_pool_member(&self, item: &CatalogItem) -> bool {
        self.matches_filters(item) && self.member_ok(item) && self.reliable(item)
    }

    pub(crate) fn matches_filters(&self, item: &CatalogItem) -> bool {
        (self.surface == "both" || item.media_type == self.surface || item.media_type == "both")
            && self.language.is_none_or(|l| language_matches(l, &item.original_language))
            && self.genre_ok(item)
            && self.audience_fits(item)
            && self.maturity.is_none_or(|m| self.maturity_allows(m, item))
            && (!safety_nets() || self.allowed_by_nets(item))
    }

    pub(crate) fn genre_ok(&self, item: &CatalogItem) -> bool {
        genre_ok_with(self.genre_spec, self.profiles, item)
    }

    pub(crate) fn maturity_allows(&self, m: crate::identity::Maturity, item: &CatalogItem) -> bool {
        maturity_allows_with(m, self.profiles, item)
    }

    pub(crate) fn audience_fits(&self, item: &CatalogItem) -> bool {
        audience_fits_with(self.genre_spec, self.maturity, self.profiles, item)
    }

    /// Vote count floor: a rating from a handful of votes says nothing. Known only for pulled titles (all of the
    /// quality pool); genre pages use a lower floor so small collections survive.
    pub(crate) fn reliable(&self, item: &CatalogItem) -> bool {
        meets_vote_floor(item, self.genre_spec.is_some() || self.language.is_some())
    }

    /// News, talk and soap never fill default rows (a page that asks for them keeps them); seasonal titles only
    /// appear inside their window (the Halloween page lifts the Halloween rule).
    pub(crate) fn allowed_by_nets(&self, item: &CatalogItem) -> bool {
        use crate::catalog::{FLAG_CHRISTMAS, FLAG_HALLOWEEN, FLAG_NONSCRIPTED, FLAG_SUGGESTIVE, FLAG_WRESTLING_EVENT};
        let f = item.flags & crate::catalog::FLAGS_RULES;
        if f == 0 {
            return true;
        }
        if f & FLAG_SUGGESTIVE != 0 {
            return false;
        }
        if f & FLAG_NONSCRIPTED != 0 && !self.genre_spec.is_some_and(|s| s.includes_nonscripted()) {
            return false;
        }
        // pay per view wrestling events are not default home content (the Sports page keeps them)
        if f & FLAG_WRESTLING_EVENT != 0 && !matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Sports)) {
            return false;
        }
        let (m, d) = self.today;
        let md = (m as u32) * 100 + d as u32;
        if f & FLAG_CHRISTMAS != 0 && !(md >= 1120 || md <= 106) {
            return false;
        }
        let halloween_page = matches!(self.genre_spec, Some(crate::genre_spec::GenreSpec::Halloween));
        if f & FLAG_HALLOWEEN != 0 && !halloween_page && !(1001..=1101).contains(&md) {
            return false;
        }
        true
    }

    /// Filters of the page without its genre rule (a seasonal page holds titles that are near the season without carrying its keyword).
    pub(crate) fn matches_filters_any_genre(&self, item: &CatalogItem) -> bool {
        (self.surface == "both" || item.media_type == self.surface || item.media_type == "both")
            && self.language.is_none_or(|l| language_matches(l, &item.original_language))
            && self.audience_fits(item)
            && self.maturity.is_none_or(|m| self.maturity_allows(m, item))
            && (!safety_nets() || self.allowed_by_nets(item))
    }

    /// A title made for children (TMDB Kids genre, or a US TV-Y or TV-Y7 rating): on a page that is not a child's it stays out of the personal rows, the popular row and
    /// the composed rows that are not about children or animation (Miraculous next to Breaking Bad in Top Picks, iCarly next to The Boys in US Series).
    pub(crate) fn childrens_title(&self, item: &CatalogItem) -> bool {
        item.genres.iter().any(|g| g == "Kids") || matches!(item.cert_us, 6 | 7)
    }

    /// The quality gate of this page (see `member_quality_for`).
    pub(crate) fn member_ok(&self, item: &CatalogItem) -> bool {
        member_quality_for(self.genre_spec, item)
    }
}
