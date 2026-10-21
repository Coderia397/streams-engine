//! Row titles from what is in the row. Candidates run from most to least specific; the page builder takes the first one
//! not yet used on the page (page.rs `claim_title`) and falls back to the older generic titles in titles.rs.
//!
//!   {country adjective} {theme} {genre}   Korean Romantic Comedies, Japanese Detective Mysteries
//!   {theme} {genre}                       Time Travel Sci-Fi, High School Comedies, Serial Killer Thrillers
//!   {theme}                               Based on True Stories, Sitcoms, Mecha Anime
//!   {genre} Anime                         Romance Anime
//!   {country adjective} {genre}           British Crime Stories, French Thrillers
//!   {decade} {genre}                      90s Action (only when at least 70 percent of the row is in that decade)
//!
//! Themes come from TMDB keywords: a theme counts when at least 40 percent of the row carries one of its keywords.

use crate::catalog::CatalogItem;
use crate::det::HashMap;

mod fit;
mod themes;
mod titles;
mod words;

pub use fit::name_fits;
pub use titles::{candidates, mark_animated, theme_title, tone_title};
pub use words::{genre_family, genre_noun, language_label, primary_country};
pub(crate) use themes::THEMES;
pub(crate) use words::*;

#[cfg(test)]
mod tests;
