//! The genre pages the clients offer, with the ids their addresses use (`/browse/genre/<id>?bc=<page id>`, as on Netflix): the order of the picker, the id and the name of each.
//! The ids are the TMDB genre ids where there is one and ids from 10001 for the collections TMDB has no genre for; the page ids are the Series page (83) and the Films page (34399).

pub const SERIES_PAGE_ID: u32 = 83;
pub const FILMS_PAGE_ID: u32 = 34399;

pub const MOVIE_PAGES: &[(u32, &str)] = &[
    (28, "Action"),
    (16, "Anime"),
    (10003, "Black Stories"),
    (10005, "British"),
    (10009, "Classics"),
    (35, "Comedies"),
    (80, "Crime"),
    (10010, "Cult"),
    (99, "Documentaries"),
    (18, "Dramas"),
    (10006, "European"),
    (14, "Fantasy"),
    (10016, "Halloween"),
    (10008, "Hollywood"),
    (27, "Horror"),
    (10011, "Independent"),
    (10012, "International"),
    (10751, "Kids & Family"),
    (10001, "LGBTQ"),
    (10402, "Music & Musicals"),
    (10749, "Romance"),
    (878, "Sci-Fi"),
    (10013, "Shorts"),
    (10014, "Sports"),
    (10017, "Stand-up Comedy"),
    (53, "Thriller"),
];

pub const TV_PAGES: &[(u32, &str)] = &[
    (10759, "Action"),
    (16, "Anime"),
    (10003, "Black Stories"),
    (10005, "British"),
    (35, "Comedies"),
    (80, "Crime"),
    (99, "Documentary"),
    (18, "Dramas"),
    (10018, "Emmys"),
    (10006, "European"),
    (10016, "Halloween"),
    (27, "Horror"),
    (10012, "International"),
    (10762, "Kids"),
    (9648, "Mystery"),
    (10764, "Reality"),
    (10749, "Romance"),
    (10765, "Sci-Fi & Fantasy"),
    (10019, "Science & Nature"),
    (10014, "Sport"),
    (10020, "Stand-up & Chat Shows"),
    (10015, "Teen"),
    (53, "Thriller"),
    (10008, "US"),
];

/// The genre list of a surface (`movie` or `tv`).
pub fn pages(surface: &str) -> &'static [(u32, &'static str)] {
    if surface == "tv" { TV_PAGES } else { MOVIE_PAGES }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_page_name_resolves_to_a_rule_and_ids_are_unique_per_surface() {
        for surface in ["movie", "tv"] {
            let mut ids = std::collections::HashSet::new();
            for (id, name) in pages(surface) {
                assert!(ids.insert(*id), "{surface}: id {id} twice");
                assert!(crate::genre_spec::resolve(name, surface) != crate::genre_spec::GenreSpec::Unsupported || *name == "Astrology", "{surface}: {name} has no rule");
            }
        }
    }
}
