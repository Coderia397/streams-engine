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

/// The Kids profile's own menus (Netflix's Kids Series and Kids Films genre lists, note 97): ids from 20001 (series) and 21001 (films); the three pages both menus
/// share (For Little Ones, Science & Arts, Science & Nature) carry one id and mix films and series, as on Netflix.
pub const KIDS_SERIES_PAGE_ID: u32 = 20000;
pub const KIDS_FILMS_PAGE_ID: u32 = 21000;

pub const KIDS_TV_PAGES: &[(u32, &str)] = &[
    (20001, "Action"),
    (20002, "Animal Time"),
    (20003, "Cars, Trucks & Trains"),
    (20004, "Dinosaurs"),
    (20005, "Fantasy"),
    (20006, "Feel-good"),
    (22001, "For Little Ones"),
    (20008, "Friends"),
    (20009, "Funny"),
    (20010, "Girls Take the Lead"),
    (20011, "Reality"),
    (20012, "Sci-Fi"),
    (22002, "Science & Arts"),
    (22003, "Science & Nature"),
    (20015, "Singing & Dancing"),
    (20016, "Spooky Stuff"),
    (20017, "Watch with the Family"),
];

pub const KIDS_MOVIE_PAGES: &[(u32, &str)] = &[
    (21001, "Action"),
    (21002, "Adventure"),
    (21003, "Animals"),
    (21004, "Animated"),
    (21005, "Cool Special Effects"),
    (21006, "Documentaries"),
    (21007, "Fantasy"),
    (21008, "Feel-good"),
    (22001, "For Little Ones"),
    (21010, "Friends"),
    (21011, "Funny"),
    (21012, "Girls Take the Lead"),
    (21013, "For Little Kids"),
    (21014, "Princess Tales"),
    (21015, "Sci-Fi"),
    (22002, "Science & Arts"),
    (22003, "Science & Nature"),
    (21018, "Singing & Dancing"),
    (21019, "Spooky Stuff"),
    (21020, "Watch with the Family"),
];

/// The genre list of a surface (`movie` or `tv`).
pub fn pages(surface: &str) -> &'static [(u32, &'static str)] {
    if surface == "tv" { TV_PAGES } else { MOVIE_PAGES }
}

/// The genre list of a surface for a children's profile.
pub fn kids_pages(surface: &str) -> &'static [(u32, &'static str)] {
    if surface == "tv" { KIDS_TV_PAGES } else { KIDS_MOVIE_PAGES }
}

/// The surface a Kids page id belongs to: "both" for the pages both menus share.
pub fn kids_page_surface(id: u32) -> &'static str {
    match id {
        22001..=22999 => "both",
        20001..=20999 => "tv",
        _ => "movie",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_page_name_resolves_to_a_rule_and_ids_are_unique_per_surface() {
        for surface in ["movie", "tv"] {
            let mut ids = crate::det::HashSet::default();
            for (id, name) in pages(surface).iter().chain(kids_pages(surface)) {
                assert!(ids.insert(*id), "{surface}: id {id} twice");
                assert!(crate::genre_spec::resolve(name, surface) != crate::genre_spec::GenreSpec::Unsupported || *name == "Astrology", "{surface}: {name} has no rule");
            }
        }
    }
}
