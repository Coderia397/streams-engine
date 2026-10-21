use super::*;

fn item(rating: f32, popularity: f32, votes: u32) -> CatalogItem {
    CatalogItem {
        id: 1, tmdb_id: 1, media_type: "movie".into(), title: "t".into(), release_year: Some(2000), genres: vec![],
        overview: String::new(), tagline: None, director: None, poster_path: Some("/p.jpg".into()), backdrop_path: Some("/b.jpg".into()),
        vote_average: rating, popularity, is_anime: false, vote_count: votes, adult: false,
        original_language: "en".into(), tv_type: None, countries: vec![], keywords: vec![], runtime: 0, networks: vec![], flags: 0, cert: 0, cert_gb: 0, cert_us: 0,
    }
}

#[test]
fn blend_alternates_films_and_series() {
    let (mut a, mut b, mut c) = (item(8.0, 5.0, 400), item(8.0, 5.0, 400), item(8.0, 5.0, 400));
    a.id = 1; b.id = 2; c.id = 3; b.media_type = "tv".into(); c.media_type = "tv".into();
    let out = blend_types(vec![&b, &c, &a]);
    assert_eq!(out.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 2, 3]);
}

#[test]
fn a_gem_is_well_rated_little_known_and_voted_on() {
    assert!(is_gem(&item(8.0, 5.0, 400)));
    assert!(!is_gem(&item(6.5, 5.0, 400)), "rating too low");
    assert!(!is_gem(&item(8.0, 90.0, 400)), "too popular");
    assert!(!is_gem(&item(8.0, 5.0, 20)), "too few votes");
}

#[test]
fn event_ids_become_labels_and_targets() {
    assert_eq!(prettify("halloween"), "Halloween");
    assert_eq!(prettify("black_history_month"), "Black History Month");
    assert_eq!(event_target("pride"), json!({ "genre": "Pride" }));
    assert_eq!(event_target("diwali"), json!({ "genre": "diwali" }));
}

#[test]
fn collection_meta_carries_label_target_and_art() {
    let row = Row { title: "x".into(), archetype: "collection".into(), kind: "collection".into(), items: vec![crate::page::RowItem::from_catalog(&item(8.0, 5.0, 400))], meta: None };
    let m = collection_meta("halloween", "event.halloween", "Halloween", "Spooky, Gothic", event_target("halloween"), &row);
    assert_eq!(m["title_key"], "collection.title");
    assert_eq!(m["collection"]["target"]["genre"], "Halloween");
    assert_eq!(m["collection"]["images"], json!(["/b.jpg"]));
    assert_eq!(m["collection"]["description"], "Spooky, Gothic");
}

#[test]
fn meta_adds_the_reason_only_when_given() {
    assert!(meta("row.gems_for_you", json!({}), None).unwrap().get("reason").is_none());
    let m = meta("row.because_you_watched", json!({ "title": "A" }), Some(json!({ "kind": "recent" }))).unwrap();
    assert_eq!(m["reason"]["kind"], "recent");
}
