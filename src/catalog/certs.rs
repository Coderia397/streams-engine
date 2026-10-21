//! Age certificates as ranks and labels.

/// The real GB certificate as a label, None when the title has no GB rating.
pub fn gb_label(rank: u8) -> Option<&'static str> {
    Some(match rank { 1 => "U", 2 => "PG", 3 => "12", 4 => "15", 5 => "18", _ => return None })
}

/// The label a client shows for a certificate rank: the British rating, or NR (not rated) when there is none.
pub fn cert_label(cert: u8) -> &'static str {
    match cert {
        1 => "U",
        2 => "PG",
        3 => "12",
        4 => "15",
        5 => "18",
        _ => "NR",
    }
}

/// US certificate codes: 1 G, 2 PG, 3 PG-13, 4 R, 5 NC-17 (films), 6 TV-Y, 7 TV-Y7, 8 TV-G, 9 TV-PG, 10 TV-14, 11 TV-MA.
pub fn us_label(code: u8) -> Option<&'static str> {
    Some(match code {
        1 => "G", 2 => "PG", 3 => "PG-13", 4 => "R", 5 => "NC-17",
        6 => "TV-Y", 7 => "TV-Y7", 8 => "TV-G", 9 => "TV-PG", 10 => "TV-14", 11 => "TV-MA",
        _ => return None,
    })
}

/// The real GB rank (1 to 5) and US code of a certificates column, each 0 when the country has none that is a rating.
pub fn cert_codes(certs: Option<&str>) -> (u8, u8) {
    let Some(text) = certs else { return (0, 0) };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else { return (0, 0) };
    let get = |c: &str| v.get(c).and_then(|x| x.as_str()).map(|s| s.trim().to_ascii_uppercase());
    let gb = match get("GB").as_deref() {
        Some("U") | Some("UC") => 1,
        Some("PG") => 2,
        Some("12") | Some("12A") => 3,
        Some("15") => 4,
        Some("18") | Some("R18") => 5,
        _ => 0,
    };
    let us = match get("US").as_deref() {
        Some("G") => 1, Some("PG") => 2, Some("PG-13") => 3, Some("R") => 4, Some("NC-17") => 5,
        Some("TV-Y") => 6, Some("TV-Y7") | Some("TV-Y7-FV") => 7, Some("TV-G") => 8, Some("TV-PG") => 9, Some("TV-14") => 10, Some("TV-MA") => 11,
        _ => 0,
    };
    (gb, us)
}

/// British age rank from the certificates column (a JSON object of country to certificate). The GB certificate wins; otherwise the US one is
/// mapped to the nearest British rating. 0 when there is none or it is not a rating (for example "NR").
pub fn cert_rank(certs: Option<&str>) -> u8 {
    let Some(text) = certs else { return 0 };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else { return 0 };
    let get = |c: &str| v.get(c).and_then(|x| x.as_str()).map(|s| s.trim().to_ascii_uppercase());
    if let Some(gb) = get("GB") {
        match gb.as_str() {
            "U" | "UC" => return 1,
            "PG" => return 2,
            "12" | "12A" => return 3,
            "15" => return 4,
            "18" | "R18" => return 5,
            _ => {}
        }
    }
    match get("US").as_deref() {
        Some("G") | Some("TV-G") | Some("TV-Y") => 1,
        Some("PG") | Some("TV-PG") | Some("TV-Y7") | Some("TV-Y7-FV") => 2,
        Some("PG-13") => 3,
        Some("TV-14") => 4,
        Some("R") | Some("TV-MA") | Some("NC-17") => 5,
        _ => 0,
    }
}
