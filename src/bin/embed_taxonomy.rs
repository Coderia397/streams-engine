/// Rust binary: embed taxonomy vibe_queries via llama.cpp and write taxonomy_embeddings.bin.
///
/// Usage:
///   embed-taxonomy --server http://localhost:8086 \
///                  --db /pstream-catalog/vibe_universe.db \
///                  --out-dir /pstream-catalog
///
/// Reads:   vibe_universe.db → mood_anchors table (title, curated_keywords)
/// Writes:  taxonomy_embeddings.bin  (u32 count + n × 2560 f32)
///          taxonomy_embeddings.json ({entries: [{id, title, media_type}]})
///
/// Embeds:  "{title}: {curated_keywords}" via llama.cpp /embedding endpoint

use anyhow::{bail, Result};
use clap::Parser;
use reqwest::Client;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const DIMS: usize = 2560;

#[derive(Parser, Debug)]
#[command(name = "embed-taxonomy", about = "Embed mood_anchors from vibe_universe.db via llama.cpp")]
struct Args {
    /// llama.cpp embedding server URL
    #[arg(long, default_value = "http://localhost:8086")]
    server: String,

    /// Path to vibe_universe.db
    #[arg(long)]
    db: PathBuf,

    /// Output directory
    #[arg(long)]
    out_dir: PathBuf,
}

#[derive(Serialize)]
struct EmbedRequest {
    content: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct SidecarEntry {
    id:         String,
    title:      String,
    media_type: String,
}

#[derive(Serialize)]
struct Sidecar {
    entries: Vec<SidecarEntry>,
}

struct VibeEntry {
    id:       i64,
    title:    String,
    keywords: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.out_dir)?;

    // Load mood_anchors from vibe_universe.db
    println!("Loading mood_anchors from {}...", args.db.display());
    let conn = Connection::open(&args.db)?;
    let mut stmt = conn.prepare(
        "SELECT id, title, curated_keywords FROM mood_anchors ORDER BY id"
    )?;
    let entries: Vec<VibeEntry> = stmt
        .query_map([], |row| {
            Ok(VibeEntry {
                id:       row.get(0)?,
                title:    row.get::<_, String>(1).unwrap_or_default(),
                keywords: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    println!("  {} mood_anchors loaded", entries.len());

    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(entries.len());
    let mut sidecar_entries: Vec<SidecarEntry> = Vec::with_capacity(entries.len());

    for (i, entry) in entries.iter().enumerate() {
        // Embed: "title: keywords"
        let text = format!("{}: {}", entry.title, entry.keywords);

        let resp = client
            .post(format!("{}/embedding", args.server))
            .json(&EmbedRequest { content: text.clone() })
            .send()
            .await?;

        if !resp.status().is_success() {
            bail!("Server {} for entry #{i} '{}'", resp.status(), entry.title);
        }

        let body: EmbedResponse = resp.json().await?;

        if body.embedding.len() != DIMS {
            bail!("Expected {DIMS}d, got {}d for '{}'", body.embedding.len(), entry.title);
        }

        // L2-normalise
        let norm: f32 = body.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let embedding: Vec<f32> = if norm > 1e-8 {
            body.embedding.iter().map(|x| x / norm).collect()
        } else {
            body.embedding
        };

        all_embeddings.push(embedding);
        sidecar_entries.push(SidecarEntry {
            id:         format!("mood_{:04}", entry.id),
            title:      entry.title.clone(),
            media_type: "both".to_string(),
        });

        if (i + 1) % 50 == 0 || i == 0 {
            println!("  [{}/{}] {}", i + 1, entries.len(), entry.title);
        }
    }

    // Write binary
    let n = all_embeddings.len();
    let bin_path = args.out_dir.join("taxonomy_embeddings.bin");
    let mut bin: Vec<u8> = Vec::with_capacity(4 + n * DIMS * 4);
    bin.extend_from_slice(&(n as u32).to_le_bytes());
    for emb in &all_embeddings {
        for &v in emb {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    fs::write(&bin_path, &bin)?;
    println!("\nWritten {}", bin_path.display());

    // Write sidecar JSON
    let json_path = args.out_dir.join("taxonomy_embeddings.json");
    fs::write(&json_path, serde_json::to_string_pretty(&Sidecar { entries: sidecar_entries })?)?;
    println!("Written {}", json_path.display());
    println!("\nDone! {} taxonomy embeddings ready.", n);

    Ok(())
}
