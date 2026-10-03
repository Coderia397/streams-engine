/// embed-catalog: embed ALL catalog items via Qwen3 and write/extend vectors.bin
///
/// Usage:
///   embed-catalog --server http://localhost:8086 \
///                 --db /pstream-catalog/catalog.db \
///                 --out /pstream-catalog/vectors.bin \
///                 [--batch 32] [--resume]
///
/// On first run:   writes vectors.bin from scratch (sorted by popularity DESC).
/// With --resume:  skips already-embedded items (reads current file size → offset).
///
/// Output format: raw f32 LE, N × 2560 floats, L2-normalised.
/// IDs implied by ORDER BY popularity DESC — same ordering VectorStore uses.

use anyhow::{bail, Result};
use clap::Parser;
use reqwest::Client;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DIMS: usize = 2560;

#[derive(Parser, Debug)]
#[command(name = "embed-catalog", about = "Embed full TMDB catalog via Qwen3 embedding server")]
struct Args {
    /// Qwen3 llama.cpp server URL
    #[arg(long, default_value = "http://localhost:8086")]
    server: String,

    /// catalog.db path
    #[arg(long)]
    db: PathBuf,

    /// Output vectors.bin path
    #[arg(long)]
    out: PathBuf,

    /// Batch size per request
    #[arg(long, default_value_t = 32)]
    batch: usize,

    /// Resume from existing file (skip already-embedded items)
    #[arg(long, default_value_t = true)]
    resume: bool,
}

#[derive(Serialize)]
struct EmbedReq { content: String }

#[derive(Deserialize)]
struct EmbedResp { embedding: Vec<f32> }

fn format_item(title: &str, overview: &str, genres: &str) -> String {
    // Same format as original Qwen3 embedding pipeline
    let title = title.trim().chars().take(120).collect::<String>();
    let overview = overview.trim().chars().take(750).collect::<String>();
    let genres = genres.trim().chars().take(120).collect::<String>();
    format!("{title}: {overview} | Genres: {genres}")
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Determine how many items already embedded
    let already_done = if args.resume && args.out.exists() {
        let sz = std::fs::metadata(&args.out)?.len() as usize;
        if sz % (DIMS * 4) != 0 {
            bail!("vectors.bin size {} not aligned to DIMS×4 — file may be corrupt", sz);
        }
        sz / (DIMS * 4)
    } else {
        0
    };

    println!("Already embedded: {} items", already_done);

    // Load all catalog items sorted by popularity DESC
    let conn = Connection::open(&args.db)?;
    let mut stmt = conn.prepare(
        "SELECT id, title, overview, genres
         FROM catalog
         ORDER BY popularity DESC"
    )?;

    struct Item { id: i64, title: String, overview: String, genres: String }

    let all_items: Vec<Item> = stmt.query_map([], |row| {
        Ok(Item {
            id:       row.get(0)?,
            title:    row.get::<_,Option<String>>(1)?.unwrap_or_default(),
            overview: row.get::<_,Option<String>>(2)?.unwrap_or_default(),
            genres:   row.get::<_,Option<String>>(3)?.unwrap_or_default(),
        })
    })?.filter_map(|r| r.ok()).collect();

    let total = all_items.len();
    let to_embed = &all_items[already_done..];

    println!("Total catalog: {} | To embed: {} | Starting from offset {}",
        total, to_embed.len(), already_done);

    if to_embed.is_empty() {
        println!("Nothing to do — all items already embedded.");
        return Ok(());
    }

    // Open output file (append if resuming, create otherwise)
    let mut file = OpenOptions::new()
        .create(true)
        .append(args.resume)
        .write(true)
        .open(&args.out)?;

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    // Verify server is up
    let health = client.get(format!("{}/health", args.server)).send().await;
    if health.is_err() {
        bail!("Embedding server not responding at {} — start Qwen3 first", args.server);
    }
    println!("Server OK. Embedding {} items in batches of {}...\n",
        to_embed.len(), args.batch);

    let t0 = Instant::now();
    let mut done = 0usize;
    let mut errors = 0usize;

    for chunk in to_embed.chunks(args.batch) {
        let texts: Vec<String> = chunk.iter()
            .map(|item| format_item(&item.title, &item.overview, &item.genres))
            .collect();

        // Embed sequentially (one request per item for reliability)
        // Switch to batched if the server supports it
        for (i, text) in texts.iter().enumerate() {
            let resp = client
                .post(format!("{}/embedding", args.server))
                .json(&EmbedReq { content: text.clone() })
                .send()
                .await;

            match resp {
                Err(e) => {
                    eprintln!("  ⚠ Request failed for '{}': {e}", chunk[i].title);
                    // Write zeros as placeholder to maintain offset alignment
                    file.write_all(&vec![0u8; DIMS * 4])?;
                    errors += 1;
                }
                Ok(r) if !r.status().is_success() => {
                    eprintln!("  ⚠ HTTP {} for '{}'", r.status(), chunk[i].title);
                    file.write_all(&vec![0u8; DIMS * 4])?;
                    errors += 1;
                }
                Ok(r) => {
                    let body: EmbedResp = r.json().await?;
                    if body.embedding.len() != DIMS {
                        bail!("Expected {DIMS}d, got {}d for '{}'",
                            body.embedding.len(), chunk[i].title);
                    }
                    // L2-normalise
                    let norm: f32 = body.embedding.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let normed: Vec<f32> = if norm > 1e-8 {
                        body.embedding.iter().map(|x| x/norm).collect()
                    } else {
                        body.embedding
                    };
                    for &v in &normed {
                        file.write_all(&v.to_le_bytes())?;
                    }
                }
            }

            done += 1;
        }
        file.flush()?;

        // Progress
        let elapsed = t0.elapsed().as_secs_f32();
        let rate = done as f32 / elapsed;
        let remaining = to_embed.len() - done;
        let eta_min = remaining as f32 / rate / 60.0;
        let pct = 100.0 * (already_done + done) as f32 / total as f32;
        print!("\r  [{:.1}%] {}/{} embedded | {:.1}/s | ETA {:.0}m | errors: {}   ",
            pct, already_done + done, total, rate, eta_min, errors);
        let _ = std::io::stdout().flush();
    }

    println!("\n\nDone! {} embedded, {} errors.", done, errors);
    let total_vecs = (std::fs::metadata(&args.out)?.len() as usize) / (DIMS * 4);
    println!("vectors.bin now contains {} vectors ({:.1} GB)",
        total_vecs, total_vecs as f64 * DIMS as f64 * 4.0 / 1e9);
    println!("\nNext: run cluster-seeds on the full {} vectors.", total_vecs);

    Ok(())
}
