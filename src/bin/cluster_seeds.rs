/// Rust binary: offline spherical k-means → cluster_seeds_v2.bin (member ids are type aware keys)
///
/// Usage:
///   cluster-seeds --vecs /pstream-catalog/vectors.bin \
///                 --db   /pstream-catalog/catalog.db  \
///                 --k 500 --iters 80 --out-dir /pstream-catalog

use anyhow::{bail, Result};
use clap::Parser;
use memmap2::MmapOptions;
use pstream_engine::vectors::resolve_ids;
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const DIMS: usize = 2560;

#[derive(Parser, Debug)]
#[command(name = "cluster-seeds", about = "Spherical k-means over catalog vectors")]
struct Args {
    #[arg(long)] vecs:    PathBuf,
    #[arg(long)] db:      PathBuf,
    #[arg(long, default_value_t = 500)]  k:     usize,
    #[arg(long, default_value_t = 30)]   iters: usize,
    /// Centroids are trained on a random sample, then every vector is assigned once.
    #[arg(long, default_value_t = 120_000)] sample: usize,
    /// Members kept per cluster (closest to the centroid first).
    #[arg(long, default_value_t = 300)]  max_members: usize,
    #[arg(long)] out_dir: PathBuf,
    #[arg(long, default_value_t = 42)]   seed:  u64,
}

fn nearest(row: &[f32], centroids: &[f32], k: usize) -> (usize, f32) {
    let mut best = (0usize, f32::MIN);
    for c in 0..k {
        let s = dot(row, &centroids[c * DIMS..(c + 1) * DIMS]);
        if s > best.1 {
            best = (c, s);
        }
    }
    best
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.out_dir)?;
    let t0 = Instant::now();

    // ── Map vectors.bin (raw f32, L2-normalised) ──────────────────────────────
    println!("Mapping vectors from {} ...", args.vecs.display());
    let file = fs::File::open(&args.vecs)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let (n, half) = pstream_engine::vectors::detect_layout(&args.vecs.to_string_lossy(), mmap.len())?;
    // f16 files are widened to f32 in memory; f32 files are used straight from the mapping
    // (mmap is page aligned, so the f32 view is valid).
    let widened: Vec<f32>;
    let data: &[f32] = if half {
        widened = mmap.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        &widened
    } else {
        unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, n * DIMS) }
    };
    let row = |i: usize| &data[i * DIMS..(i + 1) * DIMS];
    println!("  {} × {} ({:.1}s)", n, DIMS, t0.elapsed().as_secs_f32());

    // ── IDs (sidecar, frozen from catalog.db on first use) ────────────────────
    let ids = resolve_ids(&args.vecs.to_string_lossy(), &args.db.to_string_lossy(), n)?;

    // Rows with a near zero norm are failed-embedding placeholders: skip them.
    let valid: Vec<usize> = (0..n)
        .into_par_iter()
        .filter(|&i| dot(row(i), row(i)) > 0.5)
        .collect();
    println!("  {} valid vectors, {} zero placeholders", valid.len(), n - valid.len());

    let k = args.k.min(valid.len());
    let mut rng = Lcg64(args.seed);

    // ── Sample for training ───────────────────────────────────────────────────
    let sample_n = args.sample.min(valid.len());
    let sample: Vec<usize> = (0..sample_n)
        .map(|_| valid[(rng.next() % valid.len() as u64) as usize])
        .collect();

    println!("Initialising {} centroids from sample of {} ...", k, sample_n);
    let mut centroids: Vec<f32> = vec![0.0f32; k * DIMS];
    for c in 0..k {
        let idx = sample[(rng.next() % sample_n as u64) as usize];
        centroids[c * DIMS..(c + 1) * DIMS].copy_from_slice(row(idx));
    }

    // ── Lloyd's iterations on the sample ──────────────────────────────────────
    let mut assignments = vec![usize::MAX; sample_n];
    for iter in 0..args.iters {
        let t_iter = Instant::now();
        let new_assignments: Vec<usize> = sample
            .par_iter()
            .map(|&i| nearest(row(i), &centroids, k).0)
            .collect();
        let changes = assignments.iter().zip(&new_assignments).filter(|(a, b)| a != b).count();
        assignments = new_assignments;

        let mut new_centroids = vec![0.0f32; k * DIMS];
        let mut counts = vec![0usize; k];
        for (si, &c) in assignments.iter().enumerate() {
            let cr = &mut new_centroids[c * DIMS..(c + 1) * DIMS];
            for (a, &v) in cr.iter_mut().zip(row(sample[si]).iter()) { *a += v; }
            counts[c] += 1;
        }
        for c in 0..k {
            let cr = &mut new_centroids[c * DIMS..(c + 1) * DIMS];
            if counts[c] > 0 {
                let norm: f32 = cr.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 { cr.iter_mut().for_each(|x| *x /= norm); }
            } else {
                let idx = sample[(rng.next() % sample_n as u64) as usize];
                cr.copy_from_slice(row(idx));
            }
        }
        centroids = new_centroids;

        let pct = 100.0 * changes as f32 / sample_n as f32;
        println!("  Iter {:2}: {:.2}% changed ({:.1}s)", iter + 1, pct, t_iter.elapsed().as_secs_f32());
        if pct < 0.5 { println!("  Converged."); break; }
    }

    // ── Assign every valid vector once, keep the closest members per cluster ──
    println!("\nAssigning all {} vectors ...", valid.len());
    let full: Vec<(usize, f32)> = valid
        .par_iter()
        .map(|&i| nearest(row(i), &centroids, k))
        .collect();

    let mut members: Vec<Vec<(f32, u32)>> = vec![Vec::new(); k];
    for (vi, &(c, sim)) in full.iter().enumerate() {
        members[c].push((sim, ids[valid[vi]]));
    }
    for m in members.iter_mut() {
        m.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        m.truncate(args.max_members);
    }

    // Sort clusters by size (before truncation would be ideal, truncated size is fine)
    let mut order: Vec<usize> = (0..k).filter(|&c| !members[c].is_empty()).collect();
    order.sort_unstable_by(|&a, &b| members[b].len().cmp(&members[a].len()));

    // ── Serialise cluster_seeds.bin ───────────────────────────────────────────
    let n_clusters = order.len();
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&(n_clusters as u32).to_le_bytes());
    out.extend_from_slice(&(DIMS as u32).to_le_bytes());
    for &c in &order {
        for &v in &centroids[c * DIMS..(c + 1) * DIMS] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(members[c].len() as u32).to_le_bytes());
        for &(_, id) in &members[c] {
            out.extend_from_slice(&id.to_le_bytes());
        }
    }

    let out_path = args.out_dir.join("cluster_seeds_v2.bin");
    fs::write(&out_path, &out)?;
    println!(
        "\nWritten {} clusters → {} ({:.1} MB, {:.0}s total)",
        n_clusters, out_path.display(),
        out.len() as f64 / 1e6,
        t0.elapsed().as_secs_f32()
    );

    Ok(())
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

struct Lcg64(u64);
impl Lcg64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
}
