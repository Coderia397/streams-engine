# Streams engine

The recommendation engine of Streams (streams.watch): rows, Top 10, hero and search over a catalog of about 755,000 titles, with
taste vectors, an HNSW index and profile aware filtering. Written in Rust (Axum).

* `cargo run --release` with `CATALOG_DIR` pointing at the data folder (catalog, vectors, index); port from `PORT` (default 8088).
* `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` turn on profile aware requests.
* Every push to `main` runs the tests, builds the Linux binary and deploys it to the Hugging Face Space (`deploy/space/`).

MIT licence.
