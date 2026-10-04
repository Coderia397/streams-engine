mod app;
mod catalog;
mod genre_spec;
mod centroid;
mod cluster;
mod hnsw;
mod mmr;
mod row_label;
mod mood;
mod page;
mod quality;
mod routes;
mod sessions;
mod support;
mod taxonomy;
mod titles;
mod vectors;

use app::{build_app_state, Config};
use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Tracing: RUST_LOG=pstream_engine=info,tower_http=debug
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_env("RUST_LOG"))
        .init();

    let config = Config::from_env();
    // ENGINE_SAFETY_NETS=off turns off the rules that hide data defects (tile art, news and talk shows, off season titles)
    // so a baseline can be measured without them.
    if std::env::var("ENGINE_SAFETY_NETS").map(|v| v.eq_ignore_ascii_case("off")).unwrap_or(false) {
        page::SAFETY_NETS.store(false, std::sync::atomic::Ordering::Relaxed);
        info!("safety nets OFF");
    }
    let port = config.port;
    let state = build_app_state(&config).await?;

    let app = Router::new()
        .route("/api/feed",            get(routes::feed::feed))
        .route("/api/hero-feed",       get(routes::hero::hero_feed))
        .route("/api/search",          get(routes::search::search))
        .route("/api/recommendations", get(routes::recommendations::recommendations))
        .route("/api/feedback",        post(routes::feedback::feedback))
        .route("/api/health",          get(routes::health::health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("pstream-engine listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
