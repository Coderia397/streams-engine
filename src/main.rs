use streams_engine::app::{build_app_state, Config};
use streams_engine::{page, routes};
use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Tracing: RUST_LOG=streams_engine=info,tower_http=debug
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
        .route("/api/genre-support",   get(routes::genres::genre_support))
        .route("/api/language-support", get(routes::genres::language_support))
        .route("/api/new-popular",     get(routes::new_popular::new_popular))
        .route("/api/rating",          get(routes::rating::rating))
        .route("/api/ratings",         get(routes::rating::ratings))
        .route("/api/health",          get(routes::health::health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("streams-engine listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
