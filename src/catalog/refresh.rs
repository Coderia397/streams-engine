//! Reloading the catalog while the engine runs.
use super::*;

/// Size and modification time of the files the catalog is made from (the database and the daily popularity overlay): a reload is needed only when one changed.
pub(crate) fn source_stamp(db_path: &str) -> [Option<(u64, std::time::SystemTime)>; 2] {
    let stamp = |p: &std::path::Path| std::fs::metadata(p).ok().and_then(|m| Some((m.len(), m.modified().ok()?)));
    let db = std::path::Path::new(db_path);
    [stamp(db), db.parent().and_then(|d| stamp(&d.join("popularity_refresh.db")))]
}

/// Background task: every 5 minutes checks whether the catalog database or the popularity overlay changed and, only then, re-loads the catalog from SQLite
/// and atomically swaps the shared pointer. All readers see a consistent snapshot with zero locking.
pub fn spawn_catalog_refresh(catalog: SharedCatalog, db_path: String) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300));
        interval.tick().await; // skip immediate first tick
        let mut loaded = source_stamp(&db_path);
        loop {
            interval.tick().await;
            let now = source_stamp(&db_path);
            if now == loaded {
                continue;
            }
            match tokio::task::spawn_blocking({
                let p = db_path.clone();
                move || load_catalog(&p)
            })
            .await
            {
                Ok(Ok(new_map)) => {
                    catalog.store(Arc::new(new_map));
                    loaded = now;
                    info!("Catalog refreshed");
                }
                Ok(Err(e)) => warn!("Catalog refresh error: {e}"),
                Err(e)     => warn!("Catalog refresh join error: {e}"),
            }
        }
    });
}
