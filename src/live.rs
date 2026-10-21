//! Small files next to the database that can be edited while the engine runs (the removal list of rows, word aliases for search).
//!
//! A `LiveFile` looks at the modified time of its file on every call, which costs one `stat`, and reads and parses the file only when that time changed. A missing or unreadable
//! file gives the default value, so a deployment without the file behaves as before.
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

pub struct LiveFile<T> {
    name: &'static str,
    parse: fn(&str) -> Option<T>,
    cache: Mutex<Option<(Option<SystemTime>, Arc<T>)>>,
}

impl<T: Default> LiveFile<T> {
    /// `name` is the file name next to the database; `parse` returns none for text that is not valid (the default is used then).
    pub const fn new(name: &'static str, parse: fn(&str) -> Option<T>) -> Self {
        Self { name, parse, cache: Mutex::new(None) }
    }

    /// The current content of the file that sits next to `db_path`.
    pub fn get(&self, db_path: &str) -> Arc<T> {
        let Some(path) = Path::new(db_path).parent().map(|d| d.join(self.name)) else { return Arc::new(T::default()) };
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, value)) = cache.as_ref() {
            if *at == modified {
                return value.clone();
            }
        }
        let value = Arc::new(std::fs::read_to_string(&path).ok().and_then(|t| (self.parse)(&t)).unwrap_or_default());
        *cache = Some((modified, value.clone()));
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn parse_numbers(text: &str) -> Option<Vec<u32>> {
        serde_json::from_str(text).ok()
    }

    #[test]
    fn a_file_is_read_again_only_when_its_time_changes() {
        let dir = std::env::temp_dir().join(format!("live-file-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("catalog.db");
        let path = dir.join("numbers.json");
        let file: LiveFile<Vec<u32>> = LiveFile::new("numbers.json", parse_numbers);
        let db = db.to_str().unwrap();

        // missing file: the default, and the default again on the next call
        assert!(file.get(db).is_empty());
        assert!(file.get(db).is_empty());

        std::fs::write(&path, "[1, 2, 3]").unwrap();
        assert_eq!(*file.get(db), vec![1, 2, 3]);

        // the same time gives the cached value even when the text changed underneath (proves the file is not read on every call)
        let t = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::write(&path, "[4]").unwrap();
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(t).unwrap();
        assert_eq!(*file.get(db), vec![1, 2, 3]);

        // a new time reads the file again
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(t + Duration::from_secs(5)).unwrap();
        assert_eq!(*file.get(db), vec![4]);

        // text that does not parse gives the default
        std::fs::write(&path, "not json").unwrap();
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(t + Duration::from_secs(10)).unwrap();
        assert!(file.get(db).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
