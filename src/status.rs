//! What the engine knows about its own data, in words the data pipeline can act on: do the catalog (the view `catalog`) and the vectors file describe the same titles?
//!
//! The engine can only show a title that is in the catalog and can only place it in a row when it has a vector. A catalog title without a vector is lost to every row; a vector
//! of a title the catalog does not show costs memory and index time and makes the nearest neighbour search look past it. The pipeline's export writes vectors for exactly the
//! catalog titles, so after a catalog rule change both numbers say whether the export was run (`pipeline/export_engine_files.py`).
use crate::catalog::CatalogMap;
use crate::vectors::VectorStore;
use serde_json::{json, Value};

pub struct Coverage {
    pub titles: usize,
    pub vectors: usize,
    pub titles_with_vector: usize,
}

impl Coverage {
    pub fn of(catalog: &CatalogMap, vectors: &VectorStore) -> Self {
        let titles_with_vector = catalog.values().filter(|i| vectors.index_of(i.id).is_some()).count();
        Self { titles: catalog.len(), vectors: vectors.n_vecs, titles_with_vector }
    }

    pub fn titles_without_vector(&self) -> usize {
        self.titles - self.titles_with_vector
    }

    pub fn vectors_outside_catalog(&self) -> usize {
        self.vectors.saturating_sub(self.titles_with_vector)
    }

    /// What to do about it, when something should be done.
    pub fn advice(&self) -> Option<String> {
        let missing = self.titles_without_vector();
        if missing * 20 > self.titles {
            return Some(format!(
                "{missing} of {} catalog titles have no vector and can never appear in a row: run the export (pipeline/export_engine_files.py) after the catalog rule changed",
                self.titles
            ));
        }
        let extra = self.vectors_outside_catalog();
        if extra * 4 > self.vectors {
            return Some(format!(
                "{extra} of {} vectors belong to titles the catalog does not show: run the export (pipeline/export_engine_files.py) to trim them (less memory, a faster index)",
                self.vectors
            ));
        }
        None
    }

    pub fn to_json(&self) -> Value {
        json!({
            "titles": self.titles,
            "vectors": self.vectors,
            "titles_without_vector": self.titles_without_vector(),
            "vectors_outside_catalog": self.vectors_outside_catalog(),
            "advice": self.advice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cov(titles: usize, vectors: usize, with: usize) -> Coverage {
        Coverage { titles, vectors, titles_with_vector: with }
    }

    #[test]
    fn matching_data_needs_no_advice() {
        assert!(cov(60_054, 60_054, 60_054).advice().is_none());
    }

    #[test]
    fn many_titles_without_a_vector_say_to_run_the_export() {
        let a = cov(1000, 900, 900).advice().unwrap();
        assert!(a.contains("100 of 1000 catalog titles have no vector"), "{a}");
    }

    #[test]
    fn many_unused_vectors_say_to_trim_them() {
        let a = cov(60_054, 626_376, 60_054).advice().unwrap();
        assert!(a.contains("566322 of 626376 vectors"), "{a}");
        // a few unused vectors are normal
        assert!(cov(1000, 1100, 1000).advice().is_none());
    }
}
