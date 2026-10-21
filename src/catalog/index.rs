//! The loaded catalog: titles by key, plus the lists a page reads (chart keys, attribute keys, topics).
use super::*;

/// The titles of the catalog by key, plus the keys of the titles that can be on a Top 10 chart (`page::chart_ok`), most popular first (made once at load,
/// after the popularity overlay, so a chart reads its first ten matching titles instead of scanning the catalog).
pub struct Catalog {
    items: HashMap<u32, CatalogItem>,
    pub chart_keys: Vec<u32>,
    /// Per attribute row kind (`attributes::SPECS`) the quality titles that have the attribute, best quality first
    pub attr_keys: HashMap<&'static str, Vec<u32>>,
    /// Quality titles that few have found yet (`personal::is_gem`), best quality first
    pub gem_keys: Vec<u32>,
    /// Per topic slug (`topics_v1.json`) the titles the judge accepted, best quality first
    pub topic_keys: HashMap<String, Vec<u32>>,
}

impl Catalog {
    pub fn new(items: HashMap<u32, CatalogItem>) -> Self {
        let mut chart: Vec<&CatalogItem> = items.values().filter(|i| crate::page::chart_ok(i)).collect();
        chart.sort_unstable_by(|a, b| b.popularity.total_cmp(&a.popularity).then_with(|| a.id.cmp(&b.id)));
        let chart_keys = chart.into_iter().map(|i| i.id).collect();
        let mut attr_keys: HashMap<&'static str, Vec<u32>> = HashMap::default();
        for spec in crate::attributes::SPECS {
            let mut v: Vec<&CatalogItem> = items.values().filter(|i| crate::page::passes_member_quality(i) && (spec.test)(i)).collect();
            crate::quality::sort_by_quality(&mut v);
            attr_keys.insert(spec.id, v.into_iter().map(|i| i.id).collect());
        }
        let mut gems: Vec<&CatalogItem> = items.values().filter(|i| crate::page::passes_member_quality(i) && crate::personal::is_gem(i)).collect();
        crate::quality::sort_by_quality(&mut gems);
        let gem_keys = gems.into_iter().map(|i| i.id).collect();
        let mut by_topic: HashMap<String, Vec<&CatalogItem>> = HashMap::default();
        for i in items.values() {
            for k in &i.keywords {
                if let Some(slug) = k.strip_prefix("topic:") { by_topic.entry(slug.to_string()).or_default().push(i); }
            }
        }
        let topic_keys = by_topic.into_iter().map(|(slug, mut v)| { crate::quality::sort_by_quality(&mut v); (slug, v.into_iter().map(|i| i.id).collect()) }).collect();
        Self { items, chart_keys, attr_keys, gem_keys, topic_keys }
    }
}

impl std::ops::Deref for Catalog {
    type Target = HashMap<u32, CatalogItem>;
    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

pub type CatalogMap = Catalog;

pub type SharedCatalog = Arc<ArcSwap<CatalogMap>>;
