//! Which columns a pairwise / threeway kernel runs on: kwargs and name resolution.

use polars::prelude::*;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Kwargs structs
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) struct PairwiseKwargs {
    pub pairs: Option<Vec<Vec<String>>>,
}

pub(crate) struct ThreewayKwargs {
    pub triplets: Option<Vec<Vec<String>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pair / triplet resolution from kwargs
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn resolve_pairs(
    raw_pairs: &[Vec<String>],
    name_map: &HashMap<String, usize>,
) -> PolarsResult<Vec<(usize, usize)>> {
    raw_pairs
        .iter()
        .map(|pair| {
            if pair.len() != 2 {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Each pair must have exactly 2 column names, got {}",
                        pair.len()
                    )
                    .into(),
                ));
            }
            let i = *name_map
                .get(&pair[0])
                .ok_or_else(|| PolarsError::ColumnNotFound(pair[0].clone().into()))?;
            let j = *name_map
                .get(&pair[1])
                .ok_or_else(|| PolarsError::ColumnNotFound(pair[1].clone().into()))?;
            Ok((i, j))
        })
        .collect()
}

pub(crate) fn resolve_triplets(
    raw_triplets: &[Vec<String>],
    name_map: &HashMap<String, usize>,
) -> PolarsResult<Vec<(usize, usize, usize)>> {
    raw_triplets
        .iter()
        .map(|triplet| {
            if triplet.len() != 3 {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Each triplet must have exactly 3 column names, got {}",
                        triplet.len()
                    )
                    .into(),
                ));
            }
            let i = *name_map
                .get(&triplet[0])
                .ok_or_else(|| PolarsError::ColumnNotFound(triplet[0].clone().into()))?;
            let j = *name_map
                .get(&triplet[1])
                .ok_or_else(|| PolarsError::ColumnNotFound(triplet[1].clone().into()))?;
            let k = *name_map
                .get(&triplet[2])
                .ok_or_else(|| PolarsError::ColumnNotFound(triplet[2].clone().into()))?;
            Ok((i, j, k))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── resolve_pairs / resolve_triplets ────────────────────────────────────

    #[test]
    fn resolve_pairs_maps_names_to_indices_preserving_order() {
        let name_map: HashMap<String, usize> = [
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let raw = vec![
            vec!["c".to_string(), "a".to_string()],
            vec!["b".to_string(), "b".to_string()],
        ];
        let pairs = resolve_pairs(&raw, &name_map).unwrap();
        assert_eq!(pairs, vec![(2, 0), (1, 1)]);
    }

    #[test]
    fn resolve_pairs_rejects_wrong_arity() {
        let name_map: HashMap<String, usize> = [("a".to_string(), 0)].into_iter().collect();
        assert!(resolve_pairs(&[vec!["a".to_string()]], &name_map).is_err());
        let raw3 = vec![vec!["a".to_string(), "a".to_string(), "a".to_string()]];
        assert!(resolve_pairs(&raw3, &name_map).is_err());
    }

    #[test]
    fn resolve_pairs_rejects_unknown_column_name() {
        let name_map: HashMap<String, usize> = [("a".to_string(), 0)].into_iter().collect();
        let raw = vec![vec!["a".to_string(), "nope".to_string()]];
        assert!(resolve_pairs(&raw, &name_map).is_err());
    }

    #[test]
    fn resolve_pairs_does_not_deduplicate() {
        let name_map: HashMap<String, usize> = [("a".to_string(), 0), ("b".to_string(), 1)]
            .into_iter()
            .collect();
        let raw = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "b".to_string()],
        ];
        let pairs = resolve_pairs(&raw, &name_map).unwrap();
        assert_eq!(pairs, vec![(0, 1), (0, 1)]);
    }

    #[test]
    fn resolve_triplets_maps_names_to_indices_preserving_order() {
        let name_map: HashMap<String, usize> = [
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let raw = vec![vec!["c".to_string(), "a".to_string(), "b".to_string()]];
        let triplets = resolve_triplets(&raw, &name_map).unwrap();
        assert_eq!(triplets, vec![(2, 0, 1)]);
    }

    #[test]
    fn resolve_triplets_rejects_wrong_arity() {
        let name_map: HashMap<String, usize> = [("a".to_string(), 0), ("b".to_string(), 1)]
            .into_iter()
            .collect();
        let raw = vec![vec!["a".to_string(), "b".to_string()]];
        assert!(resolve_triplets(&raw, &name_map).is_err());
    }

    #[test]
    fn resolve_triplets_rejects_unknown_column_name() {
        let name_map: HashMap<String, usize> = [
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let raw = vec![vec!["a".to_string(), "b".to_string(), "nope".to_string()]];
        assert!(resolve_triplets(&raw, &name_map).is_err());
    }
}
