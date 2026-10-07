//! Top-k value frequencies (spec docs/superpowers/specs/2026-10-07-top-k-frequencies-design.md):
//! the frequency-ordered dictionary.

use super::*;
use crate::techniques::describe::Ranking;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowDictionaryKeyType;
use arrow_array::{
    downcast_dictionary_array, Array, ArrayRef, DictionaryArray, PrimitiveArray, UInt32Array,
};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType as AT;
use std::collections::HashMap;
use std::sync::Arc;

/// `a` (a dictionary) with its keys renumbered by `ranking`: the most frequent value takes
/// key 0, the next key 1, and so on (spec §5). Values the ranking lacks (only those behind
/// null lists, which Describe's flatten skips) follow every ranked value in their existing
/// order. The decoded values and the key type are unchanged; any other array, or one already
/// in order, is returned as is.
pub(crate) fn frequency_order(a: &ArrayRef, ranking: &Ranking) -> Result<ArrayRef, String> {
    let AT::Dictionary(..) = a.data_type() else {
        return Ok(a.clone());
    };
    let d = a.as_any_dictionary();
    let text = arrow_cast(d.values().as_ref(), &AT::Utf8)?;
    let text = text.as_string::<i32>();
    let rank: HashMap<&str, usize> = ranking
        .iter()
        .enumerate()
        .map(|(i, (v, _))| (v.as_str(), i))
        .collect();
    // order[new code] = old code.
    let mut order: Vec<u32> = (0..text.len() as u32).collect();
    order.sort_by_key(|&c| {
        let r = rank.get(text.value(c as usize)).copied();
        (r.unwrap_or(usize::MAX), c)
    });
    if order
        .iter()
        .enumerate()
        .all(|(new, &old)| new as u32 == old)
    {
        return Ok(a.clone());
    }
    let array = a.as_ref();
    downcast_dictionary_array!(
        array => reorder(array, &order),
        _ => unreachable!("checked above")
    )
}

/// `d` with its values taken in `order` (new code → old code) and its keys remapped in one
/// pass, in the dictionary's own key type.
fn reorder<K: ArrowDictionaryKeyType>(
    d: &DictionaryArray<K>,
    order: &[u32],
) -> Result<ArrayRef, String> {
    let mut perm = vec![K::Native::default(); order.len()];
    for (new, &old) in order.iter().enumerate() {
        perm[old as usize] = K::Native::from_usize(new).ok_or("dictionary key overflow")?;
    }
    // Not `take`: null slots must hold zero, as ZSTD sizes depend on those bytes.
    let keys: PrimitiveArray<K> = d
        .keys()
        .iter()
        .map(|k| k.map(|k| perm[k.as_usize()]))
        .collect();
    let values = arrow_select::take::take(
        d.values().as_ref(),
        &UInt32Array::from(order.to_vec()),
        None,
    )
    .map_err(|e| e.to_string())?;
    let dict = DictionaryArray::<K>::try_new(keys, values).map_err(|e| e.to_string())?;
    Ok(Arc::new(dict))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt8Type;
    use arrow_array::{ArrayRef, StringArray};
    use arrow_schema::DataType as AT;
    use std::sync::Arc;

    fn dict(values: &[Option<&str>]) -> ArrayRef {
        let s: ArrayRef = Arc::new(StringArray::from(values.to_vec()));
        arrow_cast(
            s.as_ref(),
            &AT::Dictionary(Box::new(AT::UInt8), Box::new(AT::Utf8)),
        )
        .unwrap()
    }

    fn ranking(v: &[(&str, u64)]) -> Ranking {
        v.iter().map(|&(s, n)| (s.to_string(), n)).collect()
    }

    fn decoded(a: &ArrayRef) -> Vec<Option<String>> {
        let t = arrow_cast(a.as_ref(), &AT::Utf8).unwrap();
        t.as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    }

    fn texts(v: &[&str]) -> Vec<Option<String>> {
        v.iter().map(|s| Some(s.to_string())).collect()
    }

    #[test]
    fn the_most_frequent_value_takes_key_zero() {
        let a = dict(&[
            Some("a"),
            Some("b"),
            None,
            Some("b"),
            Some("c"),
            Some("b"),
            Some("c"),
        ]);
        let out = frequency_order(&a, &ranking(&[("b", 3), ("c", 2), ("a", 1)])).unwrap();
        assert_eq!(out.data_type(), a.data_type());
        assert_eq!(decoded(&out), decoded(&a));
        let d = out.as_dictionary::<UInt8Type>();
        assert_eq!(decoded(d.values()), texts(&["b", "c", "a"]));
        assert_eq!(
            d.keys().iter().collect::<Vec<_>>(),
            [Some(2), Some(0), None, Some(0), Some(1), Some(0), Some(1)]
        );
    }

    #[test]
    fn wider_keys_keep_their_type() {
        let names: Vec<String> = (0..300).map(|i| format!("v{i}")).collect();
        let mut v: Vec<Option<&str>> = names.iter().map(|s| Some(s.as_str())).collect();
        v.extend([Some("v299"); 5]);
        let s: ArrayRef = Arc::new(StringArray::from(v));
        let t = AT::Dictionary(Box::new(AT::UInt16), Box::new(AT::Utf8));
        let a = arrow_cast(s.as_ref(), &t).unwrap();
        let out = frequency_order(&a, &ranking(&[("v299", 6)])).unwrap();
        assert_eq!(out.data_type(), &t);
        assert_eq!(decoded(&out), decoded(&a));
        assert_eq!(
            decoded(out.as_any_dictionary().values())[0].as_deref(),
            Some("v299")
        );
    }

    #[test]
    fn unranked_values_follow_the_ranked_ones() {
        let a = dict(&[Some("x"), Some("y"), Some("z")]);
        let out = frequency_order(&a, &ranking(&[("z", 5)])).unwrap();
        let d = out.as_dictionary::<UInt8Type>();
        assert_eq!(decoded(d.values()), texts(&["z", "x", "y"]));
        assert_eq!(decoded(&out), decoded(&a));
    }

    #[test]
    fn ordered_dictionaries_and_other_arrays_are_unchanged() {
        let a = dict(&[Some("x"), Some("x"), Some("y")]);
        let out = frequency_order(&a, &ranking(&[("x", 2), ("y", 1)])).unwrap();
        assert!(Arc::ptr_eq(&a, &out));
        let plain: ArrayRef = Arc::new(StringArray::from(vec!["x"]));
        assert!(Arc::ptr_eq(
            &plain,
            &frequency_order(&plain, &ranking(&[])).unwrap()
        ));
    }

    #[test]
    fn the_chosen_dictionary_is_frequency_ordered() {
        use crate::common::ipc_sizes::{classic_layout, ipc_body_bytes};
        use crate::recommenders::engine::choose::tests::{describe_one, params};
        use polars::prelude::{NamedFrom, Series};
        // "alpha" is seen first; "beta" is twice as frequent.
        let v: Vec<&str> = (0..300)
            .map(|i| if i % 3 == 0 { "alpha" } else { "beta" })
            .collect();
        let s = Series::new("x".into(), &v);
        let d = describe_one(&s, 0).unwrap();
        let values = classic_layout(&s).unwrap();
        let size = ipc_body_bytes(values.as_ref(), None).unwrap();
        let est = d.conclusions(10_000).0.est;
        let lvl = Level::of_values(s.dtype(), values, &d.outer, None, size, est, "").unwrap();
        let c = choose(&lvl, &params()).unwrap();
        assert!(
            matches!(c.target, Target::Dictionary(..)),
            "{}",
            pa_name(c.array.data_type())
        );
        assert_eq!(
            decoded(c.array.as_any_dictionary().values()),
            texts(&["beta", "alpha"])
        );
        assert!(
            c.candidates
                .iter()
                .any(|x| x.rule == "string→dictionary"
                    && x.evidence.ends_with(" key_order=frequency"))
        );
    }
}
