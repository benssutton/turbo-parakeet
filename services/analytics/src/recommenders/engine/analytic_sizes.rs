//! Analytic IPC body sizes of a candidate type (spec 2026-09-26 §5.1, §5.3).

use super::*;
use arrow_schema::DataType as AT;

// ── analytic sizes (Spec B §5.1, §5.3) ──────────────────────────────────────

/// What a predicted size depends on: rows, nulls, value bytes, distinct values and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Shape {
    pub n: f64,
    pub nulls: f64,
    pub sum_len: f64,
    pub d: f64,
    pub sum_len_unique: f64,
}

impl Shape {
    /// The same frame with a dictionary of `c` values of the observed mean length.
    pub(crate) fn project(&self, c: f64) -> Shape {
        let per_value = if self.d > 0.0 {
            self.sum_len_unique / self.d
        } else {
            0.0
        };
        Shape {
            n: self.n,
            nulls: self.nulls,
            sum_len: self.sum_len,
            d: c,
            sum_len_unique: per_value * c,
        }
    }
}

pub(crate) fn pad(x: f64) -> f64 {
    (x / 8.0).ceil() * 8.0
}

pub(crate) fn validity(n: f64, nulls: f64) -> f64 {
    if nulls > 0.0 {
        pad((n / 8.0).ceil())
    } else {
        0.0
    }
}

/// Uncompressed Arrow IPC body bytes of a scalar type `t` holding values shaped
/// like `s`, exactly as common/ipc_sizes.rs measures it. Lists are sized by the caller; a type
/// with no analytic size is an error (its candidate fails), never a wrong size.
pub(crate) fn body_size(t: &AT, s: &Shape) -> Result<f64, String> {
    let v = validity(s.n, s.nulls);
    let width = |t: &AT| {
        t.primitive_width()
            .ok_or_else(|| format!("no predicted size for {}", pa_name(t)))
    };
    Ok(match t {
        AT::Null => 0.0,
        AT::Boolean => v + pad((s.n / 8.0).ceil()),
        AT::Utf8 | AT::Binary => v + pad(4.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::LargeUtf8 | AT::LargeBinary => v + pad(8.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::Dictionary(k, values) if **values == AT::Utf8 => {
            v + pad(s.n * width(k)? as f64) + pad(4.0 * (s.d + 1.0)) + pad(s.sum_len_unique)
        }
        AT::Struct(f) if matches!(f.first().map(|x| x.data_type()), Some(AT::Timestamp(u, _)) if *t == timestamp_with_offset(*u)) => {
            v + pad(8.0 * s.n) + pad(2.0 * s.n)
        }
        t => v + pad(s.n * width(t)? as f64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ipc_sizes::ipc_body_bytes;
    use arrow_array::ArrayRef;
    use arrow_array::{
        builder::StringDictionaryBuilder, types::UInt8Type, DictionaryArray, StringArray,
    };
    use arrow_schema::{Field as AField, Fields, TimeUnit};
    use std::sync::Arc;

    #[test]
    fn predicted_sizes_match_sizes_rs() {
        let s = StringArray::from(vec![Some("ab"), None]);
        let shape = Shape {
            n: 2.0,
            nulls: 1.0,
            sum_len: 2.0,
            d: 1.0,
            sum_len_unique: 2.0,
        };
        assert_eq!(
            body_size(&AT::Utf8, &shape),
            Ok(ipc_body_bytes(&s, None).unwrap() as f64)
        );
        let mut builder = StringDictionaryBuilder::<UInt8Type>::new();
        for v in ["a", "b", "a"] {
            builder.append_value(v);
        }
        let d: DictionaryArray<UInt8Type> = builder.finish();
        let shape = Shape {
            n: 3.0,
            nulls: 0.0,
            sum_len: 3.0,
            d: 2.0,
            sum_len_unique: 2.0,
        };
        assert_eq!(
            body_size(
                &Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type(),
                &shape
            ),
            Ok(ipc_body_bytes(&d, None).unwrap() as f64)
        );
    }

    #[test]
    fn unsized_types_are_errors_not_panics() {
        let shape = Shape {
            n: 3.0,
            ..Default::default()
        };
        assert!(body_size(&AT::Utf8View, &shape).is_err());
        assert!(body_size(
            &Target::List(Box::new(Target::Fixed(AT::UInt8))).arrow_type(),
            &shape
        )
        .is_err());
        assert!(body_size(
            &AT::Struct(Fields::from(vec![AField::new("a", AT::Int8, true)])),
            &shape
        )
        .is_err());
        assert!(body_size(
            &AT::Dictionary(Box::new(AT::Utf8), Box::new(AT::Utf8)),
            &shape
        )
        .is_err());
        assert_eq!(
            body_size(&timestamp_with_offset(TimeUnit::Millisecond), &shape),
            Ok(24.0 + 8.0)
        );
        let c = candidate(
            Target::Fixed(AT::Utf8View),
            Rank::Plain,
            "r",
            String::new(),
            Err("no size".into()),
        );
        assert_eq!(
            (c.outcome, c.reason.as_deref()),
            (Outcome::Failed, Some("no size"))
        );
        let a: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1]));
        assert!(list_parts(&a).is_err());
    }
}
