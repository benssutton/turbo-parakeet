//! The recommenders' output columns (spec 2026-10-04 §4).

use crate::common::ipc_sizes::SIZE_FIELDS;
use crate::techniques::describe::value_fields;
use polars::prelude::{DataType as PT, Field as PField};

pub(crate) fn candidate_type() -> PT {
    PT::Struct(vec![
        PField::new("arrow_type".into(), PT::String),
        PField::new("rule".into(), PT::String),
        PField::new("evidence".into(), PT::String),
        PField::new("predicted_bytes".into(), PT::UInt64),
        PField::new("projected_population_bytes".into(), PT::Float64),
        PField::new("outcome".into(), PT::String),
        PField::new("reason".into(), PT::String),
    ])
}

pub(crate) fn rec_fields() -> Vec<(String, PT)> {
    [
        ("rec_nullable", PT::Boolean),
        ("rec_arrow_type", PT::String),
        ("rec_arrow_size_bytes", PT::UInt64),
        ("rec_arrow_size_zstd_bytes", PT::UInt64),
        ("rec_polars_type", PT::String),
        ("rec_polars_size_bytes", PT::UInt64),
        ("rec_polars_size_zstd_bytes", PT::UInt64),
        ("rec_lossy_formatting", PT::Boolean),
        ("rec_candidates", PT::List(Box::new(candidate_type()))),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect()
}

/// The recommenders' output columns (spec 2026-10-04 §4), one definition for both:
/// `streaming` adds `first_row` after `dtype` and the sample counts at the end.
pub(crate) fn recommender_fields(streaming: bool) -> Vec<(String, PT)> {
    let mut f: Vec<(String, PT)> = vec![
        ("column".into(), PT::String),
        ("status".into(), PT::String),
        ("dtype".into(), PT::String),
    ];
    if streaming {
        f.push(("first_row".into(), PT::UInt64));
    }
    f.push(("n_rows".into(), PT::UInt64));
    f.push(("n_null".into(), PT::UInt64));
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), PT::UInt64));
    f.extend(SIZE_FIELDS.iter().map(|n| (n.to_string(), PT::UInt64)));
    f.push(("inner_n_values".into(), PT::UInt64));
    f.push(("inner_n_null".into(), PT::UInt64));
    f.extend(
        value_fields()
            .into_iter()
            .map(|(n, d)| (format!("inner_{n}"), d)),
    );
    f.extend(rec_fields());
    if streaming {
        f.push(("n_sampled_rows".into(), PT::UInt64));
        f.push(("n_sampled_blocks".into(), PT::UInt64));
    }
    f
}
