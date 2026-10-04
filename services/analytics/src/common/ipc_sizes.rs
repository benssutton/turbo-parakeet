// ─────────────────────────────────────────────────────────────────────────────
// sizes — Arrow IPC body sizes (Describe group E; recast sizes for recommend.rs)
// ─────────────────────────────────────────────────────────────────────────────
//
// Mirrors pyarrow's IPC writer (checked against pyarrow 24; the Python oracle is
// analytics/describe/_sizes.py): a column's size is the body length of the IPC
// messages that carry it (dictionary batches + record batch). Every buffer is
// padded to 8 bytes; a validity buffer is written only when the array has nulls;
// an empty buffer takes no space; List/Utf8 offsets are rebased to 0 and their
// child/values sliced to the referenced range. With ZSTD each non-empty buffer is
// an 8-byte uncompressed-length prefix plus one ZSTD frame (content size
// included) — pyarrow never falls back to raw bytes. arrow-rs's own IPC writer is
// not used because it does not expose the ZSTD level.
//
// Works on arrow-rs ArrayData; Series arrive through arrow_io::export_series. Arrow
// sizes use CompatLevel::oldest() (LargeUtf8, LargeList); Polars sizes the plain
// and ZSTD body of CompatLevel::newest() (view types).

use crate::common::arrow_io::export_series;
use arrow_array::Array;
use arrow_buffer::{ArrowNativeType, BooleanBuffer, ToByteSlice};
use arrow_data::ArrayData;
use arrow_schema::DataType as AT;
use polars::prelude::*;
use rayon::prelude::*;

struct Body {
    level: Option<i32>,
    bytes: u64,
}

impl Body {
    fn buffer(&mut self, data: &[u8]) -> PolarsResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let len = match self.level {
            None => data.len(),
            Some(level) => {
                8 + zstd::bulk::compress(data, level)
                    .map_err(|e| polars_err!(ComputeError: "zstd: {e}"))?
                    .len()
            }
        };
        self.bytes += len.next_multiple_of(8) as u64;
        Ok(())
    }

    fn bits(&mut self, b: &BooleanBuffer) -> PolarsResult<()> {
        let packed = b.sliced(); // re-aligned to bit 0, as pyarrow writes it
        self.buffer(&packed.as_slice()[..b.len().div_ceil(8)])
    }

    fn validity(&mut self, d: &ArrayData) -> PolarsResult<()> {
        match d.nulls() {
            Some(n) if n.null_count() > 0 => self.bits(n.inner()),
            _ => Ok(()),
        }
    }

    fn fixed(&mut self, d: &ArrayData, width: usize) -> PolarsResult<()> {
        self.validity(d)?;
        let start = d.offset() * width;
        self.buffer(&d.buffers()[0].as_slice()[start..start + d.len() * width])
    }

    /// Writes the offsets (rebased to 0); returns the referenced child/values range.
    fn offsets<O: ArrowNativeType>(&mut self, d: &ArrayData) -> PolarsResult<(usize, usize)> {
        if d.is_empty() {
            self.buffer(&vec![0u8; std::mem::size_of::<O>()])?; // pyarrow writes the single offset [0]
            return Ok((0, 0));
        }
        let o = &d.buffers()[0].typed_data::<O>()[d.offset()..=d.offset() + d.len()];
        let (first, last) = (o[0].as_usize(), o[d.len()].as_usize());
        if first == 0 {
            self.buffer(o.to_byte_slice())?;
        } else {
            let rebased: Vec<O> = o
                .iter()
                .map(|x| O::usize_as(x.as_usize() - first))
                .collect();
            self.buffer(rebased.as_slice().to_byte_slice())?;
        }
        Ok((first, last))
    }

    fn var_size<O: ArrowNativeType>(&mut self, d: &ArrayData) -> PolarsResult<()> {
        self.validity(d)?;
        let (first, last) = self.offsets::<O>(d)?;
        self.buffer(&d.buffers()[1].as_slice()[first..last])
    }

    fn list<O: ArrowNativeType>(&mut self, d: &ArrayData) -> PolarsResult<()> {
        self.validity(d)?;
        let (first, last) = self.offsets::<O>(d)?;
        self.array(&d.child_data()[0].slice(first, last - first))
    }

    fn array(&mut self, d: &ArrayData) -> PolarsResult<()> {
        match d.data_type() {
            AT::Null => Ok(()),
            AT::Boolean => {
                self.validity(d)?;
                self.bits(&BooleanBuffer::new(
                    d.buffers()[0].clone(),
                    d.offset(),
                    d.len(),
                ))
            }
            AT::Utf8 | AT::Binary => self.var_size::<i32>(d),
            AT::LargeUtf8 | AT::LargeBinary => self.var_size::<i64>(d),
            AT::Utf8View | AT::BinaryView => {
                self.validity(d)?;
                // Only the 16-byte view buffer is offset/len-sliced; the variadic data
                // buffers (buffers()[1..]) hold the actual string/byte payload and are
                // written whole even for a sliced view array — pyarrow does the same,
                // since a view's inline/prefix bytes and buffer-index+offset already
                // point at the right bytes regardless of which views are in range.
                self.buffer(
                    &d.buffers()[0].as_slice()[d.offset() * 16..(d.offset() + d.len()) * 16],
                )?;
                d.buffers()[1..]
                    .iter()
                    .try_for_each(|b| self.buffer(b.as_slice()))
            }
            AT::List(_) => self.list::<i32>(d),
            AT::LargeList(_) => self.list::<i64>(d),
            // FixedSizeList/Struct children are sliced by the parent's own offset/len here:
            // needed for FFI-imported or hand-built ArrayData, where a parent can carry a
            // non-zero offset over unsliced children. arrow-rs's own `Array::slice().to_data()`
            // instead returns offset 0 with children already sliced, so this slice is a no-op
            // in that case and never double-applies.
            AT::FixedSizeList(_, w) => {
                self.validity(d)?;
                let w = *w as usize;
                self.array(&d.child_data()[0].slice(d.offset() * w, d.len() * w))
            }
            AT::Struct(_) => {
                self.validity(d)?;
                d.child_data()
                    .iter()
                    .try_for_each(|c| self.array(&c.slice(d.offset(), d.len())))
            }
            AT::Dictionary(k, _) => {
                let width = k.primitive_width().ok_or_else(
                    || polars_err!(ComputeError: "sizes: non-integer dictionary key {k}"),
                )?;
                self.fixed(d, width)?; // the keys
                self.array(&d.child_data()[0]) // the dictionary batch
            }
            t => match t.primitive_width() {
                Some(w) => self.fixed(d, w),
                None => polars_bail!(ComputeError: "sizes: unsupported Arrow type {t}"),
            },
        }
    }
}

/// Arrow IPC body bytes of `arr`: uncompressed (`level` None) or ZSTD at `level`.
pub(crate) fn ipc_body_bytes(arr: &dyn Array, level: Option<i32>) -> PolarsResult<u64> {
    let mut body = Body { level, bytes: 0 };
    body.array(&arr.to_data())?;
    Ok(body.bytes)
}

pub(crate) type Sizes = [u64; 4];
pub(crate) const SIZE_FIELDS: [&str; 4] = [
    "size_bytes",
    "size_zstd_bytes",
    "size_polars_bytes",
    "size_polars_zstd_bytes",
];

/// `s`'s classic layout (CompatLevel::oldest).
pub(crate) fn classic_layout(s: &Series) -> PolarsResult<arrow_array::ArrayRef> {
    export_series(s, CompatLevel::oldest())
}

pub(crate) fn sizes(s: &Series, level: i32) -> PolarsResult<Sizes> {
    sizes_of(s, &classic_layout(s)?, level)
}

/// `sizes` given `s`'s `classic_layout` (exported once by callers that reuse it).
pub(crate) fn sizes_of(
    s: &Series,
    classic: &arrow_array::ArrayRef,
    level: i32,
) -> PolarsResult<Sizes> {
    let native = export_series(s, CompatLevel::newest())?;
    Ok([
        ipc_body_bytes(classic.as_ref(), None)?,
        ipc_body_bytes(classic.as_ref(), Some(level))?,
        ipc_body_bytes(native.as_ref(), None)?,
        ipc_body_bytes(native.as_ref(), Some(level))?,
    ])
}

pub(crate) fn column_sizes_impl(inputs: &[Series], level: i32) -> PolarsResult<Series> {
    let rows: Vec<Sizes> = inputs
        .par_iter()
        .map(|s| sizes(s, level))
        .collect::<PolarsResult<_>>()?;
    let mut columns = vec![
        StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
            .into_series()
            .with_name("column".into()),
    ];
    for (j, name) in SIZE_FIELDS.iter().enumerate() {
        columns.push(
            UInt64Chunked::from_iter_values((*name).into(), rows.iter().map(|r| r[j]))
                .into_series(),
        );
    }
    Ok(
        StructChunked::from_series("column_sizes".into(), inputs.len(), columns.iter())?
            .into_series(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrow(s: &Series) -> arrow_array::ArrayRef {
        crate::common::arrow_io::export_series(s, CompatLevel::oldest()).unwrap()
    }

    #[test]
    fn primitive_framing_matches_pyarrow() {
        let s = Series::new("x".into(), (0..1_000).collect::<Vec<i32>>());
        assert_eq!(ipc_body_bytes(arrow(&s).as_ref(), None).unwrap(), 4_000);
        let zstd = ipc_body_bytes(arrow(&s).as_ref(), Some(1)).unwrap();
        assert!((1_890..=1_935).contains(&zstd), "{zstd}"); // pyarrow: 1912
        let one = Series::new("x".into(), &[1i32]);
        assert_eq!(ipc_body_bytes(arrow(&one).as_ref(), Some(1)).unwrap(), 24);
    }

    #[test]
    fn validity_only_with_nulls() {
        let s = Series::new(
            "x".into(),
            (0..1_000)
                .map(|i| (i % 3 != 0).then_some(i))
                .collect::<Vec<Option<i32>>>(),
        );
        assert_eq!(ipc_body_bytes(arrow(&s).as_ref(), None).unwrap(), 4_128);
        assert_eq!(
            ipc_body_bytes(
                arrow(&Series::new_empty("x".into(), &DataType::Int32)).as_ref(),
                None
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn large_utf8_offsets_values_validity() {
        assert_eq!(
            ipc_body_bytes(
                arrow(&Series::new("x".into(), &[Some("ab"), None])).as_ref(),
                None
            )
            .unwrap(),
            40
        );
        assert_eq!(
            ipc_body_bytes(
                arrow(&Series::new_empty("x".into(), &DataType::String)).as_ref(),
                None
            )
            .unwrap(),
            8
        );
    }

    #[test]
    fn polars_size_is_native_ipc_body() {
        assert_eq!(
            sizes(&Series::new("x".into(), &[Some("ab"), None]), 1).unwrap()[2],
            40
        );
    }

    #[test]
    fn dictionary_keys_and_values() {
        use polars::datatypes::Categories;
        let cats = Categories::global();
        let cat = Series::new("x".into(), &["a", "b", "a"])
            .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
            .unwrap();
        assert_eq!(sizes(&cat, 1).unwrap()[0], 48); // pyarrow: keys 16 + dictionary 32 (see test_describe.py)
    }
}
