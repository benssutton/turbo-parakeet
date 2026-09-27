// ─────────────────────────────────────────────────────────────────────────────
// describe — Arrow IPC body sizes (group E) and the `column_sizes` plugin
// ─────────────────────────────────────────────────────────────────────────────
//
// Mirrors pyarrow's IPC writer (checked against pyarrow 24; the Python oracle is
// analytics/describe/_sizes.py): a column's size is the body length of the IPC
// messages that carry it (dictionary batches + record batch). Every buffer is
// padded to 8 bytes; a validity buffer is written only when the array has nulls;
// an empty buffer takes no space; List/Utf8 offsets are rebased to 0 and their
// child/values sliced to the referenced range. With ZSTD each non-empty buffer is
// an 8-byte uncompressed-length prefix plus one ZSTD frame (content size
// included) — pyarrow never falls back to raw bytes.
//
// Arrow sizes use CompatLevel::oldest() (LargeUtf8, LargeList); Polars sizes are
// the plain and ZSTD body of CompatLevel::newest() (view types).
// Columns nesting Int128 inside List/Array/Struct get null sizes: pyarrow cannot
// import them, so there is no oracle to agree with.

use bytemuck::Pod;
use polars::prelude::*;
use polars_arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeListArray, ListArray, PrimitiveArray,
    StructArray, Utf8Array, Utf8ViewArray, View,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::buffer::Buffer;
use polars_arrow::datatypes::PhysicalType;
use polars_arrow::offset::{Offset, OffsetsBuffer};
use polars_arrow::with_match_primitive_type_full;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;

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
            Some(level) => 8 + zstd::bulk::compress(data, level).map_err(|e| polars_err!(ComputeError: "zstd: {e}"))?.len(),
        };
        self.bytes += len.next_multiple_of(8) as u64;
        Ok(())
    }

    fn bits(&mut self, bm: &Bitmap) -> PolarsResult<()> {
        let (bytes, offset, len) = bm.as_slice();
        if offset == 0 {
            return self.buffer(&bytes[..len.div_ceil(8)]);
        }
        let packed: Bitmap = bm.iter().collect(); // re-align to bit 0, as pyarrow does
        self.bits(&packed)
    }

    fn validity(&mut self, arr: &dyn Array) -> PolarsResult<()> {
        match arr.validity() {
            Some(bm) if arr.null_count() > 0 => self.bits(bm),
            _ => Ok(()),
        }
    }

    fn slice<T: Pod>(&mut self, values: &[T]) -> PolarsResult<()> {
        self.buffer(bytemuck::cast_slice(values))
    }

    /// Writes the offsets (rebased to 0); returns the referenced values range.
    fn offsets<O: Offset + Pod>(&mut self, offsets: &OffsetsBuffer<O>) -> PolarsResult<(usize, usize)> {
        let (first, last) = (offsets.first().to_usize(), offsets.last().to_usize());
        if first == 0 {
            self.slice(offsets.as_slice())?;
        } else {
            let rebased: Vec<O> = offsets.as_slice().iter().map(|o| O::from_as_usize(o.to_usize() - first)).collect();
            self.slice(&rebased)?;
        }
        Ok((first, last))
    }

    fn var_size<O: Offset + Pod>(&mut self, arr: &dyn Array, offsets: &OffsetsBuffer<O>, values: &Buffer<u8>) -> PolarsResult<()> {
        self.validity(arr)?;
        let (first, last) = self.offsets(offsets)?;
        self.buffer(&values.as_slice()[first..last])
    }

    fn views(&mut self, arr: &dyn Array, views: &Buffer<View>, data: &[Buffer<u8>]) -> PolarsResult<()> {
        self.validity(arr)?;
        self.slice(views.as_slice())?;
        data.iter().try_for_each(|b| self.buffer(b.as_slice()))
    }

    fn list<O: Offset + Pod>(&mut self, arr: &dyn Array, a: &ListArray<O>) -> PolarsResult<()> {
        self.validity(arr)?;
        let (first, last) = self.offsets(a.offsets())?;
        self.array(a.values().sliced(first, last - first).as_ref())
    }

    fn array(&mut self, arr: &dyn Array) -> PolarsResult<()> {
        let any = arr.as_any();
        match arr.dtype().to_physical_type() {
            PhysicalType::Null => Ok(()),
            PhysicalType::Boolean => {
                self.validity(arr)?;
                self.bits(any.downcast_ref::<BooleanArray>().unwrap().values())
            }
            PhysicalType::Primitive(p) => with_match_primitive_type_full!(p, |$T| {
                self.validity(arr)?;
                self.slice(any.downcast_ref::<PrimitiveArray<$T>>().unwrap().values().as_slice())
            }),
            PhysicalType::Utf8 => { let a = any.downcast_ref::<Utf8Array<i32>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::LargeUtf8 => { let a = any.downcast_ref::<Utf8Array<i64>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::Binary => { let a = any.downcast_ref::<BinaryArray<i32>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::LargeBinary => { let a = any.downcast_ref::<BinaryArray<i64>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::Utf8View => { let a = any.downcast_ref::<Utf8ViewArray>().unwrap(); self.views(arr, a.views(), a.data_buffers()) }
            PhysicalType::BinaryView => { let a = any.downcast_ref::<BinaryViewArray>().unwrap(); self.views(arr, a.views(), a.data_buffers()) }
            PhysicalType::List => self.list(arr, any.downcast_ref::<ListArray<i32>>().unwrap()),
            PhysicalType::LargeList => self.list(arr, any.downcast_ref::<ListArray<i64>>().unwrap()),
            PhysicalType::FixedSizeList => {
                let a = any.downcast_ref::<FixedSizeListArray>().unwrap();
                self.validity(arr)?;
                self.array(a.values().sliced(0, a.len() * a.size()).as_ref())
            }
            PhysicalType::Struct => {
                self.validity(arr)?;
                any.downcast_ref::<StructArray>().unwrap().values().iter().try_for_each(|f| self.array(f.as_ref()))
            }
            PhysicalType::Dictionary(_) => {
                macro_rules! dict {
                    ($($k:ty),*) => {$(
                        if let Some(d) = any.downcast_ref::<DictionaryArray<$k>>() {
                            self.array(d.keys())?;
                            return self.array(d.values().as_ref()); // the dictionary batch
                        }
                    )*};
                }
                dict!(u8, u16, u32, u64, i8, i16, i32, i64);
                polars_bail!(ComputeError: "describe sizes: unexpected dictionary key type")
            }
            other => polars_bail!(ComputeError: "describe sizes: unsupported Arrow type {other:?}"),
        }
    }
}

pub(crate) fn ipc_body_bytes(arr: &dyn Array, level: Option<i32>) -> PolarsResult<u64> {
    let mut body = Body { level, bytes: 0 };
    body.array(arr)?;
    Ok(body.bytes)
}

fn nests_int128(dtype: &DataType) -> bool {
    match dtype {
        DataType::List(inner) | DataType::Array(inner, _) => **inner == DataType::Int128 || nests_int128(inner),
        DataType::Struct(fields) => fields.iter().any(|f| *f.dtype() == DataType::Int128 || nests_int128(f.dtype())),
        _ => false,
    }
}

type Sizes = [Option<u64>; 4];
const SIZE_FIELDS: [&str; 4] = ["size_bytes", "size_zstd_bytes", "size_polars_bytes", "size_polars_zstd_bytes"];

fn sizes(s: &Series, level: i32) -> PolarsResult<Sizes> {
    if nests_int128(s.dtype()) {
        return Ok([None; 4]);
    }
    let s = s.rechunk();
    if s.n_chunks() == 0 {
        return Ok([Some(0); 4]);
    }
    let classic = s.to_arrow(0, CompatLevel::oldest());
    let native = s.to_arrow(0, CompatLevel::newest());
    Ok([
        Some(ipc_body_bytes(classic.as_ref(), None)?),
        Some(ipc_body_bytes(classic.as_ref(), Some(level))?),
        Some(ipc_body_bytes(native.as_ref(), None)?),
        Some(ipc_body_bytes(native.as_ref(), Some(level))?),
    ])
}

fn sizes_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let mut fields = vec![Field::new("column".into(), DataType::String)];
    fields.extend(SIZE_FIELDS.iter().map(|n| Field::new((*n).into(), DataType::UInt64)));
    Ok(Field::new("column_sizes".into(), DataType::Struct(fields)))
}

#[derive(Deserialize)]
struct SizesKwargs {
    zstd_level: i32,
}

pub(crate) fn column_sizes_impl(inputs: &[Series], level: i32) -> PolarsResult<Series> {
    let rows: Vec<Sizes> = inputs.par_iter().map(|s| sizes(s, level)).collect::<PolarsResult<_>>()?;
    let mut columns = vec![StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str())).into_series().with_name("column".into())];
    for (j, name) in SIZE_FIELDS.iter().enumerate() {
        columns.push(UInt64Chunked::from_iter_options((*name).into(), rows.iter().map(|r| r[j])).into_series());
    }
    Ok(StructChunked::from_series("column_sizes".into(), inputs.len(), columns.iter())?.into_series())
}

#[polars_expr(output_type_func=sizes_output_type)]
fn column_sizes(inputs: &[Series], kwargs: SizesKwargs) -> PolarsResult<Series> {
    column_sizes_impl(inputs, kwargs.zstd_level)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrow(s: &Series) -> ArrayRef {
        s.rechunk().to_arrow(0, CompatLevel::oldest())
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
        let s = Series::new("x".into(), (0..1_000).map(|i| (i % 3 != 0).then_some(i)).collect::<Vec<Option<i32>>>());
        assert_eq!(ipc_body_bytes(arrow(&s).as_ref(), None).unwrap(), 4_128);
        assert_eq!(ipc_body_bytes(arrow(&Series::new_empty("x".into(), &DataType::Int32)).as_ref(), None).unwrap(), 0);
    }

    #[test]
    fn large_utf8_offsets_values_validity() {
        assert_eq!(ipc_body_bytes(arrow(&Series::new("x".into(), &[Some("ab"), None])).as_ref(), None).unwrap(), 40);
        assert_eq!(ipc_body_bytes(arrow(&Series::new_empty("x".into(), &DataType::String)).as_ref(), None).unwrap(), 8);
    }

    #[test]
    fn nested_int128_sizes_are_null() {
        let s = Series::new("x".into(), [Some(Series::new("".into(), &[1i128]))]);
        assert_eq!(sizes(&s, 1).unwrap(), [None; 4]);
    }
}
