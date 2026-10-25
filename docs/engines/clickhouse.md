# ClickHouse optimisations

This file is intended as an aide-memoire of the mechanisms that exist within ClickHouse to improve query performance.

The general way to approach query optimisation in ClickHouse is to think about how to prune the amount of data that needs to be read.  For example: a 32-bit width column is quicker to read than a 64-bit width column; a column that can be reconstructed from a codec of 32-bit values msy, depending on the codec, be quicker to construct and read than a 64-bit column; a query that can skip granules through primary-key sorting, skip indexes or projections will read less data and will return quicker than a full scan of columns.

## Optimisation Features

### Partitions

Partitions are physical separations of data that can reduce the amount of data that needs to be read where queries only need data from a single partition.  Conversely, queries that read multiple partitions perform poorly and such partitioning criteria should be avoided.  

Advice from ClickHouse is to aim for hundreds of partitions at most, not thousands.  Ingestion datetime partitioned by month, assuming ingestion datetime is a useful predicate, is likely to improve query performance.

https://clickhouse.com/docs/concepts/core-concepts/partitions
https://clickhouse.com/docs/concepts/best-practices/partitioning-keys


### Data Type Shrinking & Explicit Nullability

Wherever possible, datatypes with fewer bits should be used.  A general hierarchy follows.

https://clickhouse.com/docs/concepts/best-practices/select-data-type
https://clickhouse.com/docs/concepts/best-practices/avoidnullablecolumns

1. Nullable datatype

Nullable datatypes require an additional bit per row which can be avoided if not needed.

Nullable columns can be identified via Null values in sampled data: there is no other way to estimate where values may be missing.


2. Boolean (1-bit)

Use for [True, False], [0, 1], [Y, N] and possibly columns with just 2 values.


3.  Unsigned Integers (8, 16, 32 or 64 bits)

Use where the min value >= 0 as unsigned integers support double the range and may consequently allow a narrower width to be used in some cases.


4.  Signed Integers (8, 16, 32 or 64 bits)

Use for all other integer values.


5.  Fixed-point Decimal (32, 64 or 128 bits)

Where the maximum number of decimal places is known, prefer Decimal over Float.  Depending on the range of values in the column, the Decimal type could result in a greater number of contiguous zero values in the upper bits which will compress or encode more efficiently than Floating types where an equivalent value saved as an exponent + mantissa would not have as many leading zeros.

Further, if a column contains similar values, the Decimal type will result in binary representations similar to one another which is not necessarily the case with the Float type which in turn provides repeated patterns that byte compression algorithms can exploit.

Compare the actual bytes for two adjacent prices, 123.45 and 123.46:

Decimal64(2):
  12345  ->  00 00 00 00 00 00 30 39
  12346  ->  00 00 00 00 00 00 30 3A

Float64:
  123.45 ->  40 5E DC CC CC CC CC CD
  123.46 ->  40 5E DD 1E B8 51 EB 85

The most efficient way to identify the max number of decimal places in floating point numbers is to cast to string and count the number of digits after the dot character.  Programming languages already employ fast float to string algorithms.


6.  Float (32 or 64 bits)

Float32 for <7 digits, Float64 for <15 significant digits.  Fallback for all other numeric columns that cannot be encoded as integers or decimals.


7.  Date32 (32 bits)

Use for date values without any time component.


8.  Date64 (64 bits)

Use for date values with a time component up to millisecond granularity.


9. Time32 (32 bits)

Use for time-of-day values without a date component up to millisecond granularity


10. Time64 (64 bits)

Use for time-of-day values without a date component up to nanosecond granularity

11. Timestamp (64 bits)

Same as Date64 but with a uniform timezone (timezone is saved as part of the column metadata)

12. Timestamp with offset (64 bits + offset metadata)

Same as Date64 but with a variable timezone (timezone is saved per value)

13. String (variable)

All alpha(-numeric) values that do not fit into the above.

14.  List or Large List

Use where there is more than one scalar value of the same type in any given row.

15.  Struct

Use where there is more than one value of different types in any given row.


### Primary Key (Ordering)

In ClickHouse the primary key works by sorting data on the given rows, and then creating a sparse index (i.e. just the first and last values) for each granule (8192 rows).

https://clickhouse.com/docs/concepts/core-concepts/primary-indexes
https://clickhouse.com/docs/guides/clickhouse/data-modelling/sparse-primary-indexes

A primary key can consist of multiple columns.  It is most effective at query time when the leading edge is used.  For example a primary key of (a, b, c) will prune effectively on a or (a,b) but potentially less effectively on c.

Candidates for the primary key can be identified by a) which columns most often appear in queries (requires observed usage), b) which columns when sorted make the remaining columns the most locally clustered.

The effect one column has on the local clustering of another column can be identified by calculating the pairwise conditional entropy (or mutual information) between columns.

https://en.wikipedia.org/wiki/Entropy_(information_theory)

And more generally, it can be determined if two columns cluster in similar ways by using the Adjusted Rand Index.

https://en.wikipedia.org/wiki/Rand_index


### Projections (Parallel data storage with a different ordering)

Projections create a second 'hidden' table from the first, either by a) storing all columns in the sort order of the projection rather than the original primary key, b) storing just the sorting key and the part offset on the original table.  At query time, the index of the projection is used to identify row positions, but the data is read from the original table.  This reduces storage at the expense of increased i/o and half as much performance as the full projection at query time.  

Projections are therefore similar to Materialised views except a) they are hidden from the user, b) asynchronously updated by ClickHouse, c) the ClickHouse query optimiser decides when to use a projection.  A full comparison is here:  https://clickhouse.com/docs/concepts/features/projections/materialized-views-versus-projections


### Codecs

Codecs re-encode individual values in the column in a more efficient manner.

https://clickhouse.com/blog/optimize-clickhouse-codecs-compression-schema
https://clickhouse.com/docs/reference/statements/create/table/codec

1. *Low Cardinality*  Also known as 'dictionary encoding' or 'categorical encoding'.  Instead of saving String values in the column, a dictionary of unique values is created and the values in the column substituted for the dictionary keys.  In Arrow and Polars it is possible to specify the key width but ClickHouse determines this for itself.  

ClickHouse recommends only applying the Low Cardinality Codec to columns <10,000 unique values (theoretically a 16-bit key).

The cardinality of a column can be estimated from three or more samples of data, as long as the cardinality ratio of those samples is <50% using the Lincoln-Petersen estimation + Chao1 correction and using Bootstrap Confidence Intervals.

https://en.wikipedia.org/wiki/Lincoln_index

Both scalar String values and Strings as the inner type of List/Array columns will nearly always be more efficiently encoded as Low Cardinality/Categorical.

2. *GCD (Greatest Common Divisor)*  Where values in a column are all multiples of a common divisor, it may be possible to shrink the datatype by dividing all values by the GCD and storing the multiplier in the column's metadata.

GCD can be used with integer, decimal and date/time columns.

A GCD can be identified by using the binary GCD algorithm, also known as Stein's algorithm or the binary Euclidean algorithm.

GCD encoding promises to improve performance if the rescaled values shrink the data type.

4. *T64*  Crops commonly unused (zero) higher (leading) bits in order to shrink the data.  The codec is applied 64 rows at a time.  The unused bits are those that do not differ between the smallest and largest value in the 64 row block.  More bits are truncated where the original values are within a narrow range.

[Need a mechanism to identify candidates - presumably narrow range per sequence of 64]

3. *Delta, Double Delta*  Instead of storing individual values, it may be possible to shrink the datatype by instead storing the arithmetic difference between values with the starting value saved in the column's metadata using the Delta codec.  Further, storing the difference of differences may shrink the column further, storing both the starting value and delta value in the column description.  Double delta encoding may lead to zero values where a column contains a consistent monotonic series.

Autocorrelation can be used to determine candidates for delta encoding.

https://en.wikipedia.org/wiki/Autocorrelation

5. *Gorilla*  Instead of storing individual values, stores the XOR delta.  Most appropriate for Floating point values with small variations between adjacent values.

[Need a mechanism to identify candidates]


### Compression

ClickHouse will apply general purpose byte compression to columns following any codecs.

https://clickhouse.com/docs/reference/statements/create/table#column_compression_codec

By default, ClickHouse will apply LZ4 compression, but also supports LZ4HC (High Compression) with a configurable level and ZSTD.  ClickHouse's general advice is to use ZSTD unless LZ4 is significantly faster, as ZSTD is the better all-round compression algorithm

Compression works in conjunction with the codecs above.  Codec on their own usually don't shrink the data that much; However, their output can be far more compressible than the raw data alone.

LZ4 is a pure LZ77 match-finder with no entropy coding. The byte stream D2 04 00 00 | 39 30 00 00 | 1A 07 00 00 has zeros at stride 4, but the interleaved payload bytes break up the runs, and LZ4 needs a minimum match length to emit a reference. In practice LZ4 recovers maybe 10–25% of that theoretical 50%. ZSTD, which has a Huffman/FSE entropy stage, encodes the frequently-occurring zero byte as a sub-bit symbol and recovers most of it. So: on LZ4 the type change is worth a lot; on ZSTD it is worth much less.

LZ4 decompression speed is roughly proportional to output bytes (typically 2–5 GB/s per core)

Compression is applied per block (min_compress_block_size 65536, max_compress_block_size 1048576 by default), which is why a granule is the minimum readable unit.



### Skipping Indexes

Skipping indexes store information about the values in a column at the granularity of one or more granules.  The number of granules included in each skipping index 'block' is determined at the time of the skipping index creation.  Skipping indexes allow the query engine to skip reading granules where the filter value falls outside of the skipping index metadata.  Skipping indexes trade query speed improvements for additional processing at ingestion (merge cost and additional storage).  

https://clickhouse.com/docs/concepts/features/performance/skip-indexes/skipping-indexes
https://clickhouse.com/docs/concepts/best-practices/using-data-skipping-indices

Skipping indexes exploit local clustering of values per block.  On randomly distributed data they cost CPU and storage and save nothing.

Clustering of values within a (numeric) dataset can be identified via the Index of Dispersion.  Poisson-distributed values will have a mean equal to the variance.  The index of dispersion is σ squared/μ.  And index of dispersion = 1 indicates a true random distribution, > 1 indicates clustering, whereas an index of dispersion < 1 indicates an under-dispersed set.

https://en.wikipedia.org/wiki/Index_of_dispersion

1. minmax

Stores the minimum and maximum value for the block.  Allows the query engine to skip granules where the filter criteria fall outside of this range.

Only efficient where the range of values per block is small compared to the full range of values in the column.

[Need a candidate identification mechanism]

2. set

Stores the unique set of values for the block.  Allows the query engine to skip granules that do not contain the filter value.

[Need a candidate identification mechanism]

3. text

Used for plain-text search.  Stores a full inverted index (like an index at the back of a book that provides page number for a given search term).

4. bloom_filter

Creates a bloom_filter per block.  

Bloom Filters return false-positives at a specified probability but will never return a false negatives and allows the block to be skipped if the filter value is not found.  Therefore Bloom Filters are most efficient where inclusive criteria are used, i.e. value = X, as granules that do not contain X in their skipping bloom filter can be dropped.  

Bloom filters are described here: https://en.wikipedia.org/wiki/Bloom_filter

[Need a technique for identifying candidates - presumably similar to PREWHERE.].

### PREWHERE

1. PREWHERE is where ClickHouse reads and evaluates a filter column in order to discount granules that do not meet the condition and therefore do not need to be read.

ClickHouse can automatically apply PREWHERE filtering by enabling the setting optimize_move_to_prewhere.  A PREWHERE clause can also be used in SQL. Syntax is the same as WHERE.  PREWHERE can be used in conjunction with WHERE: PREWHERE will be evaluated first.

Candidates for PREWHERE can be identified by their selectivity at the block level.

### Dictionaries


### Materialised Views

https://clickhouse.com/docs/concepts/best-practices/use-materialized-views


### Caches

https://clickhouse.com/docs/concepts/features/performance/caches/query-condition-cache


## ClickHouse Tooling


1. EXPLAIN variants
EXPLAIN indexes = 1 — the one you'll use most. Shows, per table, how many parts and granules each index stage eliminated. This is how you verify a new sort order or projection is actually being used before you commit to it.
EXPLAIN ESTIMATE — rows/parts/marks it expects to read, without running.
EXPLAIN PIPELINE — the execution graph and parallelism.
EXPLAIN QUERY TREE — post-analyzer logical plan; useful when you suspect a predicate isn't being pushed down.

2.  system.query_log

Set log_queries=1, give system.query_log a sensible TTL (90 days is usually enough and it's not free), and consider log_query_settings.

3.  system.parts_columns

per-column compressed and uncompressed bytes, per part. This directly answers "could data types be optimised"

4. 

system.parts — part counts, sizes, merge levels. Watch for tables accumulating parts.
system.data_skipping_indices — includes index size, so you can find skip indices that cost storage and (per query_log) skip nothing.
system.projections — projection inventory and sizes.
system.query_thread_log and system.query_views_log — per-thread detail and materialized-view execution cost (MV cost is charged to the insert, which surprises people).
system.trace_log — sampled stack traces. Set query_profiler_real_time_period_ns and you can build genuine flamegraphs of where CPU goes. Overkill for most tuning, invaluable when something is mysteriously slow.
system.metric_log / system.asynchronous_metric_log — instance-level time series for correlating slow queries with memory pressure or merge storms.
The built-in /dashboard endpoint gives you a decent visual overview without deploying anything.