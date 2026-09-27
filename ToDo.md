# ToDo - A List of future enhancements to this project

1. Run Dependabot, Secret Scanning, CodeQL, Semgrep, SonarQube community edition on this repo & address issues

2. Increase both Python and Rust test coverage to 95%+ (currently Python at 91%, Rust at 89%)

3. Introduce a new function 'Shrink' (Unordered Single Column) that reduces each column to it's optimal data type (per the recommender function) and returns optimised columns.  This is an element-wise response rather than a table of rows & values.

3. Make the column selection criteria & parameters to each of the functions uniform.  Today, some take optional 'pairs' (ordered pair-wise functions), some take other inputs.  All functions should operate as follows:
- take optional parameters for columns (single-column functions) or pairs of dataframe/columns (un ordered pair/triplet-wise colums) or pairs of columns (orderd pair-wise columns).
- in absence of a specific list of parameters, filter just on those columns that apply.  For membership/similarity this will be columns of the same arrow data type.
- function specific parameters - such as the Bloom Filter byte array - remain untouched.

4. Introduce a new class of function: Ordered Single Column.   We will then have 4 symmetrical sets of functions.
a. Unordered Single Column: GCD, Describe, Recommend, Marginal Entropy
b. Ordered Single Column - see below
c. Unordered Pair-Wise (currently called multi-set): Membership, Similarity
d. Ordered Pair-Wise: Chi-squared, Adjusted Rand Index, Joint Entropy

As the first candidates for Ordered Single Column we will add:
- Wald-Wolfowitz runs test (to identify candidates for REE)
- Window functions of 2 rows: delta, double-delta, entropy and min/max values for each of these.
- Window functions of a specified width to calculate the number of distinct values and min/max values as a portion of the entire data (to identify candidates for ClickHouse's minmax and skipindexes)
- Window functions to measure the frequency and number of rows inbetween any repeated values (to calculate selectivity and identify candidates for ClickHouse's bloom_filter skipping indexes)

5. Extend the Recommender to include additional encoding and indexing options over and above Dictionary Encoding: REE, GCD, Delta, Double Delta and Skipping Index candidates: min/max, set and bloom_filter (and granule size)

6. Implementation clean-ups documented in CLAUDE.md

7. Introduce a new function DetectUniqueKeys (ordered pair/triplet-wise) - uses the joint and marginal entropies to detect groups of up to three columns that are unique (or close to unique for a given threshold) for a dataframe using the joint and marginal entropies.  A group of columns that is unique:
- a single column will have the max entropy for the number of number of columns N: log2(N)
- any pair of columns will individually (marginally) have entropies lower than the mx entropy, have jointly the max entropy and will have low Mutual information (ie. be 'barely' unique to avoid situations were pairs of columns are unique by random chance).  To avoid searching unecessary pairs, the sum of marginal entropies must be at least the max entropy: joint entropy can never be greater than sum of marginal entropies.
- any triplet will follow the rules above.  Again early pruning is key.

8. Move /tests/data/large_dataset.arrow