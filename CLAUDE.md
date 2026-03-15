# Project Objectives
The objective of this project is to determine patterns and relationships between columns both within and between dataframes.

# Technology
Leverage the Arrow columnar data format.
Use packages that leverage the Arrow columnar data format such as Polars.
The project should be written in Python, with extensions written in Rust (using pyo3 or pyo3polars) for performance purposes.

# Python Environment
Use Conda environment p312 found here: C:\Users\Ben\.conda\envs\p312

# Coding Style
Prioritise performance & simplicity

# Current Focus
Building high-performance functions to determine relationships between columns in the dataset.  The calculation of joint entropies between two and three columns is complete.  We now focus on calculating the Chi-Squared independence metrics on categorical (binary, nominal) data.

