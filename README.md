# Dictionary Codec Implementation

## Introduction

This project implements a dictionary codec in Python to compress columnar data and accelerate search operations. Dictionary encoding replaces repeated data items with unique identifiers, reducing data footprint and enabling faster queries. The implementation includes functionality for encoding data, querying encoded data, and performing baseline scans for performance comparisons. Advanced techniques such as multi-threading and SIMD instructions are utilized to enhance performance.

## Project Structure

The project consists of the following files. Also note that the full `Column.txt` was used for the program; however, the files that have been uploaded are reduced to ten lines each for simple understanding. 

`dictionary_codec.py` is the main Python script containing all encoding and query functionalities. It includes multi-threading for parallel processing and SIMD optimization via NumPy for accelerated query operations.

`Column.txt` is the raw column data file used for encoding and querying operations.

`dictionary.txt` is the output file containing the dictionary of unique data items and their assigned IDs.

`encoded_data.txt` is the output file containing the encoded data column, where each entry is represented by its corresponding ID.

## Dependencies

This project requires Python 3.12.7. It also depends on the NumPy library for efficient array handling and SIMD operations. NumPy can be installed using the following command:

## Usage Instructions

Ensure the raw data file `Column.txt` is in the same directory as `dictionary_codec.py`. The data file should contain one data item per line. To run the script, follow these steps:

1. Open the terminal or command prompt.
2. Navigate to the directory containing the project files.
3. Execute the script using the command:

## Code Structure

### Data Reading and Writing

The `read_data` function reads the raw data from the file, optionally limiting the number of lines for testing.

The `write_encoded_column_file` function writes the dictionary and encoded data to files.

The `read_encoded_column_file` function reads the dictionary and encoded data from files.

### Encoding Functions

The `build_dictionary` function constructs a dictionary of unique data items using multi-threading. It assigns a unique integer ID to each item.

The `encode_data` function encodes the raw data by replacing each item with its corresponding dictionary ID. This function also supports multi-threading.

### Query Functions

The `query_data_item_vanilla` function performs a vanilla search for a single data item in the raw data.

The `query_data_item_no_simd` function performs a dictionary-based search for a single data item without SIMD.

The `query_data_item` function performs a dictionary-based search for a single data item with SIMD using NumPy arrays.

The `prefix_search_vanilla` function performs a vanilla prefix search on the raw data.

The `prefix_search_no_simd` function performs a dictionary-based prefix search without SIMD.

The `prefix_search` function performs a dictionary-based prefix search with SIMD using NumPy arrays.

### Main Execution Flow

The `main` function orchestrates the encoding and query processes, measures performance, and outputs results to the console.

## Multi-threading Implementation

Multi-threading is implemented during dictionary creation and data encoding using the Python `multiprocessing` module. The data is divided into chunks, and each chunk is processed independently by a separate thread. This approach reduces processing time for large datasets. For smaller datasets, the overhead of thread management can offset the benefits of parallelization.

## SIMD Applications

SIMD instructions are utilized during query operations to accelerate comparisons. By representing encoded data as NumPy arrays, vectorized operations are performed to allow multiple comparisons simultaneously. This optimization reduces computational workload and significantly speeds up prefix searches and data item checks for large datasets. The performance gain is less noticeable for small datasets due to the overhead of array conversion.

## Performance Observations

Encoding performance was tested with varying thread counts. For small datasets, increasing the number of threads did not consistently reduce processing time due to multi-threading overhead. For larger datasets, multi-threading is expected to provide significant speed improvements.

Query performance comparisons demonstrated the effectiveness of dictionary encoding and SIMD optimizations. While dictionary searches are faster than vanilla scans due to efficient integer comparisons, SIMD-enabled operations further improve performance in prefix searches.

Example results for a dataset of 100,000 entries:

## Possible Optimizations

Processing larger datasets would better demonstrate the benefits of multi-threading and SIMD. Alternative libraries like `pandas` for data manipulation or implementing performance-critical sections in Cython could further improve efficiency. Optimizing multi-threading by fine-tuning the number of processes or using threading for I/O-bound tasks could also reduce overhead. Leveraging low-level SIMD instructions through libraries like `numba` or `pythran` might enhance performance for repetitive operations.
