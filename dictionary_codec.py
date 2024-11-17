import multiprocessing as mp
import numpy as np
import time

def read_data(filename, num_lines=None):
    data = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if num_lines and i >= num_lines:
                break
            data.append(line.strip())
    return data

def get_unique_strings(data_chunk):
    return set(data_chunk)

def build_dictionary(data, num_processes):
    chunk_size = len(data) // num_processes
    chunks = [data[i*chunk_size : (i+1)*chunk_size] for i in range(num_processes)]
    if len(data) % num_processes != 0:
        chunks.append(data[num_processes*chunk_size:])

    pool = mp.Pool(processes=num_processes)
    results = pool.map(get_unique_strings, chunks)
    pool.close()
    pool.join()

    # Combine the sets
    unique_strings = set()
    for s in results:
        unique_strings.update(s)

    # Assign IDs
    string_to_id = {string: idx for idx, string in enumerate(unique_strings)}
    id_to_string = {idx: string for string, idx in string_to_id.items()}

    return string_to_id, id_to_string

def encode_data_chunk(args):
    data_chunk, string_to_id = args
    return [string_to_id[s] for s in data_chunk]

def encode_data(data, string_to_id, num_processes):
    chunk_size = len(data) // num_processes
    chunks = [data[i*chunk_size : (i+1)*chunk_size] for i in range(num_processes)]
    if len(data) % num_processes != 0:
        chunks.append(data[num_processes*chunk_size:])

    pool = mp.Pool(processes=num_processes)
    results = pool.map(encode_data_chunk, [(chunk, string_to_id) for chunk in chunks])
    pool.close()
    pool.join()

    # Flatten the list of lists
    encoded_data = [item for sublist in results for item in sublist]

    return encoded_data

def write_encoded_column_file(dictionary, encoded_data, dict_filename, data_filename):
    # Write dictionary
    with open(dict_filename, 'w') as f:
        for idx in sorted(dictionary.keys()):
            f.write(f"{idx},{dictionary[idx]}\n")

    # Write encoded data
    with open(data_filename, 'w') as f:
        for id in encoded_data:
            f.write(f"{id}\n")

def read_encoded_column_file(dict_filename, data_filename):
    # Read dictionary
    id_to_string = {}
    with open(dict_filename, 'r') as f:
        for line in f:
            idx, string = line.strip().split(',', 1)
            id_to_string[int(idx)] = string

    # Build string_to_id
    string_to_id = {string: idx for idx, string in id_to_string.items()}

    # Read encoded data
    with open(data_filename, 'r') as f:
        encoded_data = [int(line.strip()) for line in f]

    return string_to_id, id_to_string, encoded_data

def query_data_item(encoded_data, item_id):
    # Using SIMD (NumPy) for vectorized comparison
    encoded_array = np.array(encoded_data)
    indices = np.where(encoded_array == item_id)[0]
    return indices.tolist()

def query_data_item_no_simd(encoded_data, item_id):
    # Without SIMD, using regular loops
    indices = [i for i, id in enumerate(encoded_data) if id == item_id]
    return indices

def prefix_search(string_to_id, encoded_data, prefix):
    # Using SIMD (NumPy)
    matching_strings = [s for s in string_to_id.keys() if s.startswith(prefix)]
    if not matching_strings:
        return {}
    matching_ids = [string_to_id[s] for s in matching_strings]
    encoded_array = np.array(encoded_data)
    results = {}
    for item_id, s in zip(matching_ids, matching_strings):
        indices = np.where(encoded_array == item_id)[0]
        results[s] = indices.tolist()
    return results

def prefix_search_no_simd(string_to_id, encoded_data, prefix):
    # Without SIMD
    matching_strings = [s for s in string_to_id.keys() if s.startswith(prefix)]
    if not matching_strings:
        return {}
    matching_ids = [string_to_id[s] for s in matching_strings]
    results = {}
    for item_id, s in zip(matching_ids, matching_strings):
        indices = [i for i, id in enumerate(encoded_data) if id == item_id]
        results[s] = indices
    return results

def query_data_item_vanilla(data, data_item):
    indices = [i for i, s in enumerate(data) if s == data_item]
    return indices

def prefix_search_vanilla(data, prefix):
    results = {}
    for i, s in enumerate(data):
        if s.startswith(prefix):
            if s in results:
                results[s].append(i)
            else:
                results[s] = [i]
    return results

def main():
    data_file = 'Column.txt'
    dict_file = 'dictionary.txt'
    encoded_data_file = 'encoded_data.txt'
    num_processes = 4

    # Read only the first 100,000 lines for testing
    data = read_data(data_file, num_lines=100000)

    if not data:
        print("Data is empty or could not be read.")
        return

    # Measure encoding performance with different thread counts
    for n_threads in [1, 2, 4, 8]:
        print(f"\nEncoding with {n_threads} threads:")
        start_time = time.time()
        string_to_id, id_to_string = build_dictionary(data, n_threads)
        dict_time = time.time() - start_time
        print(f"Dictionary built in {dict_time:.2f} seconds.")

        start_time = time.time()
        encoded_data = encode_data(data, string_to_id, n_threads)
        encode_time = time.time() - start_time
        print(f"Data encoded in {encode_time:.2f} seconds.")

    # Write the encoded column file
    write_encoded_column_file(id_to_string, encoded_data, dict_file, encoded_data_file)

    # Querying
    string_to_id, id_to_string, encoded_data = read_encoded_column_file(dict_file, encoded_data_file)

    # Single data item to search
    data_item = data[0]
    item_id = string_to_id.get(data_item, None)

    # Vanilla search
    start_time = time.time()
    indices_vanilla = query_data_item_vanilla(data, data_item)
    time_vanilla = time.time() - start_time
    print(f"\nVanilla search found {len(indices_vanilla)} occurrences in {time_vanilla:.6f} seconds.")

    if item_id is not None:
        # Dictionary search without SIMD
        start_time = time.time()
        indices_no_simd = query_data_item_no_simd(encoded_data, item_id)
        time_no_simd = time.time() - start_time
        print(f"Dictionary search without SIMD found {len(indices_no_simd)} occurrences in {time_no_simd:.6f} seconds.")

        # Dictionary search with SIMD
        start_time = time.time()
        indices_simd = query_data_item(encoded_data, item_id)
        time_simd = time.time() - start_time
        print(f"Dictionary search with SIMD found {len(indices_simd)} occurrences in {time_simd:.6f} seconds.")

    # Prefix search
    prefix = data_item[:3]

    # Vanilla prefix search
    start_time = time.time()
    results_vanilla = prefix_search_vanilla(data, prefix)
    time_vanilla_prefix = time.time() - start_time
    print(f"\nVanilla prefix search found {len(results_vanilla)} unique items in {time_vanilla_prefix:.6f} seconds.")

    # Dictionary prefix search without SIMD
    start_time = time.time()
    results_no_simd = prefix_search_no_simd(string_to_id, encoded_data, prefix)
    time_no_simd_prefix = time.time() - start_time
    print(f"Dictionary prefix search without SIMD found {len(results_no_simd)} unique items in {time_no_simd_prefix:.6f} seconds.")

    # Dictionary prefix search with SIMD
    start_time = time.time()
    results_simd = prefix_search(string_to_id, encoded_data, prefix)
    time_simd_prefix = time.time() - start_time
    print(f"Dictionary prefix search with SIMD found {len(results_simd)} unique items in {time_simd_prefix:.6f} seconds.")

if __name__ == "__main__":
    main()
