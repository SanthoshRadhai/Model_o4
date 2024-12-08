import os
import binascii
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import torch  # Import inside the function
import logging  # Import logging module


def hex_to_binary(hex_data):
    try:
        binary_data = bin(int(binascii.unhexlify(hex_data).hex(), 16))[2:]
        return binary_data.zfill(len(hex_data) * 4)
    except Exception as e:
        return None  # Handle invalid hex data gracefully

def extract_features_batch(binary_data_list):
    from o1_Acc_Feature_Extract import FeatureExtract  # Import inside the function to avoid issues

    features_list = [{} for _ in binary_data_list]

    # Extract all values from approximate_entropy_test
    print("Extracting approximate_entropy_test features...")
    approximate_entropy_results = FeatureExtract.approximate_entropy_test_batch(binary_data_list)
    for i, result in enumerate(approximate_entropy_results):
        if result is not None:
            features = {
                f"approximate_entropy_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            }
            features_list[i].update(features)
        else:
            features_list[i]['approximate_entropy_error'] = 'Failed to compute approximate entropy.'

    # Extract all values from binary_matrix_rank_test_16 and binary_matrix_rank_test_32
    print("Extracting binary_matrix_rank_test_16 and binary_matrix_rank_test_32 features...")

    # Call the separate methods for 16x16 and 32x32 block sizes
    binary_matrix_rank_results_16 = FeatureExtract.binary_matrix_rank_test_batch_16(binary_data_list)
    binary_matrix_rank_results_32 = FeatureExtract.binary_matrix_rank_test_batch_32(binary_data_list)

    # Iterate through the results and update the features_list accordingly
    for i, (result_16, result_32) in enumerate(zip(binary_matrix_rank_results_16, binary_matrix_rank_results_32)):
        # Process 16x16 results
        if result_16 is not None:
            features_list[i].update({
                f"binary_matrix_rank_16_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result_16.items()
            })
        else:
            features_list[i]['binary_matrix_rank_16_error'] = 'Failed to compute binary matrix rank.'

        # Process 32x32 results
        if result_32 is not None:
            features_list[i].update({
                f"binary_matrix_rank_32_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result_32.items()
            })
        else:
            features_list[i]['binary_matrix_rank_32_error'] = 'Failed to compute binary matrix rank.'

    # Extract all values from cumulative_sums_test
    print("Extracting cumulative_sums_test features...")
    cumulative_sums_results = FeatureExtract.cumulative_sums_test_batch(binary_data_list)
    for i, result in enumerate(cumulative_sums_results):
        if result is not None:
            features_list[i].update({
                f"cumulative_sums_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            })
        else:
            features_list[i]['cumulative_sums_error'] = 'Failed to compute cumulative sums.'

    # Extract all values from serial_test_and_extract_features
    print("Extracting serial_test_and_extract_features features...")
    serial_test_results = FeatureExtract.serial_test_and_extract_features_batch(binary_data_list)
    for i, result in enumerate(serial_test_results):
        if result is not None:
            features_list[i].update({
                f"serial_test_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            })
        else:
            features_list[i]['serial_test_error'] = 'Failed to compute serial test.'

    # Extract all values from spectral_test
    print("Extracting spectral_test features...")
    spectral_test_results = FeatureExtract.spectral_test_batch(binary_data_list)
    for i, result in enumerate(spectral_test_results):
        if result is not None:
            features_list[i].update({
                f"spectral_test_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            })
        else:
            features_list[i]['spectral_test_error'] = 'Failed to compute spectral test.'




    print("Extracting linear_complexity_test features...")
    linear_complexity_results = FeatureExtract.linear_complexity_test_batch(
        serial_binary_data_list=binary_data_list,
        block_size=128
    )
    for i, result in enumerate(linear_complexity_results):
        if i % 100 == 0 and i > 0:
            logging.debug(f"Processed {i} / {len(linear_complexity_results)} linear complexity tests.")
        
        if result is not None and 'error' not in result:
            features_list[i].update({
                f"linear_complexity_{key}": value
                for key, value in result.items()
            })
        else:
            features_list[i]['linear_complexity_error'] = 'Failed to compute linear complexity.'




    # Extract all values from longest_one_block_test
    print("Extracting longest_one_block_test features...")
    longest_one_block_results = FeatureExtract.longest_one_block_test_batch(binary_data_list)
    for i, result in enumerate(longest_one_block_results):
        if result is not None:
            features_list[i].update({
                f"longest_one_block_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            })
        else:
            features_list[i]['longest_one_block_error'] = 'Failed to compute longest one block.'
        

    # Extract all values from block_frequency_multiple_sizes
    print("Extracting block_frequency_multiple_sizes features...")
    block_freq_results = FeatureExtract.block_frequency_multiple_sizes_batch(binary_data_list)
    for i, result_list in enumerate(block_freq_results):
        if result_list is not None:
            for result in result_list:
                features_list[i].update({
                    key: value.tolist() if isinstance(value, torch.Tensor) else value 
                    for key, value in result.items()
                })
        else:
            features_list[i]['block_frequency_multiple_sizes_error'] = 'Failed to compute block frequency multiple sizes.'

    # Extract all values from statistical_test
    print("Extracting statistical_test features...")
    statistical_test_results = FeatureExtract.statistical_test_batch(binary_data_list)
    for i, result in enumerate(statistical_test_results):
        if result is not None:
            features_list[i].update({
                f"statistical_test_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            })
        else:
            features_list[i]['statistical_test_error'] = 'Failed to compute statistical test.'

    # Extract all values from extract_run_test_features
    print("Extracting extract_run_test_features features...")
    try:
        run_test_results = FeatureExtract.extract_run_test_features_batch(binary_data_list, verbose=False)
        for i, result in enumerate(run_test_results):
            if result is not None:
                features_list[i].update({
                    f"run_test_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                    for key, value in result.items()
                })
            else:
                features_list[i]['run_test_error'] = 'Failed to compute run test.'
    except Exception as e:
        print(f"Error processing batch: {e}")

    # Extract all values from monobit_test
    print("Extracting monobit_test features...")
    monobit_results = FeatureExtract.monobit_test_batch(binary_data_list)
    for i, result in enumerate(monobit_results):
        if result is not None:
            features_list[i].update({
                f"monobit_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            })
        else:
            features_list[i]['monobit_error'] = 'Failed to compute monobit test.'

    # Extract all values from spectral_test_on_blocks
    print("Extracting spectral_test_on_blocks features...")
    spectral_test_blocks_results = FeatureExtract.spectral_test_on_blocks_batch(binary_data_list, block_size=128)
    for i, result in enumerate(spectral_test_blocks_results):
        if result is not None:
            features_list[i].update({
                f"spectral_test_blocks_{key}": value.tolist() if isinstance(value, torch.Tensor) else value 
                for key, value in result.items()
            })
        else:
            features_list[i]['spectral_test_blocks_error'] = 'Failed to compute spectral test on blocks.'

    return features_list

def process_files_batch(file_paths, labels):
    binary_data_list = []
    valid_indices = []
    for idx, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r') as file:
                hex_data = file.read().strip()
                binary_data = hex_to_binary(hex_data)
                if binary_data:
                    binary_data_list.append(binary_data)
                    valid_indices.append(idx)
                else:
                    binary_data_list.append(None)
        except Exception as e:
            binary_data_list.append(None)

    print(f"Processing batch of {len(file_paths)} files...")
    features_list = extract_features_batch(binary_data_list)
    for i, features in enumerate(features_list):
        if binary_data_list[i]:
            features['label'] = labels[i]
        else:
            features['label'] = 'error'
            features['file_error'] = 'Failed to read or convert file.'

    return features_list

def load_data(dataset_folder, batch_size=32, max_workers=None):
    data = []
    file_paths = []
    labels = []
    
    folder_list = os.listdir(dataset_folder)
    for label in tqdm(folder_list, desc="Processing folders"):
        folder_path = os.path.join(dataset_folder, label)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                file_paths.append(file_path)
                labels.append(label)
    
    batches = [(file_paths[i:i + batch_size], labels[i:i + batch_size]) 
               for i in range(0, len(file_paths), batch_size)]
    
    # Limit the number of workers to prevent memory issues
    if not max_workers:
        max_workers = min(multiprocessing.cpu_count(), 4)  # Adjust based on your system
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_files_batch, batch_fp, batch_lbl) for batch_fp, batch_lbl in batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                batch_data = future.result()
                data.extend(batch_data)
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    return pd.DataFrame(data)

def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    import torch  # Import inside the main block to avoid issues in child processes

    dataset_folder = "dataset"  # Replace with your dataset folder path
    output_file = "analysis_results.csv"  # Output CSV file
    
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear CUDA cache if using GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    df = load_data(dataset_folder, batch_size=32, max_workers=4)  # Adjust max_workers as needed
    save_to_csv(df, output_file)

    print(f"Feature extraction completed. Results saved to {output_file}.")