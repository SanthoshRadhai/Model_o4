import torch
import numpy as np
from math import floor, exp
import logging
from scipy.special import erfc
from scipy.stats import skew, kurtosis
import itertools
from typing import Any, List, Dict, Union, Tuple
import torch
from math import floor, exp
from scipy.special import gammaincc
from scipy import stats
from scipy.stats import entropy
from copy import copy 
import pandas as pd
from numba import njit
import multiprocessing
from functools import partial
from numpy import zeros as zeros
from numpy import dot as dot
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')




class FeatureExtract:

    @staticmethod
    def monobit_test_batch(binary_data_list: List[str], verbose=False) -> List[Dict[str, Union[float, int, bool]]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []

        # Convert list of binary strings to tensor
        lengths = torch.tensor([len(data) for data in binary_data_list], dtype=torch.float32, device=device)
        max_length = int(max(lengths))
        padded_data = []

        for data in binary_data_list:
            # Pad each binary string to max_length with '0's
            padded = data.ljust(max_length, '0')
            padded_data.append([1 if bit == '1' else -1 for bit in padded])

        # Create tensor of shape (batch_size, max_length)
        binary_tensor = torch.tensor(padded_data, dtype=torch.float32, device=device)
        counts = torch.sum(binary_tensor, dim=1)

        sObs = counts / torch.sqrt(lengths)
        p_values = torch.erfc(torch.abs(sObs) / torch.sqrt(torch.tensor(2.0, device=device)))

        counts_of_0s = torch.sum(binary_tensor == -1, dim=1).cpu().numpy()
        counts_of_1s = torch.sum(binary_tensor == 1, dim=1).cpu().numpy()

        proportions_of_0s = counts_of_0s / lengths.cpu().numpy()
        proportions_of_1s = counts_of_1s / lengths.cpu().numpy()
        normalized_S_n = counts.cpu().numpy() / lengths.cpu().numpy()

        sObs_cpu = sObs.cpu().numpy()
        p_values_cpu = p_values.cpu().numpy()
        counts_cpu = counts.cpu().numpy()
        lengths_cpu = lengths.cpu().numpy()

        for i in range(len(binary_data_list)):
            result = {
                "length_of_bit_string": int(lengths_cpu[i]),
                "count_of_0s": int(counts_of_0s[i]),
                "count_of_1s": int(counts_of_1s[i]),
                "proportion_of_0s": float(proportions_of_0s[i]),
                "proportion_of_1s": float(proportions_of_1s[i]),
                "S_n": float(counts_cpu[i]),
                "normalized_S_n": float(normalized_S_n[i]),
                "sObs": float(sObs_cpu[i]),
                "p_value": float(p_values_cpu[i]),
                "is_random": p_values_cpu[i] >= 0.01
            }
            if verbose:
                print('Frequency Test (Monobit Test) DEBUG BEGIN:')
                for key, value in result.items():
                    print(f"\t{key}:\t\t\t{value}")
                print('DEBUG END.')
            results.append(result)
        return results

    @staticmethod
    def extract_run_test_features_batch(
        binary_data_list: List[str],
        block_size: int = 128,
        verbose: bool = False
    ) -> List[Dict[str, Union[List, float]]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []

        for binary_data in binary_data_list:
            length_of_binary_data = len(binary_data)
            num_blocks = length_of_binary_data // block_size

            if num_blocks == 0:
                # Handle case where binary data is shorter than one block
                result = {
                    'p_values': [],
                    'block_results': [],
                    'pis': [],
                    'taus': 0.0,
                    'vObs': [],
                    "mean_p_value": 0.0,
                    "std_p_value": 0.0,
                    "pass_ratio": 0.0,
                    "skewness_p_values": 0.0,
                    "kurtosis_p_values": 0.0,
                    "median_p_value": 0.0,
                    "range_p_value": 0.0,
                    "max_p_value": 0.0,
                    "min_p_value": 0.0,
                    "vObs_variance": 0.0,
                    "pis_entropy": 0.0,
                    "fail_count": 0,
                    "weighted_avg_pi": 0.0,
                    "run_length_distribution": []
                }
                results.append(result)
                if verbose:
                    print("Binary data is shorter than one block. Skipping.")
                continue

            binary_array = [1 if bit == '1' else 0 for bit in binary_data[:num_blocks * block_size]]
            binary_tensor = torch.tensor(binary_array, dtype=torch.float32, device=device)
            blocks = binary_tensor.view(num_blocks, block_size)

            one_counts = torch.sum(blocks, dim=1)
            pis = one_counts / block_size
            taus = 2 / torch.sqrt(torch.tensor(block_size, dtype=torch.float32, device=device))

            # Convert transitions to float
            transitions = torch.sum(blocks[:, 1:] != blocks[:, :-1], dim=1).float() + 1.0
            abs_diff = torch.abs(pis - 0.5)
            block_results = abs_diff < taus

            with torch.no_grad():
                numerator = torch.abs(transitions - (2 * block_size * pis * (1 - pis)))
                # Correct denominator to ensure Tensor input
                sqrt_2_block_size = torch.sqrt(torch.tensor(2 * block_size, dtype=torch.float32, device=device))
                denominator = 2 * sqrt_2_block_size * pis * (1 - pis)
                # To avoid division by zero
                denominator = torch.where(denominator == 0, torch.tensor(1e-10, device=device), denominator)
                block_p_values = torch.erfc(numerator / denominator)

            block_p_values[~block_results] = 0.0
            block_results = block_p_values > 0.01

            p_values = block_p_values.cpu().numpy()
            block_results_list = block_results.cpu().numpy().astype(bool).tolist()
            pis_list = pis.cpu().numpy().tolist()
            taus_value = taus.item()
            vObs_list = transitions.cpu().numpy().tolist()
            pass_count = torch.sum(block_results).item()

            p_values_tensor = block_p_values
            mean_p_value = torch.mean(p_values_tensor).item()
            std_p_value = torch.std(p_values_tensor).item()
            pass_ratio = pass_count / num_blocks

            # Calculate additional aggregated features
            if len(p_values) > 0:
                skewness_p_values = skew(p_values)
                kurtosis_p_values = kurtosis(p_values)
            else:
                skewness_p_values = 0.0
                kurtosis_p_values = 0.0
            median_p_value = torch.median(p_values_tensor).item() if len(p_values_tensor) > 0 else 0.0
            range_p_value = torch.max(p_values_tensor).item() - torch.min(p_values_tensor).item() if len(p_values_tensor) > 0 else 0.0
            max_p_value = torch.max(p_values_tensor).item() if len(p_values_tensor) > 0 else 0.0
            min_p_value = torch.min(p_values_tensor).item() if len(p_values_tensor) > 0 else 0.0
            vObs_variance = torch.var(transitions).item() if len(transitions) > 0 else 0.0
            # Prevent log2(0) by adding a small epsilon
            pis_array = np.array(pis_list)
            pis_entropy = -np.sum(pis_array * np.log2(pis_array + 1e-12) + (1 - pis_array) * np.log2(1 - pis_array + 1e-12))
            fail_count = num_blocks - pass_count
            weighted_avg_pi = np.mean([pi for pi, result in zip(pis_list, block_results_list) if result]) if pass_count > 0 else 0.0

            run_lengths = [len(list(group)) for _, group in itertools.groupby(binary_data)]
            run_length_counts = np.bincount(run_lengths)
            run_length_distribution = run_length_counts / np.sum(run_length_counts) if np.sum(run_length_counts) > 0 else []

            if verbose:
                print("\n--- Aggregated Features ---")
                print(f"mean_p_value: {mean_p_value}")
                print(f"std_p_value: {std_p_value}")
                print(f"pass_ratio: {pass_ratio}")
                print(f"skewness_p_values: {skewness_p_values}")
                print(f"kurtosis_p_values: {kurtosis_p_values}")
                print(f"median_p_value: {median_p_value}")
                print(f"range_p_value: {range_p_value}")
                print(f"max_p_value: {max_p_value}")
                print(f"min_p_value: {min_p_value}")
                print(f"vObs_variance: {vObs_variance}")
                print(f"pis_entropy: {pis_entropy}")
                print(f"fail_count: {fail_count}")
                print(f"weighted_avg_pi: {weighted_avg_pi}")
                print(f"run_length_distribution: {run_length_distribution}")
                print('-' * 50)

            result = {
                "p_values": p_values.tolist(),
                "block_results": block_results_list,
                "pis": pis_list,
                "taus": taus_value,
                "vObs": vObs_list,
                "mean_p_value": mean_p_value,
                "std_p_value": std_p_value,
                "pass_ratio": pass_ratio,
                "skewness_p_values": skewness_p_values,
                "kurtosis_p_values": kurtosis_p_values,
                "median_p_value": median_p_value,
                "range_p_value": range_p_value,
                "max_p_value": max_p_value,
                "min_p_value": min_p_value,
                "vObs_variance": vObs_variance,
                "pis_entropy": pis_entropy,
                "fail_count": fail_count,
                "weighted_avg_pi": weighted_avg_pi,
                "run_length_distribution": run_length_distribution.tolist()
            }
            results.append(result)

        return results
    
    @staticmethod
    def binary_matrix_rank_test_batch_16(binary_data_list: List[str], verbose: bool = False) -> List[Dict[str, object]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []
        block_size_tuple = (16, 16)

        for binary_data in binary_data_list:
            length_of_binary_data = len(binary_data)
            rows, cols = block_size_tuple
            block_size = rows * cols
            num_blocks = length_of_binary_data // block_size

            if num_blocks == 0:
                sample_result = {
                    "p_value": None,
                    "result": False,
                    "num_blocks": 0,
                    "max_ranks": None,
                    "pi": None,
                    "xObs": None,
                    "block_ranks": None,
                    "error": f"Not enough data for {rows}x{cols} blocks."
                }
                results.append(sample_result)
                continue

            binary_array = [1 if bit == '1' else 0 for bit in binary_data[:num_blocks * block_size]]
            binary_tensor = torch.tensor(binary_array, dtype=torch.float32, device=device)
            blocks = binary_tensor.view(num_blocks, rows * cols)

            block_ranks_list = []
            bit_variances = []
            block_entropies = []

            for block in blocks:
                block_matrix = block.view(rows, cols)
                rank = torch.linalg.matrix_rank(block_matrix).item()
                block_ranks_list.append(rank)

                block_flat = block_matrix.flatten()
                prob_1 = torch.mean(block_flat).item()
                entropy = -prob_1 * np.log2(prob_1 + 1e-12) - (1 - prob_1) * np.log2(1 - prob_1 + 1e-12) if 0 < prob_1 < 1 else 0
                block_entropies.append(entropy)
                bit_variances.append(torch.var(block_flat).item())

            max_ranks = [
                block_ranks_list.count(rows),
                block_ranks_list.count(rows - 1),
                num_blocks - block_ranks_list.count(rows) - block_ranks_list.count(rows - 1)
            ]

            pi = [1.0, 0.0, 0.0]
            for x in range(1, rows + 1):
                pi[0] *= 1 - (1.0 / (2 ** x))
            pi[1] = 2 * pi[0]
            pi[2] = 1 - pi[0] - pi[1]

            max_ranks_tensor = torch.tensor(max_ranks, dtype=torch.float32, device=device)
            pi_tensor = torch.tensor(pi, dtype=torch.float32, device=device)
            num_blocks_tensor = torch.tensor(num_blocks, dtype=torch.float32, device=device)
            xObs = torch.sum(((max_ranks_tensor - pi_tensor * num_blocks_tensor) ** 2) / (pi_tensor * num_blocks_tensor)).item()
            p_value = np.exp(-xObs / 2)

            avg_rank = np.mean(block_ranks_list)
            std_rank = np.std(block_ranks_list)
            avg_entropy = np.mean(block_entropies)
            entropy_variance = np.var(block_entropies)
            avg_variance = np.mean(bit_variances)

            sample_result = {
                "p_value": p_value,
                "result": p_value >= 0.01,
                "num_blocks": num_blocks,
                "max_ranks": max_ranks,
                "pi": pi,
                "xObs": xObs,
                "block_ranks": block_ranks_list,
                "avg_rank": avg_rank,
                "std_rank": std_rank,
                "avg_entropy": avg_entropy,
                "entropy_variance": entropy_variance,
                "bit_variances": bit_variances,
                "error": None
            }

            if verbose:
                print(f"Additional Metrics for {rows}x{cols}:")
                print(f"Average Rank: {avg_rank}, Std Rank: {std_rank}")
                print(f"Average Entropy: {avg_entropy}, Entropy Variance: {entropy_variance}")
                print('-' * 50)

            results.append(sample_result)

        return results

    @staticmethod
    def binary_matrix_rank_test_batch_32(binary_data_list: List[str], verbose: bool = False) -> List[Dict[str, object]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []
        block_size_tuple = (32, 32)

        for binary_data in binary_data_list:
            length_of_binary_data = len(binary_data)
            rows, cols = block_size_tuple
            block_size = rows * cols
            num_blocks = length_of_binary_data // block_size

            if num_blocks == 0:
                sample_result = {
                    "p_value": None,
                    "result": False,
                    "num_blocks": 0,
                    "max_ranks": None,
                    "pi": None,
                    "xObs": None,
                    "block_ranks": None,
                    "error": f"Not enough data for {rows}x{cols} blocks."
                }
                results.append(sample_result)
                continue

            binary_array = [1 if bit == '1' else 0 for bit in binary_data[:num_blocks * block_size]]
            binary_tensor = torch.tensor(binary_array, dtype=torch.float32, device=device)
            blocks = binary_tensor.view(num_blocks, rows * cols)

            block_ranks_list = []
            bit_variances = []
            block_entropies = []

            for block in blocks:
                block_matrix = block.view(rows, cols)
                rank = torch.linalg.matrix_rank(block_matrix).item()
                block_ranks_list.append(rank)

                block_flat = block_matrix.flatten()
                prob_1 = torch.mean(block_flat).item()
                entropy = -prob_1 * np.log2(prob_1 + 1e-12) - (1 - prob_1) * np.log2(1 - prob_1 + 1e-12) if 0 < prob_1 < 1 else 0
                block_entropies.append(entropy)
                bit_variances.append(torch.var(block_flat).item())

            max_ranks = [
                block_ranks_list.count(rows),
                block_ranks_list.count(rows - 1),
                num_blocks - block_ranks_list.count(rows) - block_ranks_list.count(rows - 1)
            ]

            pi = [1.0, 0.0, 0.0]
            for x in range(1, rows + 1):
                pi[0] *= 1 - (1.0 / (2 ** x))
            pi[1] = 2 * pi[0]
            pi[2] = 1 - pi[0] - pi[1]

            max_ranks_tensor = torch.tensor(max_ranks, dtype=torch.float32, device=device)
            pi_tensor = torch.tensor(pi, dtype=torch.float32, device=device)
            num_blocks_tensor = torch.tensor(num_blocks, dtype=torch.float32, device=device)
            xObs = torch.sum(((max_ranks_tensor - pi_tensor * num_blocks_tensor) ** 2) / (pi_tensor * num_blocks_tensor)).item()
            p_value = np.exp(-xObs / 2)

            avg_rank = np.mean(block_ranks_list)
            std_rank = np.std(block_ranks_list)
            avg_entropy = np.mean(block_entropies)
            entropy_variance = np.var(block_entropies)
            avg_variance = np.mean(bit_variances)

            sample_result = {
                "p_value": p_value,
                "result": p_value >= 0.01,
                "num_blocks": num_blocks,
                "max_ranks": max_ranks,
                "pi": pi,
                "xObs": xObs,
                "block_ranks": block_ranks_list,
                "avg_rank": avg_rank,
                "std_rank": std_rank,
                "avg_entropy": avg_entropy,
                "entropy_variance": entropy_variance,
                "bit_variances": bit_variances,
                "error": None
            }

            if verbose:
                print(f"Additional Metrics for {rows}x{cols}:")
                print(f"Average Rank: {avg_rank}, Std Rank: {std_rank}")
                print(f"Average Entropy: {avg_entropy}, Entropy Variance: {entropy_variance}")
                print('-' * 50)

            results.append(sample_result)

        return results


    @staticmethod
    def spectral_test_batch(binary_data_list: List[str], verbose=False) -> List[Dict[str, Union[float, int, bool, List[float]]]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []

        max_length = max(len(data) for data in binary_data_list)
        processed_data = []

        # Prepare data for batch processing
        for binary_data in binary_data_list:
            # Validate input
            if not all(c in '01' for c in binary_data):
                raise ValueError("Input must be a binary string containing only '0' and '1'.")
            length = len(binary_data)
            if length < 2:
                raise ValueError("Binary data is too short for the spectral test.")

            # Pad sequences to max_length
            padded_data = binary_data.ljust(max_length, '0')
            sequence = [1 if bit == '1' else -1 for bit in padded_data]
            processed_data.append(sequence)

        # Convert to tensor
        sequences = torch.tensor(processed_data, dtype=torch.float32, device=device)
        lengths = torch.tensor([len(data) for data in binary_data_list], dtype=torch.float32, device=device)

        # Perform FFT
        spectral = torch.fft.fft(sequences)
        slice_sizes = (lengths // 2).long()

        # Compute modulus for each sample
        modulus_list = []
        for i in range(len(binary_data_list)):
            modulus = torch.abs(spectral[i, :slice_sizes[i]])
            modulus_list.append(modulus)

        # Compute tau, n0, n1, d, p_value for each sample
        for i in range(len(binary_data_list)):
            length = lengths[i]
            modulus = modulus_list[i]
            tau = torch.sqrt(torch.log(torch.tensor(1 / 0.05, device=device)) * length)
            n0 = 0.95 * (length / 2)
            n1 = torch.sum(modulus < tau).item()
            d = (n1 - n0) / torch.sqrt((length * 0.95 * 0.05) / 4)
            
            # Corrected p_value calculation
            p_value = erfc(torch.abs(d).cpu().item() / np.sqrt(2.0))
            
            result = {
                "p_value": p_value,
                "result": p_value >= 0.01,
                "length_of_binary_data": int(length.item()),
                "tau": tau.cpu().item(),
                "n0": n0,
                "n1": n1,
                "d": d.cpu().item(),
                # Only include modulus if verbose is True to save memory
                "modulus": modulus.cpu().tolist() if verbose else None
            }
            if verbose:
                print("Discrete Fourier Transform (Spectral) Test DEBUG:")
                for key, value in result.items():
                    print(f"\t{key}: {value}")
                print("-" * 50)
            results.append(result)

        return results
    
    @staticmethod
    def spectral_test_on_blocks_batch(binary_data_list: List[str], block_size: int, verbose=False) -> List[Dict[str, Union[float, int, bool, List[float]]]]:
        results = []
        for binary_data in binary_data_list:
            length_of_binary_data = len(binary_data)
            if length_of_binary_data < block_size:
                # Skip short sequences
                continue

            num_blocks = length_of_binary_data // block_size
            p_values = []
            for i in range(num_blocks):
                block = binary_data[i * block_size : (i + 1) * block_size]
                # Use the single spectral test method on each block
                block_result = FeatureExtract.spectral_test(block)
                p_values.append(block_result['p_value'])

            p_values_tensor = torch.tensor(p_values, dtype=torch.float32)
            unified_summary = {
                "unified_average_p_value": torch.mean(p_values_tensor).item(),
                "unified_standard_deviation_p_value": torch.std(p_values_tensor).item(),
                "unified_min_p_value": torch.min(p_values_tensor).item(),
                "unified_max_p_value": torch.max(p_values_tensor).item(),
                "unified_range_p_value": (torch.max(p_values_tensor) - torch.min(p_values_tensor)).item(),
                "unified_median_p_value": torch.median(p_values_tensor).item(),
                "unified_variance_p_value": torch.var(p_values_tensor).item(),
                "unified_q1": torch.quantile(p_values_tensor, 0.25).item(),
                "unified_q3": torch.quantile(p_values_tensor, 0.75).item(),
                "unified_iqr": (torch.quantile(p_values_tensor, 0.75) - torch.quantile(p_values_tensor, 0.25)).item(),
                "unified_skewness": stats.skew(p_values),
                "unified_kurtosis": stats.kurtosis(p_values),
                "unified_mode_p_value": float(stats.mode(p_values)[0]),
                "unified_overall_result": all(p >= 0.01 for p in p_values),
                "unified_num_blocks_passing": sum(p >= 0.01 for p in p_values)
            }
            result = {
                "p_values": p_values,
                **unified_summary
            }
            if verbose:
                print("Spectral Test on Blocks DEBUG:")
                for key, value in result.items():
                    print(f"\t{key}: {value}")
                print("-" * 50)
            results.append(result)
        return results
    
    @staticmethod
    def spectral_test(binary_data: str, verbose: bool = False) -> Dict[str, Union[float, int, bool, List[float]]]:
        """
        Performs the Discrete Fourier Transform (Spectral) Test on a single binary string.

        Args:
            binary_data (str): Binary string containing only '0' and '1'.
            verbose (bool): If True, prints debug information.

        Returns:
            Dict[str, Union[float, int, bool, List[float]]]: Results of the spectral test.
        """
        # Input Validation
        if not all(c in '01' for c in binary_data):
            raise ValueError("Input must be a binary string containing only '0' and '1'.")
        
        length_of_binary_data = len(binary_data)
        if length_of_binary_data < 2:
            raise ValueError("Binary data is too short for the spectral test.")

        # Convert binary string to tensor: '1' -> 1, '0' -> -1
        sequence = torch.tensor([1.0 if bit == '1' else -1.0 for bit in binary_data], dtype=torch.float32)
        
        # Perform FFT
        spectral = torch.fft.fft(sequence)
        slice_size = length_of_binary_data // 2
        modulus = torch.abs(spectral[:slice_size])

        # Precompute constants
        log_const = np.log(1 / 0.05)  # Precompute log(20) ≈ 3.0
        tau = torch.sqrt(torch.tensor(log_const * length_of_binary_data, dtype=torch.float32))

        n0 = 0.95 * slice_size

        # Count how many modulus values are below tau
        n1 = torch.sum(modulus < tau).item()

        # Compute d
        variance = (slice_size * 0.95 * 0.05) / 4
        d = (n1 - n0) / torch.sqrt(torch.tensor(variance, dtype=torch.float32))

        # Compute p-value using complementary error function
        p_value = erfc(torch.abs(d).item() / np.sqrt(2.0))

        results = {
            "p_value": p_value,
            "result": p_value >= 0.01,
            "length_of_binary_data": length_of_binary_data,
            "tau": tau.item(),
            "n0": n0,
            "n1": n1,
            "d": d.item(),
            "modulus": modulus.tolist()
        }

        if verbose:
            print("Discrete Fourier Transform (Spectral) Test DEBUG:")
            for key, value in results.items():
                print(f"\t{key}: {value}")
            print("-" * 50)

        return results
    


    @staticmethod
    def serial_test_and_extract_features_batch(binary_data_list: List[str], verbose=False, pattern_length=16) -> List[Dict[str, Any]]:
        results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for binary_data in binary_data_list:
            try:
                length_of_binary_data = len(binary_data)
                extended_data = binary_data + binary_data[:pattern_length - 1]
                max_pattern_value = int('1' * (pattern_length + 1), 2)

                vobs_counts = []
                for m in range(pattern_length, pattern_length - 3, -1):
                    vobs = torch.zeros(2 ** m, dtype=torch.float32, device=device)
                    for i in range(length_of_binary_data):
                        pattern = extended_data[i:i + m]
                        index = int(pattern, 2)
                        vobs[index] += 1
                    vobs_counts.append(vobs)

                sums = torch.zeros(3, dtype=torch.float32, device=device)
                for i, vobs in enumerate(vobs_counts):
                    sums[i] = torch.sum(vobs ** 2)
                    sums[i] = (sums[i] * (2 ** (pattern_length - i)) / length_of_binary_data) - length_of_binary_data

                nabla_01 = sums[0] - sums[1]
                nabla_02 = sums[0] - 2.0 * sums[1] + sums[2]
                p_value_01 = gammaincc(2 ** (pattern_length - 1) / 2, nabla_01.cpu().numpy() / 2.0)
                p_value_02 = gammaincc(2 ** (pattern_length - 2) / 2, nabla_02.cpu().numpy() / 2.0)

                transitions = sum(1 for i in range(1, len(binary_data)) if binary_data[i] != binary_data[i - 1])
                prob_0 = binary_data.count('0') / length_of_binary_data
                prob_1 = binary_data.count('1') / length_of_binary_data
                entropy_value = -prob_0 * np.log2(prob_0 + 1e-12) - prob_1 * np.log2(prob_1 + 1e-12)

                if verbose:
                    print('Serial Test DEBUG BEGIN:')
                    print(f"\tLength of input:\t{length_of_binary_data}")
                    print(f'\tPsi values:\t\t{sums}')
                    print(f'\tNabla values:\t\t{nabla_01}, {nabla_02}')
                    print(f'\tP-Value 01:\t\t{p_value_01}')
                    print(f'\tP-Value 02:\t\t{p_value_02}')
                    print('DEBUG END.')

                result = {
                    'p_value1': p_value_01,
                    'result1': p_value_01 >= 0.01,
                    'p_value2': p_value_02,
                    'result2': p_value_02 >= 0.01,
                    'nabla1': nabla_01.cpu().item(),
                    'nabla2': nabla_02.cpu().item(),
                    'sums': sums.cpu().tolist(),  # Move sums to CPU before converting to list
                    'transitions': transitions,
                    'entropy': entropy_value,
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing binary data: {binary_data[:50]}...: {e}")

        return results
        


    @staticmethod
    def approximate_entropy_test_batch(binary_data_list: List[str], verbose=False, pattern_length=8) -> List[Dict[str, Any]]:
        results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for binary_data in binary_data_list:
            length_of_binary_data = len(binary_data)
            extended_data = binary_data + binary_data[:pattern_length + 1]

            vobs_list = []
            for m in [pattern_length, pattern_length + 1]:
                vobs = torch.zeros(2 ** m, dtype=torch.float32, device=device)
                for i in range(length_of_binary_data):
                    pattern = extended_data[i:i + m]
                    index = int(pattern, 2)
                    vobs[index] += 1
                vobs_list.append(vobs)

            sums = torch.zeros(2, dtype=torch.float32)
            for i, vobs in enumerate(vobs_list):
                vobs = vobs[vobs > 0]
                sums[i] = torch.sum(vobs * torch.log(vobs / length_of_binary_data))
            sums /= length_of_binary_data

            ape = sums[0] - sums[1]
            xObs = 2.0 * length_of_binary_data * (torch.log(torch.tensor(2.0)) - ape)
            p_value = gammaincc(2 ** (pattern_length - 1), xObs / 2.0)

            # Additional statistics
            vobs_01 = vobs_list[0]
            vobs_02 = vobs_list[1]
            normalized_vobs_01 = vobs_01 / torch.sum(vobs_01)
            normalized_vobs_02 = vobs_02 / torch.sum(vobs_02)
            entropy_vobs_01 = -torch.sum(normalized_vobs_01 * torch.log(normalized_vobs_01 + 1e-12)).item()
            entropy_vobs_02 = -torch.sum(normalized_vobs_02 * torch.log(normalized_vobs_02 + 1e-12)).item()
            skewness_vobs_01 = stats.skew(vobs_01.cpu().numpy())
            skewness_vobs_02 = stats.skew(vobs_02.cpu().numpy())
            kurtosis_vobs_01 = stats.kurtosis(vobs_01.cpu().numpy())
            kurtosis_vobs_02 = stats.kurtosis(vobs_02.cpu().numpy())
            mean_vobs_01 = torch.mean(vobs_01).item()
            mean_vobs_02 = torch.mean(vobs_02).item()
            variance_vobs_01 = torch.var(vobs_01).item()
            variance_vobs_02 = torch.var(vobs_02).item()
            std_dev_vobs_01 = torch.std(vobs_01).item()
            std_dev_vobs_02 = torch.std(vobs_02).item()
            num_unique_patterns_01 = torch.sum(vobs_01 > 0).item()
            num_unique_patterns_02 = torch.sum(vobs_02 > 0).item()

            result = {
                'p_value': p_value,
                'result': p_value >= 0.01,
                'ape': ape.item(),
                'xObs': xObs.item(),
                'entropy_vobs_01': entropy_vobs_01,
                'entropy_vobs_02': entropy_vobs_02,
                'skewness_vobs_01': skewness_vobs_01,
                'skewness_vobs_02': skewness_vobs_02,
                'kurtosis_vobs_01': kurtosis_vobs_01,
                'kurtosis_vobs_02': kurtosis_vobs_02,
                'mean_vobs_01': mean_vobs_01,
                'mean_vobs_02': mean_vobs_02,
                'variance_vobs_01': variance_vobs_01,
                'variance_vobs_02': variance_vobs_02,
                'std_dev_vobs_01': std_dev_vobs_01,
                'std_dev_vobs_02': std_dev_vobs_02,
                'num_unique_patterns_01': num_unique_patterns_01,
                'num_unique_patterns_02': num_unique_patterns_02,
            }
            if verbose:
                print("Approximate Entropy Test DEBUG:")
                for key, value in result.items():
                    print(f"\t{key}: {value}")
                print("-" * 50)
            results.append(result)

        return results


    @staticmethod
    def cumulative_sums_test_batch(binary_data_list: List[str], mode=0, verbose=False) -> List[Dict[str, Any]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []

        # Find the maximum length to pad sequences
        max_length = max(len(data) for data in binary_data_list)

        # Prepare data for batch processing
        counts_list = []
        lengths_list = []
        for binary_data in binary_data_list:
            length_of_binary_data = len(binary_data)
            lengths_list.append(length_of_binary_data)
            # Count sequence
            sequence = [1 if bit == '1' else -1 for bit in binary_data]
            if mode != 0:
                sequence = sequence[::-1]
            # Pad sequences to maximum length
            padded_sequence = sequence + [0] * (max_length - length_of_binary_data)
            counts = torch.cumsum(torch.tensor(padded_sequence, dtype=torch.float32, device=device), dim=0)
            counts_list.append(counts)

        # Stack counts for batch processing
        counts_tensor = torch.stack(counts_list)
        abs_max_values = torch.max(torch.abs(counts_tensor), dim=1)[0]

        results = []
        for i in range(len(binary_data_list)):
            counts = counts_tensor[i][:lengths_list[i]]
            abs_max = abs_max_values[i].item()
            length_of_binary_data = lengths_list[i]
            # Calculate p-value components
            z = abs_max
            n = length_of_binary_data
            if z == 0:
                p_value = 1.0
            else:
                terms_one = []
                terms_two = []
                for k in range(int(np.floor(-n / z + 1)), int(np.floor(n / z + 1))):
                    terms_one.append(erfc((2 * k * z) / np.sqrt(2 * n)))
                    terms_two.append(erfc((2 * k * z - z) / np.sqrt(2 * n)))
                p_value = 1.0 - sum(terms_one) + sum(terms_two)

            # Additional statistics
            mean_counts = torch.mean(counts).item()
            var_counts = torch.var(counts).item()
            range_counts = (torch.max(counts) - torch.min(counts)).item()
            num_zero_crossings = (torch.diff((counts >= 0).float()) != 0).sum().item()
            max_positive_dev = torch.max(counts).item()
            max_negative_dev = torch.min(counts).item()
            mean_abs_dev = torch.mean(torch.abs(counts - mean_counts)).item()
            counts_np = counts.cpu().numpy()
            entropy_counts = entropy(np.histogram(counts_np, bins=10, density=True)[0] + 1e-12)
            skewness_counts = skew(counts_np)
            kurtosis_counts = kurtosis(counts_np)
            sum_abs_counts = torch.sum(torch.abs(counts)).item()
            trend_slope = (counts[-1] - counts[0]).item() / length_of_binary_data
            num_terms_greater_thresh = sum(1 for x in terms_one + terms_two if x > 0.5)
            # Collect results
            result = {
                'p_value': p_value,
                'result': p_value >= 0.01,
                'abs_max': abs_max,
                'terms_one': terms_one,
                'terms_two': terms_two,
                'counts': counts.cpu().tolist(),
                'mean_counts': mean_counts,
                'var_counts': var_counts,
                'range_counts': range_counts,
                'num_zero_crossings': num_zero_crossings,
                'max_positive_dev': max_positive_dev,
                'max_negative_dev': max_negative_dev,
                'mean_abs_dev': mean_abs_dev,
                'entropy_counts': entropy_counts,
                'skewness_counts': skewness_counts,
                'kurtosis_counts': kurtosis_counts,
                'sum_abs_counts': sum_abs_counts,
                'trend_slope': trend_slope,
                'num_terms_greater_thresh': num_terms_greater_thresh,
            }
            if verbose:
                print('Cumulative Sums Test DEBUG BEGIN:')
                for key, value in result.items():
                    print(f"\t{key}: {value}")
                print('DEBUG END.')
            results.append(result)

        return results
    

    @staticmethod
    def statistical_test_batch(binary_data_list: List[str], verbose=False) -> List[Dict[str, Union[float, bool, str, int]]]:
        results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for binary_data in binary_data_list:
            length_of_binary_data = len(binary_data)
            pattern_size = 5
            thresholds = [
                (387840, 6), (904960, 7), (2068480, 8), (4654080, 9), (10342400, 10),
                (22753280, 11), (49643520, 12), (107560960, 13), (231669760, 14),
                (496435200, 15), (1059061760, 16)
            ]
            for threshold, size in thresholds:
                if length_of_binary_data >= threshold:
                    pattern_size = size

            if pattern_size < 6 or pattern_size > 16:
                result = {
                    "p_value": -1.0,
                    "is_random": False,
                    "message": "Insufficient data for the test",
                    "length_of_binary_data": length_of_binary_data,
                    "pattern_size": pattern_size
                }
                results.append(result)
                continue

            num_ints = (1 << pattern_size) - 1
            vobs = torch.zeros(num_ints + 1, dtype=torch.int32, device=device)
            num_blocks = floor(length_of_binary_data / pattern_size)
            init_bits = 10 * (1 << pattern_size)
            test_bits = num_blocks - init_bits

            if test_bits < 0:
                result = {
                    "p_value": -1.0,
                    "is_random": False,
                    "message": "Not enough bits for testing",
                    "length_of_binary_data": length_of_binary_data,
                    "pattern_size": pattern_size
                }
                results.append(result)
                continue

            c = 0.7 - 0.8 / pattern_size + (4 + 32 / pattern_size) * pow(test_bits, -3 / pattern_size) / 15
            variance_values = [2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
            expected_values = [5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
                               10.170032, 11.168765, 12.168070, 13.167693, 14.167488,
                               15.167379]
            sigma = c * torch.sqrt(torch.tensor(variance_values[pattern_size - 6] / test_bits, device=device))
            expected = expected_values[pattern_size - 6]

            binary_array = [int(bit) for bit in binary_data]
            pattern_dict = {}
            cumsum = 0.0

            for i in range(init_bits):
                pattern = ''.join(map(str, binary_array[i * pattern_size:(i + 1) * pattern_size]))
                pattern_value = int(pattern, 2)
                pattern_dict[pattern_value] = i + 1

            for i in range(init_bits, num_blocks):
                pattern = ''.join(map(str, binary_array[i * pattern_size:(i + 1) * pattern_size]))
                pattern_value = int(pattern, 2)
                last_occurrence = pattern_dict.get(pattern_value, None)
                if last_occurrence is not None:
                    distance = i - last_occurrence
                    cumsum += np.log2(distance)
                pattern_dict[pattern_value] = i + 1

            phi = cumsum / test_bits
            stat = abs(phi - expected) / (np.sqrt(2) * sigma)
            p_value = erfc(stat)

            result = {
                "p_value": p_value,
                "is_random": p_value >= 0.01,
                "phi": phi,
                "stat": stat,
                "expected_phi": expected,
                "sigma": sigma,
                "cumsum": cumsum,
                "pattern_size": pattern_size,
                "num_blocks": num_blocks,
                "init_bits": init_bits,
                "test_bits": test_bits
            }

            if verbose:
                print("Maurer's Universal Statistical Test DEBUG:")
                for key, value in result.items():
                    print(f"\t{key}: {value}")
            results.append(result)

        return results
    

    @staticmethod
    def longest_one_block_test_batch(
        binary_data_list: List[str],
        verbose: bool = False
    ) -> List[Dict[str, Union[float, bool, List[float], int, str, None]]]:
        results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for binary_data in binary_data_list:
            # **Validate Binary Data**
            if not set(binary_data).issubset({'0', '1'}):
                result = {
                    'p_value': 0.0,
                    'is_random': False,
                    'error_message': 'Error: Binary data contains invalid characters.'
                }
                results.append(result)
                if verbose:
                    print(result['error_message'])
                continue

            length_of_binary_data = len(binary_data)
            if length_of_binary_data < 128:
                result = {
                    'p_value': 0.0,
                    'is_random': False,
                    'error_message': 'Error: Not enough data to run this test.'
                }
                results.append(result)
                if verbose:
                    print(result['error_message'])
                continue

            # Determine parameters based on the length
            if length_of_binary_data < 6272:
                k, m, v_values, pi_values = 3, 8, [1, 2, 3, 4], [0.2148, 0.3672, 0.2305, 0.1875]
            elif length_of_binary_data < 750000:
                k, m, v_values, pi_values = 5, 128, [4, 5, 6, 7, 8, 9], [0.1174, 0.2429, 0.2493, 0.1752, 0.1027, 0.1124]
            else:
                k, m, v_values, pi_values = 6, 10000, [10, 11, 12, 13, 14, 15, 16], [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

            number_of_blocks = length_of_binary_data // m
            if number_of_blocks == 0:
                result = {
                    'p_value': 0.0,
                    'is_random': False,
                    'error_message': 'Error: Insufficient blocks for the test.'
                }
                results.append(result)
                if verbose:
                    print(result['error_message'])
                continue

            # Prepare data
            adjusted_length = number_of_blocks * m
            data = binary_data[:adjusted_length]
            block_data = [data[i * m:(i + 1) * m] for i in range(number_of_blocks)]

            # Compute the max run of ones in each block
            max_runs = []
            for block in block_data:
                runs = [len(s) for s in ''.join(block).split('0') if s]
                max_runs.append(max(runs) if runs else 0)  # Handle blocks with no '1's

            # **Corrected Frequencies Initialization**
            frequencies = np.zeros(len(v_values) + 1)  # Extra bin for run > v_values[-1]

            for run in max_runs:
                if run < v_values[0]:
                    frequencies[0] += 1
                elif run > v_values[-1]:
                    frequencies[-1] += 1  # Assign to the overflow bin
                else:
                    # Assign run == v_values[i] to frequencies[i + 1]
                    try:
                        idx = v_values.index(run)
                        frequencies[idx + 1] += 1
                    except ValueError:
                        # If run is not in v_values, assign to overflow
                        frequencies[-1] += 1

            frequencies = torch.tensor(frequencies, device=device)
            pi_extended = pi_values + [1.0 - sum(pi_values)]  # Corrected to prevent division by zero
            pi_tensor = torch.tensor(pi_extended, device=device)

            # Ensure pi_tensor size matches frequencies
            if pi_tensor.size(0) != frequencies.size(0):
                raise ValueError(
                    f"Size mismatch: pi_values has size {pi_tensor.size(0)}, "
                    f"but frequencies has size {frequencies.size(0)}."
                )

            # Perform Chi-Squared Test
            # Handle cases where expected frequency is 0 to prevent division by zero
            with torch.no_grad():
                denominator = number_of_blocks * pi_tensor
                # Replace 0 denominators with a small value to prevent division by zero
                denominator = torch.where(denominator == 0, torch.tensor(1e-10, device=device), denominator)
                chi_squared = torch.sum(
                    (frequencies - number_of_blocks * pi_tensor) ** 2 / denominator
                ).item()
            p_value = gammaincc(len(v_values) / 2.0, chi_squared / 2.0)

            result = {
                'p_value': p_value,
                'is_random': p_value > 0.01,
                'frequencies': frequencies.cpu().tolist(),
                'expected_frequencies': (number_of_blocks * pi_tensor).cpu().tolist(),
                'block_count': number_of_blocks,
                'block_size': m,
                'error_message': None,
            }
            if verbose:
                print("Longest Run of Ones in a Block Test Details:")
                for key, value in result.items():
                    print(f"\t{key}: {value}")
            results.append(result)

        return results

    
    @staticmethod
    def block_frequency_multiple_sizes_batch(binary_data_list: List[str], block_sizes: List[int] = [8, 16, 32, 64, 128]) -> List[List[Dict[str, Union[str, float, int]]]]:
        results_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for binary_data in binary_data_list:
            length_of_bit_string = len(binary_data)
            data_results = []
            for block_size in block_sizes:
                if length_of_bit_string < block_size:
                    data_results.append({
                        "Block Size": block_size,
                        "P-Value": "N/A",
                        "Random": "Too Short",
                        "Number of Blocks": 0,
                        "Chi-Squared": "N/A",
                        "Proportion Sum": "N/A",
                        "Number of Ones": "N/A",
                        "Proportion of Ones": "N/A",
                        "Mean Proportion of Ones per Block": "N/A",
                        "Variance of Proportion of Ones per Block": "N/A",
                        "Standard Deviation of Proportion of Ones per Block": "N/A"
                    })
                    continue
                number_of_blocks = length_of_bit_string // block_size
                if number_of_blocks == 0:
                    data_results.append({
                        "Block Size": block_size,
                        "P-Value": "N/A",
                        "Random": "Insufficient Blocks",
                        "Number of Blocks": 0,
                        "Chi-Squared": "N/A",
                        "Proportion Sum": "N/A",
                        "Number of Ones": "N/A",
                        "Proportion of Ones": "N/A",
                        "Mean Proportion of Ones per Block": "N/A",
                        "Variance of Proportion of Ones per Block": "N/A",
                        "Standard Deviation of Proportion of Ones per Block": "N/A"
                    })
                    continue
                binary_tensor = torch.tensor([1 if bit == '1' else 0 for bit in binary_data[:number_of_blocks * block_size]], dtype=torch.float32, device=device)
                binary_tensor = binary_tensor.view(number_of_blocks, block_size)
                one_counts = torch.sum(binary_tensor, dim=1)
                proportions = one_counts / block_size
                proportion_sum = torch.sum((proportions - 0.5) ** 2).item()
                chi_squared = 4.0 * block_size * proportion_sum
                p_value = gammaincc(number_of_blocks / 2, chi_squared / 2)
                mean_proportion = torch.mean(proportions).item()
                variance_proportion = torch.var(proportions).item()
                std_dev_proportion = torch.std(proportions).item()
                proportion_of_ones = torch.sum(one_counts).item() / (number_of_blocks * block_size)
                normalized_number_of_ones = torch.sum(one_counts).item() / length_of_bit_string
                normalized_chi_squared = chi_squared / number_of_blocks
                normalized_proportion_sum = proportion_sum / number_of_blocks
                data_results.append({
                    "Block Size": block_size,
                    "P-Value": round(p_value, 6),
                    "Random": "Yes" if p_value >= 0.01 else "No",
                    "Number of Blocks": number_of_blocks,
                    "Chi-Squared": normalized_chi_squared,
                    "Proportion Sum": normalized_proportion_sum,
                    "Number of Ones Normalized": normalized_number_of_ones,
                    "Proportion of Ones": proportion_of_ones,
                    "Mean Proportion of Ones per Block": mean_proportion,
                    "Variance of Proportion of Ones per Block": variance_proportion,
                    "Standard Deviation of Proportion of Ones per Block": std_dev_proportion
                })
            results_list.append(data_results)
        return results_list
    
    

    @staticmethod
    def berlekamp_massey_algorithm(block_data: str) -> int:
        """
        An implementation of the Berlekamp-Massey Algorithm.
        Finds the linear complexity of a binary sequence.

        :param block_data: A binary string (e.g., "1100101...")
        :return: Linear complexity as an integer
        """
        n = len(block_data)
        c = np.zeros(n, dtype=np.int64)
        b = np.zeros(n, dtype=np.int64)
        c[0], b[0] = 1, 1
        l, m, i = 0, -1, 0
        int_data = [int(el) for el in block_data]

        while i < n:
            # Compute discrepancy d
            d = int_data[i]
            for j in range(1, l + 1):
                d ^= (c[j] & int_data[i - j])
            if d == 1:
                t = c.copy()  # Replaced copy(c) with c.copy()
                p = np.zeros(n, dtype=np.int64)
                for j in range(0, l + 1):
                    if b[j] == 1:
                        if (i - m + j) < n:
                            p[i - m + j] = 1
                c = (c + p) % 2
                if l <= 0.5 * i:
                    l = i + 1 - l
                    m = i
                    b = t
            i += 1

        return l

    @staticmethod
    def linear_complexity_test(binary_data: str, verbose: bool = False, block_size: int = 500) -> Dict[str, Union[float, bool, List[int], List[float], float, Dict, Tuple]]:
        """
        Computes the linear complexity features of a binary data string.

        :param binary_data: A binary string (e.g., "1100101...")
        :param verbose: If True, prints debug information
        :param block_size: Size of each block for processing
        :return: A dictionary containing various linear complexity features
        """
        length_of_binary_data = len(binary_data)
        degree_of_freedom = 6
        pi = torch.tensor([0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833], dtype=torch.float32)
        t2 = (block_size / 3.0 + 2.0 / 9) / (2 ** block_size)
        mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2
        number_of_blocks = length_of_binary_data // block_size

        if number_of_blocks > 1:
            blocks = [binary_data[i * block_size:(i + 1) * block_size] for i in range(number_of_blocks)]
            complexities = torch.tensor(
                [FeatureExtract.berlekamp_massey_algorithm(block) for block in blocks],
                dtype=torch.float32
            )
            t = ((-1) ** block_size) * (complexities - mean) + 2.0 / 9
            min_complexity = torch.min(complexities).item()
            max_complexity = torch.max(complexities).item()
            mean_complexity = torch.mean(complexities).item()
            std_complexity = torch.std(complexities).item()

            # Define finite bin edges suitable for your data distribution
            bin_edges = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=torch.float32)

            # Use torch.histc with finite min and max
            vg = torch.histc(t, bins=7, min=-2.5, max=2.5)  # bins=7 covers the same count as pi=7

            # No manual handling of underflow and overflow to keep vg size=7
            vg = vg.flip(dims=[0])
            im = ((vg - number_of_blocks * pi) ** 2) / (number_of_blocks * pi)
            xObs = torch.sum(im).item()
            p_value = gammaincc(degree_of_freedom / 2.0, xObs / 2.0).item()

            result = {
                'p_value': p_value,
                'complexities': complexities.tolist(),
                'min_complexity': min_complexity,
                'max_complexity': max_complexity,
                'mean_complexity': mean_complexity,
                'std_complexity': std_complexity,
                't_values': t.tolist(),
                'observed_frequencies': vg.tolist(),
                'interval_edges': bin_edges.tolist(),
                'intermediate_values': im.tolist(),
                'chi_squared_statistic': xObs,
                'precomputed_probabilities': pi.tolist(),
                'mean_t_value': torch.mean(t).item(),
                'range_t_value': (torch.min(t).item(), torch.max(t).item())
            }

            if verbose:
                logging.debug(f"Result: {result}")

            return result
        else:
            return {
                'p_value': -1.0,
                'complexities': None,
                't_values': None,
                'observed_frequencies': None,
                'interval_edges': None,
                'intermediate_values': None,
                'chi_squared_statistic': None,
                'min_complexity': None,
                'max_complexity': None,
                'mean_complexity': None,
                'std_complexity': None,
                'mean_t_value': None,
                'range_t_value': None
            }

    @staticmethod
    def linear_complexity_test_batch(serial_binary_data_list: List[str], block_size: int = 128) -> List[Dict[str, Union[float, bool, List[int], List[float], float, Dict, Tuple]]]:
        """
        Processes a batch of binary data strings to compute their linear complexity features.

        :param serial_binary_data_list: List of binary strings
        :param block_size: Size of each block for processing
        :return: List of dictionaries containing linear complexity features for each binary string
        """
        results_list = []
        for idx, binary_data in enumerate(serial_binary_data_list):
            if binary_data:
                result = FeatureExtract.linear_complexity_test(binary_data, verbose=False, block_size=block_size)
                results_list.append(result)
                if idx % 100 == 0 and idx > 0:
                    logging.debug(f"Processed {idx} / {len(serial_binary_data_list)} binary data strings.")
            else:
                result = {
                    'p_value': None,
                    'complexities': None,
                    't_values': None,
                    'observed_frequencies': None,
                    'interval_edges': None,
                    'intermediate_values': None,
                    'chi_squared_statistic': None,
                    'min_complexity': None,
                    'max_complexity': None,
                    'mean_complexity': None,
                    'std_complexity': None,
                    'mean_t_value': None,
                    'range_t_value': None,
                    'error': 'Invalid or empty binary data.'
                }
                results_list.append(result)
                logging.warning(f"Binary data at index {idx} is invalid or empty.")
        return results_list


    