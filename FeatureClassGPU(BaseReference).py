import time
start_time = time.time()

import binascii
import itertools
from math import fabs as fabs, log
from math import floor as floor
from math import sqrt as sqrt
from typing import Dict, List, Tuple, Union, get_type_hints
from numpy import zeros as zeros
from math import exp as exp
from numpy import histogram as histogram
from numpy import dot as dot
from copy import copy as copy
from scipy.stats import norm
from numpy import array as array
from scipy import stats
from typing import Dict, Tuple, Any
from numpy import zeros, mean, std, array, var, sum, cumsum, diff
from scipy.stats import norm, entropy, skew, kurtosis

from scipy.special import erfc as erfc
from scipy.special import gammaincc
import numpy as np

import torch

import inspect
import typing



class FeatureExtract:
    @staticmethod
    def monobit_test(binary_data: str, verbose=False) -> Dict[str, Union[float, int, bool]]:
        length_of_bit_string = len(binary_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        binary_tensor = torch.tensor([1 if bit == '1' else -1 for bit in binary_data], dtype=torch.float32, device=device)
        count = torch.sum(binary_tensor)
        sObs = count / torch.sqrt(torch.tensor(length_of_bit_string, dtype=torch.float32, device=device))
        p_value = erfc(torch.abs(sObs).cpu().item() / torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device)).cpu().item())
        count_of_0s = (binary_tensor == -1).sum().item()
        count_of_1s = (binary_tensor == 1).sum().item()
        proportion_of_0s = count_of_0s / length_of_bit_string
        proportion_of_1s = count_of_1s / length_of_bit_string
        normalized_S_n = count.item() / length_of_bit_string

        result = {
            "length_of_bit_string": length_of_bit_string,
            "count_of_0s": count_of_0s,
            "count_of_1s": count_of_1s,
            "proportion_of_0s": proportion_of_0s,
            "proportion_of_1s": proportion_of_1s,
            "S_n": count.item(),
            "normalized_S_n": normalized_S_n,
            "sObs": sObs.cpu().item(),
            "p_value": p_value,
            "is_random": p_value >= 0.01
        }

        if verbose:
            print('Frequency Test (Monobit Test) DEBUG BEGIN:')
            for key, value in result.items():
                print(f"\t{key}:\t\t\t{value}")
            print('DEBUG END.')

        return result
    

    # @staticmethod
    # def block_frequency_multiple_sizes(binary_data: str, block_sizes: List[int] = [8, 16, 32, 64, 128]) -> List[Dict[str, Union[str, float, int]]]:
    #     """
    #     Perform the Block Frequency Test for multiple block sizes to evaluate the randomness of the binary sequence.
        
    #     :param      binary_data: str         The binary sequence to be tested (e.g., "101010...").
    #     :param      block_sizes: list        A list of block sizes to test (default is [8, 16, 32, 64, 128]).
    #     :return:    list                     A list of dictionaries containing block size, p-value, and test result.
    #     """
    #     # Results list to store results for each block size
    #     results = []
        
    #     # Perform the test for each block size
    #     for block_size in block_sizes:
    #         length_of_bit_string = len(binary_data)
            
    #         # Ensure block size doesn't exceed the length of the binary string
    #         if length_of_bit_string < block_size:
    #             results.append({
    #                 "Block Size": block_size,
    #                 "P-Value": "N/A",
    #                 "Random": "Too Short",
    #                 "Number of Blocks": 0,  # Suitable for ML
    #                 "Chi-Squared": "N/A",  # Suitable for ML
    #                 "Proportion Sum": "N/A",  # Suitable for ML
    #                 "Number of Ones": "N/A",  # Suitable for ML
    #                 "Proportion of Ones": "N/A",  # Suitable for ML
    #                 "Mean Proportion of Ones per Block": "N/A",  # Suitable for ML
    #                 "Variance of Proportion of Ones per Block": "N/A",  # Suitable for ML
    #                 "Standard Deviation of Proportion of Ones per Block": "N/A"  # Suitable for ML
    #             })
    #             continue

    #         # Compute the number of blocks
    #         number_of_blocks = floor(length_of_bit_string / block_size)

    #         if number_of_blocks == 0:
    #             results.append({
    #                 "Block Size": block_size,
    #                 "P-Value": "N/A",
    #                 "Random": "Insufficient Blocks",
    #                 "Number of Blocks": 0,  # Suitable for ML
    #                 "Chi-Squared": "N/A",  # Suitable for ML
    #                 "Proportion Sum": "N/A",  # Suitable for ML
    #                 "Number of Ones": "N/A",  # Suitable for ML
    #                 "Proportion of Ones": "N/A",  # Suitable for ML
    #                 "Mean Proportion of Ones per Block": "N/A",  # Suitable for ML
    #                 "Variance of Proportion of Ones per Block": "N/A",  # Suitable for ML
    #                 "Standard Deviation of Proportion of Ones per Block": "N/A"  # Suitable for ML
    #             })
    #             continue

    #         # Initialize variables
    #         block_start = 0
    #         block_end = block_size
    #         proportion_sum = 0.0
    #         proportions = []
    #         total_ones = 0

    #         # Process each block
    #         for _ in range(number_of_blocks):
    #             # Extract block data
    #             block_data = binary_data[block_start:block_end]

    #             # Count the number of ones in the block
    #             one_count = sum(1 for bit in block_data if bit == '1')
    #             total_ones += one_count

    #             # Calculate the proportion of ones in the block
    #             pi = one_count / block_size
    #             proportions.append(pi)

    #             # Compute the squared deviation and add to the proportion sum
    #             proportion_sum += pow(pi - 0.5, 2.0)

    #             # Move to the next block
    #             block_start += block_size
    #             block_end += block_size

    #         # Compute the test statistic: 4M Σ(πi - ½)^2
    #         chi_squared = 4.0 * block_size * proportion_sum

    #         # Compute the p-value using the complementary incomplete gamma function
    #         p_value = gammaincc(number_of_blocks / 2, chi_squared / 2)

    #         # Calculate additional metrics
    #         mean_proportion = np.mean(proportions)
    #         variance_proportion = np.var(proportions)
    #         std_dev_proportion = np.std(proportions)
    #         proportion_of_ones = total_ones / length_of_bit_string

    #         # Normalize metrics
    #         normalized_number_of_ones = total_ones / length_of_bit_string
    #         normalized_chi_squared = chi_squared / number_of_blocks
    #         normalized_proportion_sum = proportion_sum / number_of_blocks

    #         # Store results in a dictionary
    #         results.append({
    #             "Block Size": block_size,
    #             "P-Value": round(p_value, 6),
    #             "Random": "Yes" if p_value >= 0.01 else "No",
    #             "Number of Blocks": number_of_blocks,  # Suitable for ML
    #             "Chi-Squared": normalized_chi_squared,  # Suitable for ML
    #             "Proportion Sum": normalized_proportion_sum,  # Suitable for ML
    #             "Number of Ones Normalized": normalized_number_of_ones,  # Suitable for ML
    #             "Proportion of Ones": proportion_of_ones,  # Suitable for ML
    #             "Mean Proportion of Ones per Block": mean_proportion,  # Suitable for ML
    #             "Variance of Proportion of Ones per Block": variance_proportion,  # Suitable for ML
    #             "Standard Deviation of Proportion of Ones per Block": std_dev_proportion  # Suitable for ML
    #         })

    #     return results


    @staticmethod
    def extract_run_test_features(binary_data: str, block_size: int = 128, verbose=False) -> Dict[str, Union[List, float]]:
        length_of_binary_data = len(binary_data)
        num_blocks = length_of_binary_data // block_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        binary_tensor = torch.tensor([1 if bit == '1' else 0 for bit in binary_data], dtype=torch.float32, device=device)
        blocks = binary_tensor.unfold(0, block_size, block_size)

        one_counts = torch.sum(blocks, dim=1)
        pis = one_counts / block_size
        taus = 2 / torch.sqrt(torch.tensor(block_size, dtype=torch.float32, device=device))

        transitions = torch.sum(blocks[:, 1:] != blocks[:, :-1], dim=1) + 1
        abs_diff = torch.abs(pis - 0.5)
        block_results = abs_diff < taus
        block_p_values = erfc((torch.abs(transitions - (2 * block_size * pis * (1 - pis))) /
                            (2 * torch.sqrt(torch.tensor(2.0 * block_size, dtype=torch.float32, device=device)) * pis * (1 - pis))).cpu().numpy())
        block_p_values = torch.tensor(block_p_values, dtype=torch.float32, device=device)
        block_p_values[~block_results] = 0.0
        block_results = block_p_values > 0.01

        p_values = block_p_values.cpu().tolist()
        block_results_list = block_results.cpu().tolist()
        pis_list = pis.cpu().tolist()
        taus_list = taus.cpu().tolist()
        vObs_list = transitions.cpu().tolist()
        pass_count = torch.sum(block_results).item()

        if verbose:
            for i, block in enumerate(blocks):
                print(f"Block: {block.cpu().numpy()}")
                print(f"\tPI: {pis_list[i]}, Tau: {taus_list[i]}")
                print(f"\tObserved Transitions (vObs): {vObs_list[i]}")
                print(f"\tP-Value: {p_values[i]}, Result: {block_results_list[i]}")

        p_values_tensor = torch.tensor(p_values, dtype=torch.float32, device=device)
        mean_p_value = torch.mean(p_values_tensor).item()
        std_p_value = torch.std(p_values_tensor).item()
        pass_ratio = pass_count / num_blocks
        skewness_p_values = skew(p_values)
        kurtosis_p_values = kurtosis(p_values)
        median_p_value = torch.median(p_values_tensor).item()
        range_p_value = torch.max(p_values_tensor).item() - torch.min(p_values_tensor).item()
        max_p_value = torch.max(p_values_tensor).item()
        min_p_value = torch.min(p_values_tensor).item()
        vObs_variance = torch.var(torch.tensor(vObs_list, dtype=torch.float32, device=device)).item()
        pis_entropy = -np.sum(np.fromiter((pi * torch.log2(torch.tensor(pi, dtype=torch.float32, device=device)).item() +
                                        (1 - pi) * torch.log2(torch.tensor(1 - pi, dtype=torch.float32, device=device)).item()
                                        for pi in pis_list if pi > 0 and pi < 1), dtype=np.float32))
        fail_count = num_blocks - pass_count
        weighted_avg_pi = torch.mean(torch.tensor([pi for pi, result in zip(pis_list, block_results_list) if result], dtype=torch.float32, device=device)).item()
        # relative_differences = torch.tensor(
                                #     [abs(pi - tau) / tau for pi, tau in zip(pis_list, taus_list)],
                                #     dtype=torch.float32,
                                #     device=device
                                # )
        # avg_relative_difference = torch.mean(relative_differences).item()
        run_lengths = [sum(1 for _ in group) for _, group in itertools.groupby(binary_data)]
        run_length_distribution = torch.histc(torch.tensor(run_lengths, dtype=torch.float32, device=device), bins=len(set(run_lengths)))

        aggregated_features = {
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
            # "avg_relative_difference": avg_relative_difference,
            "run_length_distribution": run_length_distribution.cpu().tolist()
        }

        if verbose:
            print("\n--- Aggregated Features ---")
            for key, value in aggregated_features.items():
                print(f"{key}: {value}")

        return {
            "p_values": p_values,
            "block_results": block_results_list,
            "pis": pis_list,
            "taus": taus_list,
            "vObs": vObs_list,
            "aggregated_features": aggregated_features
        }

    


    # @staticmethod
    # def extract_run_test_features(binary_data: str, block_size: int = 128, verbose=False) -> Dict[str, Union[List, float]]:
    #     length_of_binary_data = len(binary_data)
    #     num_blocks = length_of_binary_data // block_size
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     binary_tensor = torch.tensor([1 if bit == '1' else 0 for bit in binary_data], dtype=torch.float32, device=device)
    #     blocks = binary_tensor.unfold(0, block_size, block_size)

    #     one_counts = torch.sum(blocks, dim=1)
    #     pis = one_counts / block_size
    #     taus = 2 / torch.sqrt(torch.tensor(block_size, dtype=torch.float32, device=device))

    #     transitions = torch.sum(blocks[:, 1:] != blocks[:, :-1], dim=1) + 1
    #     abs_diff = torch.abs(pis - 0.5)
    #     block_results = abs_diff < taus
    #     block_p_values = erfc((torch.abs(transitions - (2 * block_size * pis * (1 - pis))) /
    #                         (2 * torch.sqrt(torch.tensor(2.0 * block_size, dtype=torch.float32, device=device)) * pis * (1 - pis))).cpu().numpy())
    #     block_p_values = torch.tensor(block_p_values, dtype=torch.float32, device=device)
    #     block_p_values[~block_results] = 0.0
    #     block_results = block_p_values > 0.01

    #     p_values = block_p_values.cpu().tolist()
    #     block_results_list = block_results.cpu().tolist()
    #     pis_list = pis.cpu().tolist()
    #     taus_list = taus.cpu().tolist()
    #     vObs_list = transitions.cpu().tolist()
    #     pass_count = torch.sum(block_results).item()

    #     if verbose:
    #         for i, block in enumerate(blocks):
    #             print(f"Block: {block.cpu().numpy()}")
    #             print(f"\tPI: {pis_list[i]}, Tau: {taus_list[i]}")
    #             print(f"\tObserved Transitions (vObs): {vObs_list[i]}")
    #             print(f"\tP-Value: {p_values[i]}, Result: {block_results_list[i]}")

    #     p_values_tensor = torch.tensor(p_values, dtype=torch.float32, device=device)
    #     mean_p_value = torch.mean(p_values_tensor).item()
    #     std_p_value = torch.std(p_values_tensor).item()
    #     pass_ratio = pass_count / num_blocks
    #     skewness_p_values = skew(p_values)
    #     kurtosis_p_values = kurtosis(p_values)
    #     median_p_value = torch.median(p_values_tensor).item()
    #     range_p_value = torch.max(p_values_tensor).item() - torch.min(p_values_tensor).item()
    #     max_p_value = torch.max(p_values_tensor).item()
    #     min_p_value = torch.min(p_values_tensor).item()
    #     vObs_variance = torch.var(torch.tensor(vObs_list, dtype=torch.float32, device=device)).item()
    #     pis_entropy = -np.sum(np.fromiter((pi * torch.log2(torch.tensor(pi, dtype=torch.float32, device=device)).item() +
    #                                     (1 - pi) * torch.log2(torch.tensor(1 - pi, dtype=torch.float32, device=device)).item()
    #                                     for pi in pis_list if pi > 0 and pi < 1), dtype=np.float32))
    #     fail_count = num_blocks - pass_count
    #     weighted_avg_pi = torch.mean(torch.tensor([pi for pi, result in zip(pis_list, block_results_list) if result], dtype=torch.float32, device=device)).item()
    #     relative_differences = torch.tensor([abs(pi - tau) / tau for pi, tau in zip(pis_list, taus_list)], dtype=torch.float32, device=device)
    #     avg_relative_difference = torch.mean(relative_differences).item()
    #     run_lengths = [sum(1 for _ in group) for _, group in itertools.groupby(binary_data)]
    #     run_length_distribution = torch.histc(torch.tensor(run_lengths, dtype=torch.float32, device=device), bins=len(set(run_lengths)))

    #     aggregated_features = {
    #         "mean_p_value": mean_p_value,
    #         "std_p_value": std_p_value,
    #         "pass_ratio": pass_ratio,
    #         "skewness_p_values": skewness_p_values,
    #         "kurtosis_p_values": kurtosis_p_values,
    #         "median_p_value": median_p_value,
    #         "range_p_value": range_p_value,
    #         "max_p_value": max_p_value,
    #         "min_p_value": min_p_value,
    #         "vObs_variance": vObs_variance,
    #         "pis_entropy": pis_entropy,
    #         "fail_count": fail_count,
    #         "weighted_avg_pi": weighted_avg_pi,
    #         "avg_relative_difference": avg_relative_difference,
    #         "run_length_distribution": run_length_distribution.cpu().tolist()
    #     }

    #     if verbose:
    #         print("\n--- Aggregated Features ---")
    #         for key, value in aggregated_features.items():
    #             print(f"{key}: {value}")

    #     return {
    #         "p_values": p_values,
    #         "block_results": block_results_list,
    #         "pis": pis_list,
    #         "taus": taus_list,
    #         "vObs": vObs_list,
    #         "aggregated_features": aggregated_features
    #     }
    

    @staticmethod
    def binary_matrix_rank_test(binary_data: str, verbose: bool = False, block_sizes: List[Tuple[int, int]] = [(32, 32), (16, 16)]) -> Dict[str, Dict[str, object]]:
        length_of_binary_data = len(binary_data)
        results = {}
        for rows, cols in block_sizes:
            block_size = rows * cols
            num_blocks = floor(length_of_binary_data / block_size)
            if num_blocks == 0:
                results[f"{rows}x{cols}"] = {
                    "p_value": None,
                    "result": False,
                    "num_blocks": 0,
                    "max_ranks": None,
                    "pi": None,
                    "xObs": None,
                    "block_ranks": None,
                    "error": f"Not enough data for {rows}x{cols} blocks."
                }
                continue
            max_ranks = [0, 0, 0]
            block_start = 0
            block_ranks_list = []
            bit_variances = []
            block_entropies = []
            for _ in range(num_blocks):
                block_data = binary_data[block_start:block_start + block_size]
                block_start += block_size
                block = torch.tensor([1 if bit == '1' else 0 for bit in block_data], dtype=torch.float32).reshape((rows, cols))
                rank = torch.linalg.matrix_rank(block).item()
                block_ranks_list.append(rank)
                block_flat = block.flatten()
                prob_1 = torch.mean(block_flat).item()
                entropy = -prob_1 * torch.log2(torch.tensor(prob_1)) - (1 - prob_1) * torch.log2(torch.tensor(1 - prob_1)) if 0 < prob_1 < 1 else 0
                block_entropies.append(entropy)
                bit_variances.append(torch.var(block_flat).item())
                if rank == rows:
                    max_ranks[0] += 1
                elif rank == rows - 1:
                    max_ranks[1] += 1
                else:
                    max_ranks[2] += 1
            pi = [1.0, 0.0, 0.0]
            for x in range(1, rows + 1):
                pi[0] *= 1 - (1.0 / (2 ** x))
            pi[1] = 2 * pi[0]
            pi[2] = 1 - pi[0] - pi[1]
            xObs = sum(pow((max_ranks[i] - pi[i] * num_blocks), 2.0) / (pi[i] * num_blocks) for i in range(3))
            p_value = exp(-xObs / 2)
            avg_rank = torch.mean(torch.tensor(block_ranks_list, dtype=torch.float32)).item()
            std_rank = torch.std(torch.tensor(block_ranks_list, dtype=torch.float32)).item()
            avg_entropy = torch.mean(torch.tensor(block_entropies, dtype=torch.float32)).item()
            entropy_variance = torch.var(torch.tensor(block_entropies, dtype=torch.float32)).item()
            avg_variance = torch.mean(torch.tensor(bit_variances, dtype=torch.float32)).item()
            results[f"{rows}x{cols}"] = {
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
        return results




    @staticmethod
    def spectral_test(binary_data: str, verbose=False) -> Dict[str, Union[float, int, bool, List[float]]]:
        if not all(c in '01' for c in binary_data):
            raise ValueError("Input must be a binary string containing only '0' and '1'.")
        
        length_of_binary_data = len(binary_data)
        if length_of_binary_data < 2:
            raise ValueError("Binary data is too short for the spectral test.")

        sequence = torch.tensor([1 if bit == '1' else -1 for bit in binary_data], dtype=torch.float32)
        spectral = torch.fft.fft(sequence)
        slice_size = length_of_binary_data // 2
        modulus = torch.abs(spectral[:slice_size])
        tau = torch.sqrt(torch.log(torch.tensor(1 / 0.05)) * length_of_binary_data)
        n0 = 0.95 * slice_size
        n1 = torch.sum(modulus < tau).item()
        d = (n1 - n0) / torch.sqrt(torch.tensor((slice_size * 0.95 * 0.05) / 4))
        p_value = erfc(torch.abs(d) / torch.sqrt(torch.tensor(2.0)))

        results = {
            "p_value": p_value.item(),
            "result": p_value.item() >= 0.01,
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
    def spectral_test_on_blocks(binary_data: str, block_size: int, verbose=False) -> Dict[str, Union[float, int, bool, List[float], Dict[str, float]]]:
        length_of_binary_data = len(binary_data)
        if length_of_binary_data < block_size:
            raise ValueError("Binary data is too short for the given block size.")

        block_results = {
            "block_index": [],
            "length_of_binary_data": [],
            "tau": [],
            "n0": [],
            "n1": [],
            "d": [],
            "modulus": [],
            "p_value": [],
            "result": []
        }

        p_values = []
        for i, start in enumerate(range(0, length_of_binary_data, block_size)):
            block = binary_data[start:start + block_size]
            if len(block) < block_size:
                continue
            result = FeatureExtract.spectral_test(block, verbose)
            block_results["block_index"].append(i)
            block_results["length_of_binary_data"].append(result["length_of_binary_data"])
            block_results["tau"].append(result["tau"])
            block_results["n0"].append(result["n0"])
            block_results["n1"].append(result["n1"])
            block_results["d"].append(result["d"])
            block_results["modulus"].append(result["modulus"])
            block_results["p_value"].append(result["p_value"])
            block_results["result"].append(result["result"])
            p_values.append(result["p_value"])

        p_values_tensor = torch.tensor(p_values, dtype=torch.float32)
        unified_summary = {
            "average_p_value": torch.mean(p_values_tensor).item(),
            "standard_deviation_p_value": torch.std(p_values_tensor).item(),
            "min_p_value": torch.min(p_values_tensor).item(),
            "max_p_value": torch.max(p_values_tensor).item(),
            "range_p_value": (torch.max(p_values_tensor) - torch.min(p_values_tensor)).item(),
            "median_p_value": torch.median(p_values_tensor).item(),
            "variance_p_value": torch.var(p_values_tensor).item(),
            "q1": torch.quantile(p_values_tensor, 0.25).item(),
            "q3": torch.quantile(p_values_tensor, 0.75).item(),
            "iqr": (torch.quantile(p_values_tensor, 0.75) - torch.quantile(p_values_tensor, 0.25)).item(),
            "skewness": stats.skew(p_values),
            "kurtosis": stats.kurtosis(p_values),
            "mode_p_value": stats.mode(p_values)[0][0] if isinstance(stats.mode(p_values)[0], np.ndarray) and len(stats.mode(p_values)[0]) > 0 else stats.mode(p_values)[0],
            "overall_result": all(p >= 0.01 for p in p_values),
            "num_blocks_passing": sum(p >= 0.01 for p in p_values)
        }

        block_results["unified_summary"] = unified_summary

        if verbose:
            print("Spectral Test on Blocks DEBUG:")
            for key, value in block_results.items():
                print(f"\t{key}: {value}")
            print("-" * 50)

        return block_results

    
    @staticmethod
    def linear_complexity_test(binary_data: str, verbose: bool = False, block_size: int = 500) -> Dict[str, Union[float, bool, List[int], List[float], List[int], List[float], float, Dict, Tuple]]:
        length_of_binary_data = len(binary_data)
        degree_of_freedom = 6
        pi = torch.tensor([0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833], dtype=torch.float32)
        t2 = (block_size / 3.0 + 2.0 / 9) / (2 ** block_size)
        mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2
        number_of_blocks = length_of_binary_data // block_size

        if number_of_blocks > 1:
            blocks = [binary_data[i * block_size:(i + 1) * block_size] for i in range(number_of_blocks)]
            complexities = torch.tensor([FeatureExtract.berlekamp_massey_algorithm(block) for block in blocks], dtype=torch.float32)
            t = (-1) ** block_size * (complexities - mean) + 2.0 / 9
            min_complexity = torch.min(complexities).item()
            max_complexity = torch.max(complexities).item()
            mean_complexity = torch.mean(complexities).item()
            std_complexity = torch.std(complexities).item()
            vg, bin_edges = torch.histogram(t, bins=torch.tensor([-float('inf'), -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, float('inf')], dtype=torch.float32))
            vg = vg.flip(dims=[0])
            im = ((vg - number_of_blocks * pi) ** 2) / (number_of_blocks * pi)
            xObs = torch.sum(im).item()

            result = {
                'p_value': torch.special.gammaincc(torch.tensor(degree_of_freedom / 2.0, dtype=torch.float32), torch.tensor(xObs / 2.0, dtype=torch.float32)).item(),
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



    @staticmethod    ###NO
    def berlekamp_massey_algorithm(block_data):
        """
        An implementation of the Berlekamp Massey Algorithm. Taken from Wikipedia [1]
        [1] - https://en.wikipedia.org/wiki/Berlekamp-Massey_algorithm
        The Berlekamp–Massey algorithm is an algorithm that will find the shortest linear feedback shift register (LFSR)
        for a given binary output sequence. The algorithm will also find the minimal polynomial of a linearly recurrent
        sequence in an arbitrary field. The field requirement means that the Berlekamp–Massey algorithm requires all
        non-zero elements to have a multiplicative inverse.
        :param block_data:
        :return:
        """
        n = len(block_data)
        c = zeros(n)
        b = zeros(n)
        c[0], b[0] = 1, 1
        l, m, i = 0, -1, 0
        int_data = [int(el) for el in block_data]
        while i < n:
            v = int_data[(i - l):i]
            v = v[::-1]
            cc = c[1:l + 1]
            d = (int_data[i] + dot(v, cc)) % 2
            if d == 1:
                temp = copy(c)
                p = zeros(n)
                for j in range(0, l):
                    if b[j] == 1:
                        p[j + i - m] = 1
                c = (c + p) % 2
                if l <= 0.5 * i:
                    l = i + 1 - l
                    m = i
                    b = temp
            i += 1
        return l
    
    @staticmethod
    def serial_test_and_extract_features(binary_data: str, verbose=False, pattern_length=16) -> Dict[str, Any]:
        length_of_binary_data = len(binary_data)
        binary_data += binary_data[:(pattern_length - 1)]
        max_pattern = '1' * (pattern_length + 1)
        vobs_01 = torch.zeros(int(max_pattern[:pattern_length], 2) + 1, dtype=torch.float32)
        vobs_02 = torch.zeros(int(max_pattern[:pattern_length - 1], 2) + 1, dtype=torch.float32)
        vobs_03 = torch.zeros(int(max_pattern[:pattern_length - 2], 2) + 1, dtype=torch.float32)
        
        for i in range(length_of_binary_data):
            vobs_01[int(binary_data[i:i + pattern_length], 2)] += 1
            vobs_02[int(binary_data[i:i + pattern_length - 1], 2)] += 1
            vobs_03[int(binary_data[i:i + pattern_length - 2], 2)] += 1
        
        vobs = [vobs_01, vobs_02, vobs_03]
        sums = torch.zeros(3, dtype=torch.float32)
        
        for i in range(3):
            sums[i] = torch.sum(vobs[i] ** 2)
            sums[i] = (sums[i] * (2 ** (pattern_length - i)) / length_of_binary_data) - length_of_binary_data
        
        nabla_01 = sums[0] - sums[1]
        nabla_02 = sums[0] - 2.0 * sums[1] + sums[2]
        
        p_value_01 = torch.special.gammaincc(torch.tensor(2 ** (pattern_length - 1) / 2, dtype=torch.float32), nabla_01 / 2.0).item()
        p_value_02 = torch.special.gammaincc(torch.tensor(2 ** (pattern_length - 2) / 2, dtype=torch.float32), nabla_02 / 2.0).item()
        
        transitions = torch.sum(torch.tensor([1 for i in range(1, len(binary_data)) if binary_data[i] != binary_data[i - 1]], dtype=torch.float32)).item()
        entropy_value = entropy([binary_data.count('0') / len(binary_data), binary_data.count('1') / len(binary_data)])
        
        if verbose:
            print('Serial Test DEBUG BEGIN:')
            print(f"\tLength of input:\t{length_of_binary_data}")
            print(f'\tPsi values:\t\t{sums}')
            print(f'\tNabla values:\t\t{nabla_01}, {nabla_02}')
            print(f'\tP-Value 01:\t\t{p_value_01}')
            print(f'\tP-Value 02:\t\t{p_value_02}')
            print('DEBUG END.')
        
        return {
            'p_value1': p_value_01,
            'result1': p_value_01 >= 0.01,
            'p_value2': p_value_02,
            'result2': p_value_02 >= 0.01,
            'nabla1': nabla_01.item(),
            'nabla2': nabla_02.item(),
            'sums': sums.tolist(),
            'transitions': transitions,
            'entropy': entropy_value,
        }
    
    @staticmethod
    def approximate_entropy_test(binary_data: str, verbose=False, pattern_length=8) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        length_of_binary_data = len(binary_data)
        binary_data += binary_data[:pattern_length + 1]
        max_pattern = '1' * (pattern_length + 2)
        vobs_01 = torch.zeros(int(max_pattern[:pattern_length], 2) + 1, dtype=torch.float32, device=device)
        vobs_02 = torch.zeros(int(max_pattern[:pattern_length + 1], 2) + 1, dtype=torch.float32, device=device)
        
        for i in range(length_of_binary_data):
            vobs_01[int(binary_data[i:i + pattern_length], 2)] += 1
            vobs_02[int(binary_data[i:i + pattern_length + 1], 2)] += 1
        
        vobs = [vobs_01, vobs_02]
        sums = torch.zeros(2, dtype=torch.float32, device=device)
        
        for i in range(2):
            for j in range(len(vobs[i])):
                if vobs[i][j] > 0:
                    sums[i] += vobs[i][j] * torch.log(vobs[i][j] / length_of_binary_data)
        sums /= length_of_binary_data
        ape = sums[0] - sums[1]
        xObs = 2.0 * length_of_binary_data * (torch.log(torch.tensor(2.0, device=device)) - ape)
        p_value = torch.special.gammaincc(torch.tensor(pow(2, pattern_length - 1), dtype=torch.float32, device=device), xObs / 2.0)
        
        normalized_vobs_01 = vobs_01 / torch.sum(vobs_01)
        normalized_vobs_02 = vobs_02 / torch.sum(vobs_02)
        entropy_vobs_01 = -torch.sum(vobs_01 * torch.log(vobs_01 / length_of_binary_data + 1e-9)) / length_of_binary_data
        entropy_vobs_02 = -torch.sum(vobs_02 * torch.log(vobs_02 / length_of_binary_data + 1e-9)) / length_of_binary_data
        skewness_vobs_01 = torch.tensor(skew(vobs_01.cpu().numpy()), device=device)
        skewness_vobs_02 = torch.tensor(skew(vobs_02.cpu().numpy()), device=device)
        kurtosis_vobs_01 = torch.tensor(kurtosis(vobs_01.cpu().numpy()), device=device)
        kurtosis_vobs_02 = torch.tensor(kurtosis(vobs_02.cpu().numpy()), device=device)
        mean_vobs_01 = torch.mean(vobs_01)
        mean_vobs_02 = torch.mean(vobs_02)
        variance_vobs_01 = torch.var(vobs_01)
        variance_vobs_02 = torch.var(vobs_02)
        std_dev_vobs_01 = torch.std(vobs_01)
        std_dev_vobs_02 = torch.std(vobs_02)
        num_unique_patterns_01 = torch.sum(vobs_01 > 0).item()
        num_unique_patterns_02 = torch.sum(vobs_02 > 0).item()
        
        return {
            'p_value': p_value.item(),
            'result': p_value.item() >= 0.01,
            'ape': ape.item(),
            'xObs': xObs.item(),
            'vobs_01': vobs_01.cpu().tolist(),
            'vobs_02': vobs_02.cpu().tolist(),
            'normalized_vobs_01': normalized_vobs_01.cpu().tolist(),
            'normalized_vobs_02': normalized_vobs_02.cpu().tolist(),
            'entropy_vobs_01': entropy_vobs_01.item(),
            'entropy_vobs_02': entropy_vobs_02.item(),
            'skewness_vobs_01': skewness_vobs_01.item(),
            'skewness_vobs_02': skewness_vobs_02.item(),
            'kurtosis_vobs_01': kurtosis_vobs_01.item(),
            'kurtosis_vobs_02': kurtosis_vobs_02.item(),
            'mean_vobs_01': mean_vobs_01.item(),
            'mean_vobs_02': mean_vobs_02.item(),
            'variance_vobs_01': variance_vobs_01.item(),
            'variance_vobs_02': variance_vobs_02.item(),
            'std_dev_vobs_01': std_dev_vobs_01.item(),
            'std_dev_vobs_02': std_dev_vobs_02.item(),
            'num_unique_patterns_01': num_unique_patterns_01,
            'num_unique_patterns_02': num_unique_patterns_02,
        }
    
    @staticmethod
    def cumulative_sums_test(binary_data: str, mode=0, verbose=False) -> Dict[str, Any]:
        length_of_binary_data = len(binary_data)
        counts = torch.zeros(length_of_binary_data, dtype=torch.float32)
        if mode != 0:
            binary_data = binary_data[::-1]
        counter = 0
        for char in binary_data:
            sub = 1 if char == '1' else -1
            counts[counter] = counts[counter - 1] + sub if counter > 0 else sub
            counter += 1
        abs_max = torch.max(torch.abs(counts)).item()
        length_of_binary_data_tensor = torch.tensor(length_of_binary_data, dtype=torch.float32)
        start = int(torch.floor(0.25 * torch.floor(torch.tensor(-length_of_binary_data / abs_max + 1, dtype=torch.float32))).item())
        end = int(torch.floor(0.25 * torch.floor(torch.tensor(length_of_binary_data / abs_max - 1, dtype=torch.float32))).item())
        terms_one = [torch.distributions.Normal(0, 1).cdf((4 * k + 1) * abs_max / torch.sqrt(length_of_binary_data_tensor)) - torch.distributions.Normal(0, 1).cdf((4 * k - 1) * abs_max / torch.sqrt(length_of_binary_data_tensor)) for k in range(start, end + 1)]
        start = int(torch.floor(0.25 * torch.floor(torch.tensor(-length_of_binary_data / abs_max - 3, dtype=torch.float32))).item())
        end = int(torch.floor(0.25 * torch.floor(torch.tensor(length_of_binary_data / abs_max, dtype=torch.float32))).item() - 1)
        terms_two = [torch.distributions.Normal(0, 1).cdf((4 * k + 3) * abs_max / torch.sqrt(length_of_binary_data_tensor)) - torch.distributions.Normal(0, 1).cdf((4 * k + 1) * abs_max / torch.sqrt(length_of_binary_data_tensor)) for k in range(start, end + 1)]
        p_value = 1.0 - sum(terms_one) + sum(terms_two)
        mean_counts = torch.mean(counts).item()
        var_counts = torch.var(counts).item()
        range_counts = (torch.max(counts) - torch.min(counts)).item()
        num_zero_crossings = torch.sum(torch.diff((counts >= 0).float())).item()
        max_positive_dev = torch.max(counts).item()
        max_negative_dev = torch.min(counts).item()
        mean_abs_dev = torch.mean(torch.abs(counts - mean_counts)).item()
        entropy_counts = torch.distributions.Categorical(probs=torch.softmax(counts, dim=0)).entropy().item()
        skewness_counts = torch.tensor(skew(counts.numpy())).item()
        kurtosis_counts = torch.tensor(kurtosis(counts.numpy())).item()
        sum_abs_counts = torch.sum(torch.abs(counts)).item()
        trend_slope = (counts[-1] - counts[0]).item() / length_of_binary_data
        num_terms_greater_thresh = sum(1 for x in terms_one + terms_two if x > 0.5)
        mean_terms_one = torch.mean(torch.tensor(terms_one)).item()
        std_terms_one = torch.std(torch.tensor(terms_one)).item()
        min_terms_one = torch.min(torch.tensor(terms_one)).item()
        max_terms_one = torch.max(torch.tensor(terms_one)).item()
        mean_terms_two = torch.mean(torch.tensor(terms_two)).item()
        std_terms_two = torch.std(torch.tensor(terms_two)).item()
        min_terms_two = torch.min(torch.tensor(terms_two)).item()
        max_terms_two = torch.max(torch.tensor(terms_two)).item()
        if verbose:
            print('Cumulative Sums Test DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_binary_data)
            print('\tMode:\t\t\t\t', mode)
            print('\tValue of z:\t\t\t', abs_max)
            print('\tCounts:\t\t\t', counts)
            print('\tMean Counts:\t\t', mean_counts)
            print('\tVariance Counts:\t', var_counts)
            print('\tRange Counts:\t\t', range_counts)
            print('\tNumber of Zero Crossings:\t', num_zero_crossings)
            print('\tMax Positive Deviation:\t', max_positive_dev)
            print('\tMax Negative Deviation:\t', max_negative_dev)
            print('\tMean Absolute Deviation:\t', mean_abs_dev)
            print('\tEntropy Counts:\t', entropy_counts)
            print('\tSkewness Counts:\t', skewness_counts)
            print('\tKurtosis Counts:\t', kurtosis_counts)
            print('\tSum of Absolute Counts:\t', sum_abs_counts)
            print('\tTrend Slope:\t\t', trend_slope)
            print('\tNumber of Terms > 0.5:\t', num_terms_greater_thresh)
            print('\tP-Value:\t\t\t', p_value)
            print('\tMean Terms One:\t', mean_terms_one)
            print('\tStandard Deviation Terms One:\t', std_terms_one)
            print('\tMin Terms One:\t\t', min_terms_one)
            print('\tMax Terms One:\t\t', max_terms_one)
            print('\tMean Terms Two:\t', mean_terms_two)
            print('\tStandard Deviation Terms Two:\t', std_terms_two)
            print('\tMin Terms Two:\t\t', min_terms_two)
            print('\tMax Terms Two:\t\t', max_terms_two)
            print('DEBUG END.')
        return {
            'p_value': p_value,
            'result': p_value >= 0.01,
            'abs_max': abs_max,
            'terms_one': terms_one,
            'terms_two': terms_two,
            'counts': counts.tolist(),
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
            'mean_terms_one': mean_terms_one,
            'std_terms_one': std_terms_one,
            'min_terms_one': min_terms_one,
            'max_terms_one': max_terms_one,
            'mean_terms_two': mean_terms_two,
            'std_terms_two': std_terms_two,
            'min_terms_two': min_terms_two,
            'max_terms_two': max_terms_two
        }
    
    @staticmethod
    def statistical_test(binary_data: str, verbose=False) -> Dict[str, Union[float, bool, str, int]]:
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

        if 5 < pattern_size < 16:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_ints = (1 << pattern_size) - 1
            vobs = torch.zeros(num_ints + 1, dtype=torch.float32, device=device)
            num_blocks = floor(length_of_binary_data / pattern_size)
            init_bits = 10 * pow(2, pattern_size)
            test_bits = num_blocks - init_bits
            c = 0.7 - 0.8 / pattern_size + (4 + 32 / pattern_size) * pow(test_bits, -3 / pattern_size) / 15
            variance = torch.tensor([0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421], dtype=torch.float32, device=device)
            expected = torch.tensor([0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
                                    10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379], dtype=torch.float32, device=device)
            sigma = c * sqrt(variance[pattern_size] / test_bits)
            cumsum = torch.tensor(0.0, dtype=torch.float32, device=device)

            binary_tensor = torch.tensor([int(bit) for bit in binary_data], dtype=torch.float32, device=device)
            block_data = binary_tensor.unfold(0, pattern_size, pattern_size)
            int_reps = block_data.mv(torch.tensor([2**i for i in range(pattern_size-1, -1, -1)], dtype=torch.float32, device=device))

            for i in range(num_blocks):
                int_rep = int(int_reps[i].item())  # Convert to integer
                if i < init_bits:
                    vobs[int_rep] = i + 1
                else:
                    initial = vobs[int_rep]
                    vobs[int_rep] = i + 1
                    cumsum += torch.log2(torch.tensor(i - initial + 1, dtype=torch.float32, device=device))

            phi = float(cumsum / test_bits)
            stat = abs(phi - expected[pattern_size].item()) / (float(sqrt(2)) * sigma)
            p_value = erfc(stat)
            results = {
                "p_value": p_value,
                "is_random": p_value >= 0.01,
                "phi": phi,
                "stat": stat,
                "expected_phi": expected[pattern_size].item(),
                "sigma": sigma,
                "cumsum": cumsum.item(),
                "pattern_size": pattern_size,
                "num_blocks": num_blocks,
                "init_bits": init_bits,
                "test_bits": test_bits
            }
            if verbose:
                print("Maurer's Universal Statistical Test DEBUG:")
                for key, value in results.items():
                    print(f"\t{key}: {value}")
            return results
        else:
            return {
                "p_value": -1.0,
                "is_random": False,
                "message": "Insufficient data for the test",
                "length_of_binary_data": length_of_binary_data,
                "pattern_size": pattern_size
            }
        
    @staticmethod
    def longest_one_block_test(binary_data: str, verbose=False) -> Dict[str, Union[float, bool, torch.Tensor, List[int], int, str, None]]:
        length_of_binary_data = len(binary_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if length_of_binary_data < 128:
            return {
                'p_value': 0.0,
                'is_random': False,
                'frequencies': None,
                'expected_frequencies': None,
                'longest_runs_per_block': None,
                'block_count': 0,
                'block_size': 0,
                'error_message': 'Error: Not enough data to run this test',
                'mean_longest_run': None,
                'std_longest_run': None,
                'max_longest_run': None,
                'min_longest_run': None,
                'block_discrepancy': None,
                'p_value_significance': 'Insufficient data',
                'run_length_histogram': None,
                'unexpected_runs': 0,
                'relative_frequencies': None
            }
        elif length_of_binary_data < 6272:
            k, m, v_values, pi_values = 3, 8, [1, 2, 3, 4], [0.21484375, 0.3671875, 0.23046875, 0.1875]
        elif length_of_binary_data < 750000:
            k, m, v_values, pi_values = 5, 128, [4, 5, 6, 7, 8, 9], \
                                        [0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847]
        else:
            k, m, v_values, pi_values = 6, 10000, [10, 11, 12, 13, 14, 15, 16], \
                                        [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

        number_of_blocks = length_of_binary_data // m
        if number_of_blocks == 0:
            return {
                'p_value': 0.0,
                'is_random': False,
                'frequencies': None,
                'expected_frequencies': None,
                'longest_runs_per_block': None,
                'block_count': 0,
                'block_size': m,
                'error_message': 'Error: Insufficient blocks for the test',
                'mean_longest_run': None,
                'std_longest_run': None,
                'max_longest_run': None,
                'min_longest_run': None,
                'block_discrepancy': None,
                'p_value_significance': 'Insufficient data',
                'run_length_histogram': None,
                'unexpected_runs': 0,
                'relative_frequencies': None
            }

        frequencies = torch.zeros(k + 1, device=device)
        longest_runs_per_block = torch.zeros(number_of_blocks, dtype=torch.int32, device=device)

        # Adjust the length of binary_data to be a multiple of m
        adjusted_length = number_of_blocks * m
        binary_data = binary_data[:adjusted_length]

        block_data = torch.tensor([int(bit) for bit in binary_data], dtype=torch.int32, device=device).view(number_of_blocks, m)
        max_run_counts = torch.zeros(number_of_blocks, dtype=torch.int32, device=device)

        for i in range(number_of_blocks):
            block = block_data[i]
            run_lengths = torch.diff(torch.cat((torch.tensor([0], device=device), block, torch.tensor([0], device=device)))).nonzero(as_tuple=False).view(-1)
            run_lengths = run_lengths[1::2] - run_lengths[::2]
            max_run_counts[i] = run_lengths.max()

        longest_runs_per_block = max_run_counts

        for max_run_count in longest_runs_per_block:
            if max_run_count < v_values[0]:
                frequencies[0] += 1
            for j in range(k):
                if max_run_count == v_values[j]:
                    frequencies[j + 1] += 1
            if max_run_count > v_values[-1]:
                frequencies[-1] += 1

        longest_runs_per_block = longest_runs_per_block.float()
        mean_longest_run = torch.mean(longest_runs_per_block).item()
        std_longest_run = torch.std(longest_runs_per_block).item()
        max_longest_run = torch.max(longest_runs_per_block).item()
        min_longest_run = torch.min(longest_runs_per_block).item()
        block_discrepancy = torch.mean(torch.abs(frequencies - number_of_blocks * torch.tensor(pi_values, device=device))).item()
        run_length_histogram = [torch.sum(longest_runs_per_block == i).item() for i in range(int(max_longest_run) + 1)]
        unexpected_runs = torch.sum(~torch.isin(longest_runs_per_block, torch.tensor(v_values, device=device))).item()
        relative_frequencies = (frequencies / number_of_blocks).tolist()

        xObs = torch.sum((frequencies - number_of_blocks * torch.tensor(pi_values, device=device)) ** 2 / (number_of_blocks * torch.tensor(pi_values, device=device))).item()
        p_value = gammaincc(k / 2, xObs / 2)

        p_value_significance = 'Highly Significant' if p_value <= 0.01 else 'Marginally Significant' if p_value <= 0.05 else 'Not Significant'

        if verbose:
            print("Longest Run of Ones in a Block Test Details:")
            print(f"Block size (m): {m}")
            print(f"Number of blocks: {number_of_blocks}")
            print(f"Longest runs per block: {longest_runs_per_block.tolist()}")
            print(f"Mean longest run: {mean_longest_run}")
            print(f"Std of longest runs: {std_longest_run}")
            print(f"Max longest run: {max_longest_run}")
            print(f"Min longest run: {min_longest_run}")
            print(f"Block discrepancy: {block_discrepancy}")
            print(f"p-value: {p_value}")
            print(f"p-value significance: {p_value_significance}")

        return {
            'p_value': p_value,
            'is_random': p_value > 0.01,
            'frequencies': frequencies.cpu().numpy(),
            'expected_frequencies': pi_values,
            'longest_runs_per_block': longest_runs_per_block.cpu().tolist(),
            'block_count': number_of_blocks,
            'block_size': m,
            'error_message': None,
            'mean_longest_run': mean_longest_run,
            'std_longest_run': std_longest_run,
            'max_longest_run': max_longest_run,
            'min_longest_run': min_longest_run,
            'block_discrepancy': block_discrepancy,
            'p_value_significance': p_value_significance,
            'run_length_histogram': run_length_histogram,
            'unexpected_runs': unexpected_runs,
            'relative_frequencies': relative_frequencies
        }
    
    @staticmethod
    def block_frequency_multiple_sizes(binary_data: str, block_sizes: List[int] = [8, 16, 32, 64, 128]) -> List[Dict[str, Union[str, float, int]]]:
        results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for block_size in block_sizes:
            length_of_bit_string = len(binary_data)
            if length_of_bit_string < block_size:
                results.append({
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
                results.append({
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

            binary_tensor = torch.tensor([1 if bit == '1' else 0 for bit in binary_data], dtype=torch.float32, device=device)
            binary_tensor = binary_tensor[:number_of_blocks * block_size].view(number_of_blocks, block_size)
            one_counts = torch.sum(binary_tensor, dim=1)
            proportions = one_counts / block_size
            proportion_sum = torch.sum((proportions - 0.5) ** 2)
            chi_squared = 4.0 * block_size * proportion_sum
            p_value = gammaincc(number_of_blocks / 2, chi_squared.cpu().item() / 2)
            mean_proportion = torch.mean(proportions).item()
            variance_proportion = torch.var(proportions).item()
            std_dev_proportion = torch.std(proportions).item()
            proportion_of_ones = torch.sum(one_counts).item() / length_of_bit_string
            normalized_number_of_ones = torch.sum(one_counts).item() / length_of_bit_string
            normalized_chi_squared = chi_squared.item() / number_of_blocks
            normalized_proportion_sum = proportion_sum.item() / number_of_blocks

            results.append({
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

        return results


# def hex_to_binary(hex_data):
#     binary_data = bin(int(binascii.unhexlify(hex_data).hex(), 16))[2:]
#     return binary_data.zfill(len(hex_data) * 4)
    
# with open('1.txt', 'r') as file:
#     hex_data = file.read().strip()
#     binary_data = hex_to_binary(hex_data)


# elapsed_time = time.time() - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    
        

# Read binary data from a .bin file
# with open("1.bin", "rb") as file:
#     binary_data = ''.join(format(byte, '08b') for byte in file.read())



# monobit Block test

# # print(binary_data)
# block_sizes = [8, 16, 32, 64, 128]
# # results = FeatureExtract.block_frequency_multiple_sizes(binary_data, block_sizes)

# # # Print Results in a User-Friendly Format
# # print(f"{'Block Size':<12}{'P-Value':<10}{'Random':<10}")
# # print("-" * 32)
# # for result in results:
# #     print(f"{result['Block Size']:<12}{result['P-Value']:<10}{result['Random']:<10}")


# Longest one block test

# result = FeatureExtract.longest_one_block_test(binary_data, verbose=True)
# print("P-Value:", result[0])
# print("Is Random:", result[1])
# if result[2]:
#     print("Error:", result[2])

# Run Test


# block_size = 16  # Example block size

# # Extract features
# raw_features, aggregated_features = FeatureExtract.extract_run_test_features(binary_data, block_size, verbose=False)

# # Print results
# # print("\nRaw Features (p_values per block):", raw_features)
# print("Aggregated Features [Mean p_value, Std p_value, Pass Ratio]:", aggregated_features)

# Binary matrix rank test\
# results = FeatureExtract.binary_matrix_rank_test(binary_data, verbose=True, block_sizes=[(32, 32), (16, 16)])

# Print results
# for block_size, result in results.items():
#     print(f"Block Size: {block_size}")
#     print(f"  P-Value: {result['p_value']}")
#     print(f"  Result: {'Pass' if result['result'] else 'Fail'}")
#     if 'error' in result:
#         print(f"  Error: {result['error']}")
#     else:
#         print(f"  Number of Blocks: {result['num_blocks']}")
#         print(f"  Max Ranks: {result['max_ranks']}")
#         print(f"  Chi-square: {result['xObs']}")
#     print('-' * 50)

#Spectral Test

# p_value, result = FeatureExtract.spectral_test(binary_data, verbose=False)

# print(f"P-Value: {p_value}")
# print(f"Result: {'Pass' if result else 'Fail'}")

# # Spectral block test

# block_size = 128  # Specify block size
# results = FeatureExtract.spectral_test_on_blocks(binary_data, block_size, verbose=False)

# # Aggregate results
# passes = sum(1 for _, _, result in results if result)
# print(f"Blocks Passed: {passes}/{len(results)}")

#Universal Statical test

# r = FeatureExtract.statistical_test(binary_data,verbose=True)
# print(r)

#Linear Complexity Test
# FeatureExtract.linear_complexity_test(binary_data=binary_data,verbose=True)








# a = FeatureExtract()
# print(a.block_frequency_multiple_sizes(binary_data))










# Function to inspect a class and get its methods and return types
# def get_class_methods_and_return_types(cls):
#     class_name = cls.__name__
#     methods_info = []
    
#     for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
#         # Get return type using type hints
#         type_hints = get_type_hints(method)
#         return_type = type_hints.get('return', 'Unknown')  # Default to 'Unknown' if no return type specified
#         methods_info.append((name, return_type))
    
#     return class_name, methods_info

# # Using the function
# class_name, methods_info = get_class_methods_and_return_types(FeatureExtract)

# print(f"Class Name: {class_name}")
# print("Methods and Return Types:")
# for method_name, return_type in methods_info:
#     print(f"  Method: {method_name}, Return Type: {return_type}")


# print(FeatureExtract.approximate_entropy_test(binary_data=binary_data))
# print(FeatureExtract.binary_matrix_rank_test(binary_data=binary_data))
# print(FeatureExtract.cumulative_sums_test(binary_data=binary_data))


# elapsed_time = time.time() - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")


# print(FeatureExtract.serial_test_and_extract_features(binary_data=binary_data))
# print(FeatureExtract.spectral_test(binary_data=binary_data))
# print(FeatureExtract.linear_complexity_test(binary_data=binary_data))
# print(FeatureExtract.longest_one_block_test(binary_data=binary_data))
# print(FeatureExtract.block_frequency_multiple_sizes(binary_data=binary_data))
# print(FeatureExtract.statistical_test(binary_data=binary_data))

# elapsed_time = time.time() - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")


# print(FeatureExtract.extract_run_test_features(binary_data=binary_data))
# print(FeatureExtract.monobit_test(binary_data=binary_data))
# print(FeatureExtract.spectral_test_on_blocks(binary_data=binary_data, block_size=128))


# print('test')