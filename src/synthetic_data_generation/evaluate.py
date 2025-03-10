import multiprocessing
from tqdm import tqdm
import heapq
import argparse
import math
from ast import literal_eval
import gzip
from src.utils import (
    get_num_bits_per_program_uniform,
    get_text_encoding_with_gzip_prior,
    read_jsonl,
)
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_text_encoding_num_bits_with_gzip_prior(text: str) -> int:
    compressed_data = gzip.compress(text.encode())
    num_bits = len(compressed_data) * 8
    return num_bits


def decode_bytes_to_string(byte_list):
    bytes_obj = bytes(byte_list)
    decoded_string = bytes_obj.decode("latin1")

    return decoded_string


# Initiators
def range_func_up(start_range, end_range):
    return [i for i in range(start_range, end_range + 1)]


def range_func_interval(start_range, end_range, interval):
    return [i for i in range(start_range, end_range, interval)]


def repeat_num(num, val):
    return [val] * num


def set_list(lst):
    return lst


# list operations
repeat_list = lambda num, lst: lst * num
reverse_list = lambda lst: lst[::-1]
substitute = lambda x, pattern, replacement: [
    replacement if i == pattern else i for i in x
]
get_subsequence_with_jumps = lambda seq, jump: seq[::jump]
get_subsequence_between_indices = lambda seq, start, end: seq[start:end]

max_n = lambda lst, n: heapq.nlargest(n, lst)
min_n = lambda lst, n: heapq.nsmallest(n, lst)
add_item_list = lambda lst, constant: [x + constant for x in lst]
subtract_item_list = lambda lst, constant: [x - constant for x in lst]
modulo_item_list = lambda lst, constant: [x % constant for x in lst]

# Filtering functions
filter = lambda f, x: [i for i in x if f(i)]
is_even = lambda x: x % 2 == 0
is_odd = lambda x: x % 2 != 0
is_not_zero = lambda x: x != 0

# Operations on two lists
add_two_lists = lambda lst1, lst2: [x + y for x, y in zip(lst1, lst2)]
subtract_two_lists = lambda lst1, lst2: [x - y for x, y in zip(lst1, lst2)]
modulo_two_lists = lambda lst1, lst2: [
    x % y for x, y in zip(lst1, lst2) if y != 0
]  # Avoid division by zero


def scan_add(elems, initializer=0):
    result = [initializer]
    for elem in elems:
        result.append(result[-1] + elem)
    return result[1:]


# Concatenate and interleave functions
def concatenate(*lst):
    return [x for sublist in lst for x in sublist]


def interleave(x, y):
    # Interleave elements from both lists
    interleaved = [val for pair in zip(x, y) for val in pair]
    # Determine the longer list and append the remaining elements
    longer = x if len(x) > len(y) else y
    tail_start = min(len(x), len(y))
    tail = longer[tail_start:]
    # Return the interleaved list followed by the remaining elements
    return interleaved + tail


methods_dict = {
    "range_func_up": range_func_up,
    "range_func_interval": range_func_interval,
    "repeat_num": repeat_num,
    "set_list": set_list,
    "repeat_list": repeat_list,
    "reverse_list": reverse_list,
    "substitute": substitute,
    "get_subsequence_with_jumps": get_subsequence_with_jumps,
    "get_subsequence_between_indices": get_subsequence_between_indices,
    "max_n": max_n,
    "min_n": min_n,
    "add_item_list": add_item_list,
    "subtract_item_list": subtract_item_list,
    "modulo_item_list": modulo_item_list,
    "filter": filter,
    "is_even": is_even,
    "is_odd": is_odd,
    "is_not_zero": is_not_zero,
    "add_two_lists": add_two_lists,
    "subtract_two_lists": subtract_two_lists,
    "modulo_two_lists": modulo_two_lists,
    "scan_add": scan_add,
    "concatenate": concatenate,
    "interleave": interleave,
}


def worker(code):
    try:
        local_scope = methods_dict
        exec(code, globals(), local_scope)
        return local_scope.get("output")
    except Exception as e:
        return str(e)


def execute_with_timeout(codes, timeout=2):
    results = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        result_objects = [pool.apply_async(worker, (code,)) for code in codes]
        for result in tqdm(result_objects, desc="Processing codes", unit="code"):
            try:
                output = result.get(timeout)
                results.append(output)
            except multiprocessing.TimeoutError:
                results.append("Execution timed out")
            except Exception as e:
                results.append(f"caught another exception: {e}")
    return results


def parse_data_points(data_points):
    codes = [
        dp["generation"]
        .replace("```", "")
        .split("###")[0]
        .split("python")[-1]
        .split("<|end_of_text|>")[0]
        for dp in data_points
    ]
    outputs = []
    chunked_codes = [codes[i : i + 128] for i in range(0, len(codes), 128)]
    for chunk in tqdm(chunked_codes):
        outputs.extend(execute_with_timeout(chunk, timeout=10))
    return outputs


def main(input_file_path: str):
    all_res = []
    all_ex = []
    all_data = []

    data = read_jsonl(input_file_path)
    all_data.append(data)
    for i, x in enumerate(data):
        x["sequence"] = literal_eval(data[i]["prompt"].split(": ")[-1].split(" [")[0])
    results = parse_data_points(data)

    for i, x in enumerate(results):
        data[i]["ex_res"] = results[i] if results[i] is not None else []
        data[i]["length_correct_prefix"] = max(
            [0]
            + [
                j
                for j in range(130)
                if data[i]["ex_res"][:j] == data[i]["sequence"][:j]
            ]
        )
        data[i]["ex_acc"] = results[i] == data[i]["sequence"]
    all_ex.append(results)
    res = len(
        [
            [x, data[i]["sequence"]]
            for i, x in enumerate(results)
            if x == data[i]["sequence"]
        ]
    ) / len(results)
    print(res)
    all_res.append(res)
    r_prefix = [
        len(
            [
                i
                for i, x in enumerate(results)
                if data[i]["ex_res"][:j] == data[i]["sequence"][:j]
            ]
        )
        for j in range(129)
    ]

    acc_at_prefix_16 = r_prefix[16] / len(data)
    acc_at_prefix_32 = r_prefix[32] / len(data)
    acc_at_prefix_64 = r_prefix[64] / len(data)
    acc_at_prefix_128 = r_prefix[128] / len(data)

    logger.info(f"acc_at_prefix_16: {acc_at_prefix_16}")
    logger.info(f"acc_at_prefix_32: {acc_at_prefix_32}")
    logger.info(f"acc_at_prefix_64: {acc_at_prefix_64}")
    logger.info(f"acc_at_prefix_128: {acc_at_prefix_128}")

    separator_bits = [
        1 if results[i] == data[i]["sequence"] else 0 for i in range(len(results))
    ]
    bits = [
        get_num_bits_per_program_uniform(
            data[i]["generation"].split("<|end_of_text|>")[0], 24, 256
        )
        for i, x in enumerate(results)
        if separator_bits[i]
    ]
    all_sequences_to_compress = "\n".join(
        [
            decode_bytes_to_string(data[i]["sequence"])
            for i, x in enumerate(results)
            if not separator_bits[i]
        ]
    )
    sequence_encoding = get_text_encoding_with_gzip_prior(all_sequences_to_compress)
    sequence_encoding_bits = len(sequence_encoding) * 8
    all_bits = math.ceil(sum(bits) + sequence_encoding_bits + len(separator_bits))
    compression_rate = all_bits / (1000105 * 8)
    logger.info(f"compression_rate: {compression_rate}")
    logger.info(f"res: {all_res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--input_file_path",
        type=str,
        default="synthetic_data_generation/data/sc_audio_test_data.jsonl",
    )
    args = parser.parse_args()
    main(args.input_file_path)
