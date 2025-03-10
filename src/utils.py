import math
import json
import gzip
from ast import literal_eval
from tqdm import tqdm


def get_num_bits_per_program_uniform(program, num_functions, number_range):
    bit_size = math.log(number_range, 2)
    max_range = 8
    max_length = 20

    num_bits = 0
    for i, l in enumerate(program.split("\n")):

        num_bits += math.log(num_functions + 1, 2)

        seq_bit_size = math.log(i + 1, 2)
        range_size = math.log(max_range, 2)
        repeat_size = math.log(max_length, 2)
        input_vars = l.split("(")[-1].split(")")[0]
        func = l.split("= ")[-1].split("(")[0]

        if func in {"set_list"}:
            list_size = len(literal_eval(input_vars))
            num_bits += list_size * bit_size

        elif func in {"range_func_up"}:
            num_bits += bit_size * 2

        elif func in {"get_subsequence_between_indices"}:
            num_bits += seq_bit_size + 2 * bit_size  # this can be optimized maybe?

        elif func in {"add_two_lists", "subtract_two_lists", "modulo_two_lists"}:
            num_bits += 2 * seq_bit_size

        elif func in {"substitute"}:
            num_bits += seq_bit_size + bit_size + bit_size

        elif func in {"range_func_interval"}:
            num_bits += bit_size + bit_size + range_size

        elif func in {"modulo_item_list", "add_item_list", "subtract_item_list"}:
            num_bits += bit_size + seq_bit_size

        elif func in {"repeat_num"}:
            num_bits += bit_size + repeat_size

        elif func in {"get_subsequence_with_jumps"}:
            num_bits += seq_bit_size + range_size

        elif func in {"repeat_list"}:
            num_bits += seq_bit_size + range_size

        elif func in {"filter"}:
            num_bits += seq_bit_size

        elif func in {"interleave", "concatenate"}:
            num_bits += seq_bit_size + seq_bit_size

        elif func in {"reverse_list"}:
            num_bits += seq_bit_size

        elif func in {"scan_add"}:
            num_bits += seq_bit_size

        elif l.startswith("output"):
            num_bits += 0

        elif func in {"min_n", "max_n"}:
            num_bits += bit_size + seq_bit_size

        else:
            raise ValueError

    return num_bits


def read_jsonl(file_path):
    data_points = []
    with open(file_path, "r") as file:
        for line in tqdm(file):
            data = json.loads(line)
            data_points.append(data)
    return data_points


def save_results_as_jsonl(results, filename):
    with open(filename, "w") as f:
        for result in tqdm(results):
            try:
                json_record = json.dumps(result)
            except ValueError:
                result["program_output"] = ""
                json_record = json.dumps(result)
            f.write(json_record + "\n")


def get_text_encoding_with_gzip_prior(text: str) -> int:
    compressed_data = gzip.compress(text.encode())
    return compressed_data
