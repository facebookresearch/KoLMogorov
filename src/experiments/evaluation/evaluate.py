from tqdm import tqdm
import gzip
import numpy as np
import logging
import pandas as pd

from src.utils import read_jsonl, save_results_as_jsonl


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


SEQ_SPLIT = "#_#_#_#_#_#_#"


def get_text_encoding_with_gzip_prior(text: str) -> int:
    compressed_data = gzip.compress(text.encode())
    return compressed_data


def get_text_encoding_num_bits_with_gzip_prior(text: str) -> int:
    compressed_data = gzip.compress(text.encode())
    num_bits = len(compressed_data) * 8
    return num_bits


def decode_bytes_to_string(byte_list):
    bytes_obj = bytes(byte_list)
    decoded_string = bytes_obj.decode("latin1")

    return decoded_string


def calculate_compression_cost(data, output_file):
    # populate the necessary field (keep program)
    for data_point in tqdm(data):

        # calculate the size of the sequence without compression
        data_point["num_bits_no_compression"] = len(data_point["sequence_input"]) * 8
        data_point["executable_program"] = len(data_point["program_output"]) > 0

        # if we keep the program, calculate the correction cost
        if data_point["executable_program"]:
            data_point["program_output_parsed_to_bits"] = [
                x % 256 for x in data_point["program_output"]
            ]

            data_point["sequence_encoding_cost_bits"] = min(
                len(data_point["sequence_input"]) * 8,
                get_text_encoding_num_bits_with_gzip_prior(
                    decode_bytes_to_string(data_point["sequence_input"])
                ),
            )

            data_point["ex_acc"] = (
                1 if data_point["program_output"] == data_point["sequence_input"] else 0
            )
        # 1 bit if to keep the program or the sequence
        data_point["use_program"] = data_point["executable_program"]

    # global encoding for programs
    all_programs_to_compress = SEQ_SPLIT.join(
        [x["program"] for x in data if x["use_program"]]
    )
    progam_encoding = get_text_encoding_with_gzip_prior(all_programs_to_compress)
    progam_encoding_bits = len(progam_encoding) * 8

    # global encoding for sequences
    all_sequences_to_compress = SEQ_SPLIT.join(
        [
            decode_bytes_to_string(x["sequence_input"])
            for x in data
            if not x["use_program"]
        ]
    )
    sequence_encoding = get_text_encoding_with_gzip_prior(all_sequences_to_compress)
    sequence_encoding_bits = len(sequence_encoding) * 8

    # global encoding for separators
    separator_bits = [1 if x["use_program"] else 0 for x in data]
    separator_encoding = get_text_encoding_with_gzip_prior(str(separator_bits))
    separator_encoding_bits = len(separator_encoding) * 8

    # print stats
    percent_ex_acc = len(
        [x["ex_acc"] for x in data if x["executable_program"] if x["ex_acc"]]
    ) / len(data)
    all_programs_ex = "\n".join(
        [x["program"] for x in data if x["executable_program"] if x["ex_acc"]]
    )
    all_sequeunces_ex = "\n".join(
        [
            decode_bytes_to_string(x["sequence_input"])
            for x in data
            if x["executable_program"]
            if x["ex_acc"]
        ]
    )
    all_programs_ex_bits = get_text_encoding_num_bits_with_gzip_prior(all_programs_ex)
    all_sequeunces_ex_bits = get_text_encoding_num_bits_with_gzip_prior(
        all_sequeunces_ex
    )
    comp_rate_ex_acc = all_programs_ex_bits / all_sequeunces_ex_bits
    logger.info(f"all_programs_ex_bits: {all_programs_ex_bits}")
    logger.info(f"all_sequeunces_ex_bits: {all_sequeunces_ex_bits}")
    logger.info(f"comp_rate_ex_acc (precision): {comp_rate_ex_acc}")
    logger.info(f"percent_ex_acc: {percent_ex_acc}")

    # overall bits
    data_size = len(data)
    num_separator_bits = (
        separator_encoding_bits  # bits to separate between programs and sequences
    )
    num_sequence_examples = len([x for x in data if not x["use_program"]])

    num_separator_sequence_bits = num_sequence_examples

    # % use programs
    percentage_programs = np.average([x["use_program"] for x in data])

    total_sequence_bits = num_separator_sequence_bits + sequence_encoding_bits
    logger.info(f"percentage_programs: {percentage_programs:.3f}")

    logger.info(
        f"total_sequence_bits: {total_sequence_bits} ({total_sequence_bits/8:.3f} bytes)"
    )

    logger.info(
        f"num_separator_bits: {num_separator_bits} ({num_separator_bits/8:.3f} bytes)"
    )
    logger.info(
        f"progam_encoding_bits: {progam_encoding_bits} ({progam_encoding_bits/8:.3f} bytes)"
    )
    logger.info(
        f"sequence_encoding_bits: {sequence_encoding_bits} ({sequence_encoding_bits/8:.3f} bytes)"
    )

    # average bits per example
    avg_bits_per_sequence_example = total_sequence_bits / num_sequence_examples
    logger.info(f"avg_bits_per_sequence_example: {avg_bits_per_sequence_example:.3f}")
    logger.info(f"Writing eval file to: {output_file}")
    save_results_as_jsonl(data, output_file)

    percent_executable_programs = (
        np.average([x["executable_program"] for x in data]) * 100
    )
    percent_ex_acc = (
        np.average([x["ex_acc"] for x in data if x["executable_program"]]) * 100
    )
    logger.info(f"percent_executable_programs: {percent_executable_programs}")
    logger.info(f"accuracy: {percent_ex_acc}")
    pd.DataFrame(data).to_csv(output_file.replace(".jsonl", ".csv"))

    all_sequences_to_compress_no_filter = SEQ_SPLIT.join(
        [
            decode_bytes_to_string(x["sequence_input"])
            for x in data
            if not x["use_program"]
        ]
    )
    sequence_encoding_bits_no_filter = (
        len(get_text_encoding_with_gzip_prior(all_sequences_to_compress_no_filter)) * 8
    )
    avg_zipped_sequence_size = sequence_encoding_bits_no_filter / data_size
    logger.info(f"avg_zipped_sequence_size: {avg_zipped_sequence_size}")


def evaluate(data_input_path: str, data_output_path: str):

    data = read_jsonl(data_input_path)
    calculate_compression_cost(data, data_output_path)
