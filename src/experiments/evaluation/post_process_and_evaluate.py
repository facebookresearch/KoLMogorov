from src.experiments.evaluation.evaluate import evaluate
from src.experiments.evaluation.post_process_programs import post_process_programs
import argparse


def main(input_data_path: str):
    post_processed_progams_path = input_data_path.replace(
        ".jsonl", "_postprocessed.jsonl"
    )
    evaluated_programs_path = input_data_path.replace(".jsonl", "_evaluated.jsonl")

    post_process_programs(
        data_input_path=input_data_path,
        output_file_path=post_processed_progams_path,
    )

    evaluate(
        data_input_path=post_processed_progams_path,
        data_output_path=evaluated_programs_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--input_file_path",
        type=str,
        default="src/experiments/data/o1_mini_generated_programs_text.jsonl",
    )
    args = parser.parse_args()
    main(args.input_file_path)
