import random
import heapq
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from src.utils import get_num_bits_per_program_uniform


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


# Function names for printing
function_names = {
    range_func_up: "range_func_up",
    range_func_interval: "range_func_interval",
    repeat_num: "repeat_num",
    set_list: "set_list",
}


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


# Sample between 1 to 5 base sequences
def get_program(
    num_sequences,
    min_length,
    max_length,
    min_value,
    max_value,
    concat_prob,
    reuse_prob,
    list_update_prob,
    reverse_prob,
    substitute_prob,
    exclude_substition_prob,
    subsequence_prob,
    repeat_list_prob,
    math_update_prob,
    min_max_prob,
    add_subtract_prob,
    modulo_prob,
    filter_update_prob,
    filter_sequences_prob,
    two_list_update_prob,
    two_list_sub_add_prob,
    two_list_modulo_prob,
    scan_add_prob,
):
    num_sequences = random.randint(1, num_sequences)
    sequences = []
    sequence_names = []
    sequence_full_names = []

    current_sequence_index = 1
    print(f"Number of sequences: {num_sequences}")
    res = []
    for i in range(1, num_sequences + 1):
        length = random.randint(min_length, max_length)
        start_range = random.randint(min_value, max_value - length)
        choice = random.choice(list(function_names.keys()))
        sequence_name = f"sequence_{i}"
        sequence_full_name = f"sequence_{i}"
        if choice == range_func_interval:
            interval = random.randint(2, 5)
            end_range = min(start_range + (length - 1) * interval, max_value)
            sequence = range_func_interval(start_range, end_range + 1, interval)
            res.append(
                f"sequence_{i} = {function_names[choice]}({start_range}, {end_range + 1}, {interval}) # {sequence}"
            )
        elif choice == repeat_num:
            value = random.randint(min_value, max_value)
            sequence = repeat_num(length, value)
            sequence_full_name = f"sequence_{current_sequence_index}_repeat"  # Tagging set_list sequences
            res.append(
                f"sequence_{i} = {function_names[choice]}({length}, {value}) # {sequence}"
            )
        elif choice == set_list:
            random_list = [random.randint(min_value, max_value) for _ in range(length)]
            sequence = set_list(random_list)
            sequence_full_name = f"sequence_{current_sequence_index}_set_list"  # Tagging set_list sequences
            res.append(
                f"sequence_{i} = {function_names[choice]}({random_list}) # {sequence}"
            )
        else:  # range_func_up
            end_range = start_range + length - 1
            sequence = range_func_up(start_range, end_range)
            res.append(
                f"sequence_{i} = {function_names[choice]}({start_range}, {end_range}) # {sequence}"
            )
        sequences.append(sequence)
        sequence_names.append(sequence_name)
        sequence_full_names.append(sequence_full_name)
        current_sequence_index += 1

    reversed_indices = set()
    excluded_indices = set()  # Set to keep track of excluded sequence indices

    if random.random() < list_update_prob:
        for index, seq in enumerate(sequences):
            if (
                random.random() < reverse_prob
                and index not in reversed_indices
                and not sequence_full_names[index].endswith("_repeat")
            ):
                reversed_seq = reverse_list(seq)
                new_sequence_name = f"sequence_{current_sequence_index}"
                sequences.append(reversed_seq)
                sequence_names.append(new_sequence_name)
                sequence_full_names.append(new_sequence_name)
                res.append(
                    f"{new_sequence_name} = reverse_list({sequence_names[index]}) # {reversed_seq}"
                )
                reversed_indices.add(
                    index
                )  # Add the original sequence index to the set
                reversed_indices.add(
                    len(sequences) - 1
                )  # Also add the index of the new reversed sequence
                current_sequence_index += 1

        # Apply new subsequence methods
        for index, seq in enumerate(sequences):
            if random.random() < repeat_list_prob:  # 5% chance to repeat the list
                repeat_times = random.randint(
                    2, 4
                )  # Repeat the list between 2 and 4 times
                repeated_seq = repeat_list(repeat_times, seq)
                new_sequence_name = f"sequence_{current_sequence_index}"
                sequences.append(repeated_seq)
                sequence_names.append(new_sequence_name)
                sequence_full_names.append(new_sequence_name)
                res.append(
                    f"{new_sequence_name} = repeat_list({repeat_times}, {sequence_names[index]}) # {repeated_seq}"
                )
                current_sequence_index += 1

            if index not in reversed_indices and index not in excluded_indices:
                if (
                    random.random() < subsequence_prob
                ):  # 10% chance for get_subsequence_with_jumps
                    jump = random.randint(2, 5)  # Jump between 2 and 5
                    sub_seq = get_subsequence_with_jumps(seq, jump)
                    new_sequence_name = f"sequence_{current_sequence_index}"
                    sequences.append(sub_seq)
                    sequence_names.append(new_sequence_name)
                    sequence_full_names.append(new_sequence_name)
                    res.append(
                        f"{new_sequence_name} = get_subsequence_with_jumps({sequence_names[index]}, {jump}) # {sub_seq}"
                    )
                    current_sequence_index += 1

                if (
                    random.random() < subsequence_prob
                ):  # 10% chance for get_subsequence_between_indices
                    if len(seq) > 1:
                        start = random.randint(0, len(seq) - 2)
                        end = random.randint(start + 2, len(seq))
                        sub_seq = get_subsequence_between_indices(seq, start, end)
                        new_sequence_name = f"sequence_{current_sequence_index}"
                        sequences.append(sub_seq)
                        sequence_names.append(new_sequence_name)
                        sequence_full_names.append(new_sequence_name)
                        res.append(
                            f"{new_sequence_name} = get_subsequence_between_indices({sequence_names[index]}, {start}, {end}) # {sub_seq}"
                        )
                        current_sequence_index += 1

        # Apply substitution
        for index, seq in enumerate(sequences):
            if (
                random.random() < substitute_prob
            ):  # Adjust the probability for substitution as needed
                possible_patterns = list(set(seq))  # Unique elements in the sequence
                if possible_patterns:
                    pattern = random.choice(possible_patterns)
                    replacement = random.randint(min_value, max_value)
                    # Ensure replacement is within range and not equal to the pattern
                    if pattern != replacement and min_value <= replacement <= max_value:
                        substituted_seq = substitute(seq, pattern, replacement)
                        new_sequence_name = f"sequence_{current_sequence_index}"
                        sequences.append(substituted_seq)
                        sequence_names.append(new_sequence_name)
                        sequence_full_names.append(new_sequence_name)
                        res.append(
                            f"{new_sequence_name} = substitute({sequence_names[index]}, {pattern}, {replacement}) # {substituted_seq}"
                        )
                        current_sequence_index += 1

                        # Decide whether to exclude the original sequence from final output
                        if (
                            random.random() < exclude_substition_prob
                        ):  # 25% chance to exclude the original sequence
                            excluded_indices.add(index)

    if random.random() < math_update_prob:
        # After generating sequences
        original_length = len(
            sequences
        )  # Store the original length of the sequences list

        for index in range(original_length):  # Loop only over the original sequences
            seq = sequences[index]
            operation_rand = random.random()

            if (
                sequence_full_names[index].endswith("_set_list")
                and random.random() < min_max_prob
            ):
                # Decide randomly whether to use max_n or min_n
                if random.choice([True, False]):
                    n = random.randint(
                        3, len(seq) - 2
                    )  # Choose n between 3 and the length of the sequence - 2
                    new_seq = max_n(seq, n)
                    operation_name = "max_n"
                else:
                    n = random.randint(3, len(seq) - 2)
                    new_seq = min_n(seq, n)
                    operation_name = "min_n"

                new_sequence_name = f"sequence_{current_sequence_index}"
                sequences.append(new_seq)
                sequence_names.append(new_sequence_name)
                sequence_full_names.append(new_sequence_name)
                res.append(
                    f"{new_sequence_name} = {operation_name}({sequence_names[index]}, {n}) # {new_seq}"
                )
                current_sequence_index += 1

            elif operation_rand < add_subtract_prob:
                # Decide randomly whether to add or subtract
                operation_name = None
                if random.choice([True, False]):
                    max_add = min(max_value - x for x in seq)
                    if max_add > 1:
                        constant = random.randint(1, max_add)
                        new_seq = add_item_list(seq, constant)
                        operation_name = "add_item_list"
                else:
                    min_subtract = min(x - min_value for x in seq)
                    if min_subtract > 1:
                        constant = random.randint(1, min_subtract)
                        new_seq = subtract_item_list(seq, constant)
                        operation_name = "subtract_item_list"
                if operation_name:
                    new_sequence_name = f"sequence_{current_sequence_index}"
                    sequences.append(new_seq)
                    sequence_names.append(new_sequence_name)
                    sequence_full_names.append(new_sequence_name)
                    res.append(
                        f"{new_sequence_name} = {operation_name}({sequence_names[index]}, {constant}) # {new_seq}"
                    )
                    current_sequence_index += 1

            elif operation_rand < add_subtract_prob + modulo_prob:
                seq_max = max_n(seq, 3)[-1]
                if seq_max > 2:
                    constant = random.randint(2, seq_max)
                    new_seq = modulo_item_list(seq, constant)
                    operation_name = f"modulo_item_list"

                    new_sequence_name = f"sequence_{current_sequence_index}"
                    sequences.append(new_seq)
                    sequence_names.append(new_sequence_name)
                    sequence_full_names.append(new_sequence_name)
                    res.append(
                        f"{new_sequence_name} = {operation_name}({sequence_names[index]}, {constant}) # {new_seq}"
                    )
                    current_sequence_index += 1

    # Inside the get_program function, after other sequence manipulations
    for index, seq in enumerate(sequences):
        if random.random() < scan_add_prob:
            # Check that the sequence does not contain zero
            if len({x for x in seq}) > 2:
                scanned_seq = scan_add(seq)

                # Check that the maximum value in the scanned sequence does not exceed max_value
                if max(scanned_seq) <= max_value:
                    new_sequence_name = f"sequence_{current_sequence_index}"
                    sequences.append(scanned_seq)
                    sequence_names.append(new_sequence_name)
                    sequence_full_names.append(new_sequence_name)
                    res.append(
                        f"{new_sequence_name} = scan_add({sequence_names[index]}) # {scanned_seq}"
                    )
                    current_sequence_index += 1
                else:
                    continue
            else:
                continue

    # Inside the get_program function, after other sequence manipulations
    if random.random() < filter_update_prob:
        # Choose a filter function randomly
        filter_func = random.choice([is_even, is_odd, is_not_zero])
        filtered_seq = filter(filter_func, seq)

        # Check if the filtered sequence is different from the original
        if (
            filtered_seq != seq
            and len(filtered_seq) > 2
            and random.random() < filter_sequences_prob
        ):
            new_sequence_name = f"sequence_{current_sequence_index}"
            sequences.append(filtered_seq)
            sequence_names.append(new_sequence_name)
            sequence_full_names.append(new_sequence_name)
            filter_func_name = (
                "is_even"
                if filter_func == is_even
                else "is_odd" if filter_func == is_odd else "is_not_zero"
            )
            res.append(
                f"{new_sequence_name} = filter({filter_func_name}, {sequence_names[index]}) # {filtered_seq}"
            )
            current_sequence_index += 1
        else:
            # Optional: Handle the case where the filtered sequence is the same as the original
            # For example, you might want to log this or try a different filter
            print(
                f"Filtered sequence is identical to the original for {sequence_names[index]}. Skipping."
            )

    if random.random() < two_list_update_prob:
        original_length = len(sequences)
        for i in range(original_length):
            for j in range(i + 1, original_length):
                operation_rand = random.random()
                if operation_rand < two_list_sub_add_prob:
                    operation = random.choice([add_two_lists, subtract_two_lists])
                elif operation_rand < two_list_modulo_prob + two_list_sub_add_prob:
                    operation = modulo_two_lists
                else:
                    operation = None

                if len(sequences[i]) == len(sequences[j]) and operation:

                    result_seq = operation(sequences[i], sequences[j])

                    # Check if the result is within the desired range
                    if (
                        all(min_value <= x <= max_value for x in result_seq)
                        and len({x for x in result_seq}) > 2
                    ):
                        new_sequence_name = f"sequence_{current_sequence_index}"
                        sequences.append(
                            result_seq
                        )  # Append to the main list but only operate on original sequences
                        sequence_names.append(new_sequence_name)
                        sequence_full_names.append(new_sequence_name)
                        operation_name = (
                            "add_two_lists"
                            if operation == add_two_lists
                            else (
                                "subtract_two_lists"
                                if operation == subtract_two_lists
                                else "modulo_two_lists"
                            )
                        )
                        res.append(
                            f"{new_sequence_name} = {operation_name}({sequence_names[i]}, {sequence_names[j]}) # {result_seq}"
                        )
                        current_sequence_index += 1
                    else:
                        # Handle cases where the result is out of range
                        continue

    # Concat / interleave all sequences progressively, allowing reuse
    usage_count = {
        i: 0 for i in range(len(sequences))
    }  # Start with 0 because none is used initially
    output = None
    seq_name = None
    # Use each sequence at least once
    for index in range(len(sequences)):
        if index in excluded_indices:
            print(f"program will not include index: {index}")
            continue  # Skip this sequence as it's excluded
        seq_to_use = sequences[index]
        seq_to_use_name = sequence_names[index]
        usage_count[index] += 1  # Increment usage count
        if output is None:
            # Initialize output with the first sequence
            output = seq_to_use
            seq_name = seq_to_use_name
        else:
            # Decide operation
            operation = "concatenate" if random.random() < concat_prob else "interleave"
            new_seq_name = f"sequence_{len(sequence_names) + 1}"
            sequence_names.append(new_seq_name)
            if operation == "concatenate":
                output = concatenate(output, seq_to_use)
            else:
                output = interleave(output, seq_to_use)
            res.append(
                f"{new_seq_name} = {operation}({seq_name}, {seq_to_use_name}) # {output}"
            )
            seq_name = new_seq_name

    # Allow reusing
    for _ in range(len(usage_count)):
        rand_val = random.random()
        if rand_val < reuse_prob:
            reuse_index = random.choice(
                [i for i, count in usage_count.items() if count < 2]
            )
        else:
            continue
        seq_to_use = sequences[reuse_index]
        seq_to_use_name = sequence_names[reuse_index]
        usage_count[reuse_index] += 1  # Increment usage count
        operation = "concatenate" if random.random() < concat_prob else "interleave"
        new_seq_name = f"sequence_{len(sequence_names) + 1}"
        sequence_names.append(new_seq_name)
        if operation == "concatenate":
            output = concatenate(output, seq_to_use)
        else:
            output = interleave(output, seq_to_use)
        res.append(
            f"{new_seq_name} = {operation}({seq_name}, {seq_to_use_name}) # {output}"
        )
        seq_name = new_seq_name

    res.append(f"output = {seq_name} # {output}")
    res = "\n".join(res)
    return res


def execute_and_validate_program(program_output):
    lines = program_output.split("\n")
    results = {}
    for line in lines:
        if "=" in line:
            # Extract the left-hand side (variable name) and right-hand side (expression)
            var_name, expression = line.split("=", 1)
            var_name = var_name.strip()
            expression = expression.split("#")[0].strip()  # Remove the comment part
            # Execute the expression and store the result in the dictionary
            exec(f"{var_name} = {expression}", globals(), results)
            # Extract the expected result from the comment
            expected_result = eval(line.split("#")[1].strip(), globals(), results)
            # Compare the executed result with the expected result
            if results[var_name] != expected_result:
                return False
            else:
                continue
    return True


def main(num_examples: int, test_examples_path: str, output_file_path: str):
    num_sequences = 5
    min_length = 5
    max_length = 25
    min_value = 0
    max_value = 255
    concat_prob = 0.8

    list_update_prob = 0.4
    repeat_list_prob = 0.1
    reverse_prob = 0.1
    substitute_prob = 0.1
    exclude_substition_prob = 0.25
    subsequence_prob = 0.05
    reuse_prob = 0.2

    math_update_prob = 0.4
    min_max_prob = 0.1
    add_subtract_prob = 0.1
    modulo_prob = 0.1

    filter_update_prob = 0.4
    filter_sequences_prob = 0.3  # 3 options

    two_list_update_prob = 0.4
    two_list_sub_add_prob = 0.1
    two_list_modulo_prob = 0.1
    scan_add_prob = 0.1

    programs = []
    num_bytes = 0
    last_1k_bytes = 0
    num_bytes_to_stop = np.inf
    num_examples_to_stop = num_examples
    i = 0

    if test_examples_path:
        test_data_df = pd.read_csv(test_examples_path)
        test_data = test_data_df.to_dict("records")
        test_sequences = {
            " ".join([str(z) for z in literal_eval(x["sequence"])]) for x in test_data
        }
    else:
        test_sequences = {}

    while True:
        print(f"\n### Generating program: {i} ###\n\n")
        program_output = ""
        while len(program_output) < 1:
            program_output = get_program(
                num_sequences,
                min_length,
                max_length,
                min_value,
                max_value,
                concat_prob,
                reuse_prob,
                list_update_prob,
                reverse_prob,
                substitute_prob,
                exclude_substition_prob,
                subsequence_prob,
                repeat_list_prob,
                math_update_prob,
                min_max_prob,
                add_subtract_prob,
                modulo_prob,
                filter_update_prob,
                filter_sequences_prob,
                two_list_update_prob,
                two_list_sub_add_prob,
                two_list_modulo_prob,
                scan_add_prob,
            )
        program_bytes = len(literal_eval(program_output.split("#")[-1].strip()))
        num_bytes += program_bytes
        # print(program_output)
        assert execute_and_validate_program(program_output)
        programs.append(program_output)
        i += 1
        if num_bytes // 1000 > last_1k_bytes:
            last_1k_bytes = num_bytes // 1000
            print(f"# bytes: {num_bytes}")

        if num_bytes > num_bytes_to_stop:
            break
        if i >= num_examples_to_stop * 1.2:
            break

    data = []
    calc_bits = False
    all_funcs = {
        w
        for z in [
            [
                x.split("= ")[-1].split("(")[0]
                for x in p.split("\n")
                if x.startswith("seq")
            ]
            for p in programs
        ]
        for w in z
    }
    for p in tqdm(programs):
        data_point = {"program": p, "sequence": literal_eval(p.split("#")[-1].strip())}
        if calc_bits:
            data_point["program_bits"] = get_num_bits_per_program_uniform(
                p, len(all_funcs), max_value - min_value + 1
            )

            program_func = {
                x.split("= ")[-1].split("(")[0]
                for x in p.split("\n")
                if x.startswith("seq")
            }
            for f in all_funcs:
                data_point[f] = f in program_func
        data.append(data_point)

    # Iterate over each item
    unique_data = {}
    for item in data:
        sequence = " ".join([str(z) for z in item["sequence"]])

        if sequence not in test_sequences:
            program = item["program"]

            # Check if the sequence is already in the dictionary
            if sequence not in unique_data or len(program) < len(
                unique_data[sequence]["program"]
            ):
                unique_data[sequence] = item
        else:
            print(f"filtering: {sequence}")

    # Convert the dictionary back to a list of dictionaries if needed
    data = list(unique_data.values())[:num_examples_to_stop]
    pd.DataFrame(data).to_csv(output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--test_examples_path",
        type=str,
        default="src/synthetic_data_generation/data/synthetic_test_data_v1.0.csv",
        help="File with test examples if exists. Test sequences will not be part of the training data.",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="src/synthetic_data_generation/data/generated_data_v1.0.csv",
        help="Output csv file",
    )
    args = parser.parse_args()
    main(args.num_examples, args.test_examples_path, args.output_file_path)
