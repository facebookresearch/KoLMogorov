import json
from tqdm import tqdm
from openai import OpenAI
import os
import concurrent.futures
import argparse

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument("--input_file_path", type=str, help="Input file path")
parser.add_argument("--output_file_path", type=str, help="Output file path")
parser.add_argument("--host_name", type=str, help="Path to the vLLM server")
parser.add_argument("--port", type=str, help="Path to the vLLM server port")
parser.add_argument("--model", type=str, help="Name of inference model")
parser.add_argument("--first_ind", type=int, default=0)
parser.add_argument("--num_examples", type=int, default=1000)
parser.add_argument("--max_output_tokens", type=int, default=1024)
parser.add_argument("--cot", type=bool, default=False, help="CoT prompt")

args = parser.parse_args()

input_file_path = args.input_file_path
output_file_path = args.output_file_path
num_examples = args.num_examples
host_name = args.host_name
port = args.port
first_ind = args.first_ind
model = args.model
use_cot = args.cot


print(f"input_file_path: {input_file_path}")
print(f"output_file_path: {output_file_path}")
print(f"num_examples: {num_examples}")
print(f"host_name: {host_name}")
print(f"port: {port}")
print(f"model: {model}")
print(f"use_cot: {use_cot}")


if os.path.exists(output_file_path):
    raise FileExistsError(
        f"The file {output_file_path} already exists. Please remove it or specify a different file."
    )

# model params
client = OpenAI(base_url=f"http://{host_name}:{port}/v1", api_key="EMPTY")
prompt_user_message_no_cot = """### Instructions:
- Write a multi-line Python program. Each line should either assign a new variable or define a new function. These variables and functions can be reused throughout the program.
- Identify and utilize patterns in the input sequence to minimize the length of the program.
- Assign the final output of the sequence to the variable `output`. This output will be used to verify the correctness of the program. Do not include print statements or return statements.
- Ensure that the generated code is executable in a Python interpreter without modifications. Do not include the `python` code block syntax in your response.
- End your response with `###`.

### Input Sequence:
#SEQ#

### Expected Output:
The Python program that generates the sequence is:"""

prompt_user_message_with_cot = """### Instructions:
- Write a multi-line Python program. Each line should either assign a new variable or define a new function. These variables and functions can be reused throughout the program.
- Identify and utilize patterns in the input sequence to minimize the length of the program.
- Assign the final output of the sequence to the variable `output`. This output will be used to verify the correctness of the program. Do not include print statements or return statements.
- Ensure that the generated code is executable in a Python interpreter without modifications. Do not include the `python` code block syntax in your response.
- End your response with `###`.
- Before the program, you can use the Thought field to generate how you think the task should be solved. After the thought, generate "\nThe Python program that generates the sequence is:\n", followed by the program.

### Input Sequence:
#SEQ#

### Thought:"""

prompt_user_message = (
    prompt_user_message_with_cot if use_cot else prompt_user_message_no_cot
)
messages = [
    {
        "role": "system",
        "content": "Generate a Python program that, when executed, reproduces a specified input sequence. The program should be as concise as possible.",
    },
    {"role": "user", "content": ""},
]


# get data
data = []
with open(input_file_path, "r") as file:
    for line in file:
        data_point = json.loads(line.strip())
        data.append(data_point)
model_inputs = [str(x["sequence"]) for x in data]


def fetch_completion(sequence, file):
    local_messages = [
        {
            "role": "system",
            "content": "Generate a Python program that, when executed, reproduces a specified input sequence. The program should be as concise as possible.",
        },
        {"role": "user", "content": prompt_user_message.replace("#SEQ#", sequence)},
    ]
    try:
        completion = client.chat.completions.create(
            model=model, messages=local_messages, max_tokens=args.max_output_tokens
        )
        result = {
            "prompt": local_messages[1]["content"],
            "generation": completion.choices[0].message.content,
        }
        print(f"Prompt:\n{local_messages[1]['content']}")
        print(f"Response:\n{completion.choices[0].message.content}")
        result_json = json.dumps(result)
        file.write(result_json + "\n")
    except Exception as e:
        print(f"Error processing sequence: {sequence}. Error: {str(e)}")
        return {}
    return result_json


with open(output_file_path, "a") as file:
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(
            tqdm(
                executor.map(
                    lambda seq: fetch_completion(seq, file),
                    model_inputs[first_ind:num_examples],
                ),
                total=num_examples,
            )
        )
