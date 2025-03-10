import ast
import multiprocessing
import json
from tqdm import tqdm
from src.utils import read_jsonl, save_results_as_jsonl


def worker(code):
    try:
        local_scope = {}
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


def is_disallowed_code(i, code):
    # Parse the code into an AST
    try:
        tree = ast.parse(code)
    except:
        return True
    # Iterate over all nodes in the AST
    for node in ast.walk(tree):
        # Check if the node is a function call
        if isinstance(node, ast.Call):
            # Check if the function being called is named 'open'
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                return True
    return False


def parse_data_points(data_points, parse_from_string):
    if parse_from_string:
        data_points = [json.loads(x) for x in data_points]
    codes = [
        dp["generation"]
        .split("that generates the sequence is:")[-1]
        .replace("```", "")
        .split("###")[0]
        .split("python")[-1]
        for dp in data_points
    ]
    edge_cases_indexes = [i for i, x in enumerate(codes) if is_disallowed_code(i, x)]
    for i in edge_cases_indexes:
        codes[i] = ""
    outputs = []
    chunked_codes = [codes[i : i + 128] for i in range(0, len(codes), 128)]
    for chunk in tqdm(chunked_codes):
        outputs.extend(execute_with_timeout(chunk, timeout=2))

    results = []
    for i, (data_point, output) in enumerate(zip(data_points, outputs)):
        if isinstance(output, Exception):
            print(f"Error during execution: {output}")
            output = ""
        results.append(
            {
                "task_id": (
                    data_point["raw"]["task_id"] if "raw" in data_point else str(i)
                ),
                "sequence_input": ast.literal_eval(
                    data_point["prompt"].split(" Input Sequence:\n")[-1].split("\n")[0]
                ),
                "program": codes[i],
                "program_output": output,
            }
        )
    return results


def post_process_programs(data_input_path: str, output_file_path: str):

    data = read_jsonl(data_input_path)
    results = parse_data_points(data, parse_from_string=True)

    for res in results:
        if type(res["program_output"]) != list:
            res["program_output"] = ""
        elif len(res["program_output"]) > 256:
            res["program_output"] = ""
        elif len(res["program_output"]) > 0 and not all(
            isinstance(x, int) for x in res["program_output"]
        ):
            res["program_output"] = ""
    save_results_as_jsonl(results, output_file_path)
