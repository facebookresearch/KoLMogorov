# Experiments

## Inference
To reproduce the experiments with prompted models, follow the below steps:

- Download the input data parsed to sub-sequences of length 128 from [here](s3_link).
- Set the relevent inference server with [vLLM](https://github.com/vllm-project/vllm).
- Run the relevant [`inference script`](inference/inference_vllm.py).
- For evaluation, see below.

## Training
To train your own `SeqCoder` models, please see our training input-output pairs [here](link_to_training_data). We recommend using the [SFTTrainer from HuggingFace](https://huggingface.co/docs/trl/en/sft_trainer), which also supports the [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model used to train `Seq-Coder-8B`.

## Evaluation

For latest detials and for making new submissions, please refer to the [official KT leaderboard](https://huggingface.co/spaces/KoLMogorov-Test/Leaderboard).

We provide code to reproduce experiments from the paper in our our [evaluation source code](evaluation). The [`post_process_and_evaluate.py`](evaluation/post_process_and_evaluate.py) script has two main components - (a) executing the generated programs (using [`post_process_programs.py`](evaluation/post_process_programs.py)) and calculating the evaluation metrics (using [`evaluate.py`](evaluation/evaluate.py)).
