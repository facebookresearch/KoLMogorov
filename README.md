# The KoLMogorov-Test

Official repository for the paper [*"The KoLMogorov Test: Compression by Code Generation"*](https://openreview.net/forum?id=C45YqeBDUM).

The Kolmogorov complexity of a sequence is the length of the shortest computer program that produces the sequence.

The aim of the KoLMogorov-Test (KT) is to empirically evaluate the ability of CodeLMs to detect patterns in and compress sequences by writing short programs that output them.

## 👋 Getting started

To get started, first [download the data](src/data).

KT currently includes six modalities - text, DNA, three encodings of audio data (MFCC, 16-bit, and 8-bit), and synthetic sequences produced by random programs. Two dataset sizes are available: a small one with 1MB per modality, and a large one with 1GB (DNA and text only).

For more information on making a submission, please check our [leaderboard](https://huggingface.co/spaces/KoLMogorov-Test/Leaderboard).

## 🧪 Experiments
Please check out our [experiments source code](src/experiments) to run experiments from the paper. Specifically, we provide information on how to run inference with prompted models and how to train our specialized SeqCoder models. 

## 📈 Evaluation
The main evaluation is simply the compression rate of the encoded programs and the relevant decoder. Please see our [leaderboard](https://huggingface.co/spaces/KoLMogorov-Test/Leaderboard) and our [evaluation source code](src/experiments/evaluation) for more information.

## 🧠 Synthetic data generation
In the paper, we show that for synthetic distributions where sampling program-sequence pairs is possible, we can train models with lower compression rate than current approaches. Please see our [synthetic data generation source code](src/synthetic_data_generation) for our DSL and data generation scripts.

## ✍️ Citation
```
@inproceedings{
    anonymous2024the,
    title={The Ko{LM}ogorov Test: Compression by Code Generation},
    author={Anonymous},
    booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=C45YqeBDUM},
    note={under review}
}
```

If you have any questions, please email us at [thekolmogorovtest@gmail.com](thekolmogorovtest@gmail.com)
