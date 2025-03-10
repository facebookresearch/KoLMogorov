# Synthetic data generation

To generate synthetic data with our DSL, see our [data generation script](generate_data.py). The main method is `get_program` which generates a program-sequence pair.

See `get_num_bits_per_program_uniform` in [utils](src/utils.py) for the method that calculates the size of the encoding of a program from our DSL wiht a uniform  prior over all programs in the DSL.

Note that to prevent cases sequences appear in test and train, we filter test sequences when generating training examples. For simplicity, we provide a file with our [test data](data/synthetic_test_data_v1.0.csv) as part of this repo.
