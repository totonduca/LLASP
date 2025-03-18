# LLASP
LLASP: Fine-tuning Large Language Models for Answer Set Programming

The `data` folder contains the compressed version of the train, validation and test sets used to perform the experiments.

The code `fine_tuning.py` is the collection of model_training.py, model_testing.py, dataset_generation.py...

First of all you must choose the model, it could be:
    `base` -> Gemma-2b;
    `core` -> Gemma-2b trained on some ASP problems;
    `core-invariance` -> Gemma-2b-core trained on prompt invariance;
    `complex` -> Gemma-2b-core-invariance trained for complex problems.

If you want to regenerate the data, set `DATASET_GENERATION = True`, for train and validation sets, and `TEST = True and TEST_DATASET_GENERATION = True` for the test set.

Set `TRAIN = True` to fine-tune model (chosen before) and `LOAD = True, TEST = True and EXHAUSTIVE = True` to test its performance over the test set.

**For the core-invariance model, you can choose whether to test on 1 tuple of the 20 in the training set or on a 21st chosen at random* 

**You can also submit an easy test on 10 prompts, just set `EXHAUSTIVE = False`*

Then you can set `SHOW_RESULTS = True` to show the results of the exhaustive test.