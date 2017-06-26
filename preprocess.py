from data.datasets.kaggle_popcorn_challenge import preprocess_kaggle
from data.kaggle_loader import KaggleLoader

__author__ = 'georgi.val.stoyan0v@gmail.com'

preprocess_kaggle.do_magic('./data/datasets/kaggle_popcorn_challenge/', 'labeledTrainData.tsv', KaggleLoader.TSV_DELIM,
                           KaggleLoader.DATA_COLUMN, KaggleLoader.DEFAULT_VOCABULARY_SIZE,
                           KaggleLoader.DEFAULT_MAX_DATA_LENGTH, KaggleLoader.DEFAULT_TEST_SPLIT)
