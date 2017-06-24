from data.datasets.kaggle_popcorn_challenge import preprocess_kaggle
from data.kaggle_loader import KaggleLoader

preprocess_kaggle.do_magic('./data/datasets/kaggle_popcorn_challenge/', 'labeledTrainData.tsv', KaggleLoader.TSV_DELIM,
                           .2, KaggleLoader.DATA_COLUMN)
