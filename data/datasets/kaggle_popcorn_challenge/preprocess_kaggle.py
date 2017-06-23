from data.preprocessors.kaggle_preprocessor import KagglePreprocessor

def do_magic(path):
    preprocessor = KagglePreprocessor([3], path, 'test.tsv', '\t')
    preprocessor.apply_preprocessing('review')
    preprocessor.apply_padding('review')
    preprocessor.save_preprocessed_file()