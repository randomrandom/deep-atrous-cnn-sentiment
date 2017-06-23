from data.preprocessors.kaggle_preprocessor import KagglePreprocessor

def do_magic(path):
    preprocessor = KagglePreprocessor(path, 'test.tsv', '\t')
    preprocessor.apply_preprocessing('review')
    #preprocessor.apply_padding('review', [3, 6])
    preprocessor.save_preprocessed_file()