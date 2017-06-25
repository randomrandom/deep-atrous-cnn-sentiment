from data.preprocessors.kaggle_preprocessor import KagglePreprocessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


def do_magic(path, file_name, delim, test_percentage, column_name, vocabulary_size):
    preprocessor = KagglePreprocessor(path, file_name, delim, test_percentage, vocabulary_size=vocabulary_size)
    preprocessor.read_file()
    preprocessor.apply_preprocessing(column_name)
    preprocessor.save_preprocessed_file()
