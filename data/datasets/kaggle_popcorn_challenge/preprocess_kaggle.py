from data.preprocessors.kaggle_preprocessor import KagglePreprocessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


def do_magic(path, file_name, delim, column_name, vocabulary_size, max_data_length, test_split):
    preprocessor = KagglePreprocessor(path, file_name, delim, vocabulary_size, max_data_length, test_split)
    preprocessor.read_file()
    preprocessor.apply_preprocessing(column_name)
    preprocessor.save_preprocessed_file()
