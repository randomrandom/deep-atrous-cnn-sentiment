from data.preprocessors.kaggle_preprocessor import KagglePreprocessor

def do_magic(path, file_name, delim, test_percentage, column_name):
    preprocessor = KagglePreprocessor(path, file_name, delim, test_percentage)
    preprocessor.read_file()
    preprocessor.apply_preprocessing(column_name)
    preprocessor.save_preprocessed_file()