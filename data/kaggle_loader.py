import sugartensor as tf

from data.base_data_loader import BaseDataLoader
from data.preprocessors.kaggle_preprocessor import KagglePreprocessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


class KaggleLoader(BaseDataLoader):
    _name = 'KaggleLoader'
    TSV_DELIM = '\t'
    DATA_COLUMN = 'review'

    def __init__(self, bucket_boundaries, file_names, *args, **kwargs):
        self.__file_preprocessor = None

        self.field_delim = KaggleLoader.TSV_DELIM
        self.file_names = file_names

        record_defaults = [["0"], [0], [""]]
        skip_header_lines = 1
        data_column = KaggleLoader.DATA_COLUMN

        super().__init__(record_defaults, self.field_delim, data_column, bucket_boundaries, file_names, *args,
                         skip_header_lines=skip_header_lines, **kwargs)

        self.source, self.target = self.get_data()

        data_size = self.test_size if self._used_for_test_data else self.train_size
        self.num_batches = data_size // self._batch_size

    def _read_file(self, filename_queue, record_defaults, field_delim=BaseDataLoader._CSV_DELIM,
                   skip_header_lines=BaseDataLoader._DEFAULT_SKIP_HEADER_LINES):
        """
        Reading of the Kaggle TSV file
        :param filename_queue: 
        :return: single example and label
        """

        reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
        key, value = reader.read(filename_queue)

        _, label, example = tf.decode_csv(value, record_defaults, field_delim)

        return example, label

    def _preprocess_example(self, example):
        return example

    def build_eval_graph(self, x):
        # convert to tensor of strings
        split_example = tf.string_split(x, " ")

        # convert sparse to dense
        tensor_entry = tf.sparse_tensor_to_dense(split_example, default_value="")

        tensor_entry = self.table.lookup(tensor_entry)

        return tensor_entry

    def process_console_input(self, entry):
        if self.__file_preprocessor is None:
            voca_path, voca_name = BaseDataLoader._split_file_to_path_and_name(
                self.file_names[0])  # TODO: will be break with multiple filenames
            voca_name = KagglePreprocessor.VOCABULARY_PREFIX + voca_name
            self.__file_preprocessor = KagglePreprocessor(voca_path, voca_name, self.field_delim,
                                                          self.DEFAULT_VOCABULARY_SIZE, self.DEFAULT_MAX_DATA_LENGTH,
                                                          self.DEFAULT_TEST_SPLIT)

        entry = self.__file_preprocessor.preprocess_single_entry(entry)

        return entry
