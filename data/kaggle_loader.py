import sugartensor as tf

from data.base_data_loader import BaseDataLoader


class KaggleLoader(BaseDataLoader):
    _name = 'KaggleLoader'
    TSV_DELIM = '\t'
    DATA_COLUMN = 'review'

    def __init__(self, bucket_boundaries, data_size, *args, **kwargs):
        record_defaults = [["0"], [0], [""]]
        field_delim = KaggleLoader.TSV_DELIM
        skip_header_lines = 1
        data_column = KaggleLoader.DATA_COLUMN

        super().__init__(record_defaults, field_delim, data_column, bucket_boundaries, *args,
                         skip_header_lines=skip_header_lines, **kwargs)

        self.source, self.target = self.get_data()
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
