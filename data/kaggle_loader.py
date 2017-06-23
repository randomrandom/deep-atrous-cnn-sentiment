import sugartensor as tf

from data.data_loader import DataLoader


class KaggleLoader(DataLoader):
    _name = 'KaggleLoader'
    _TSV_DELIM = '\t'

    def __init__(self, file_names, num_threads=DataLoader._num_threads, batch_size=DataLoader._batch_size,
                 min_after_dequeue=DataLoader._min_after_dequeue, capacity=DataLoader._capacity, name=_name):
        record_defaults = [["0"], [0], [""]]
        field_delim = KaggleLoader._TSV_DELIM
        skip_header_lines = 1

        super().__init__(file_names, record_defaults, field_delim, skip_header_lines, num_threads, batch_size,
                         min_after_dequeue, capacity, name)

        self.source, self.target = self.get_data()

    def _read_file(self, filename_queue, record_defaults, field_delim=DataLoader._CSV_DELIM,
                   skip_header_lines=DataLoader._DEFAULT_SKIP_HEADER_LINES):
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