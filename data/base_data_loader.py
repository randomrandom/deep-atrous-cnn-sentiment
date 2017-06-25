import ntpath
import sugartensor as tf
from abc import abstractclassmethod
from pathlib import Path

from data.preprocessors.kaggle_preprocessor import KagglePreprocessor

__author__ = 'george.val.stoyan0v@gmail.com'


class BaseDataLoader(object):
    _CSV_DELIM = ","
    _DEFAULT_SKIP_HEADER_LINES = 0
    _name = "data_loader"
    _num_threads = 1 # 32
    _batch_size = 4  # 64
    _min_after_dequeue = 8 #_batch_size * _num_threads
    _capacity = 100# #_min_after_dequeue + (_num_threads + 2) * _batch_size  # as recommended in tf tutorial

    def __init__(self, record_defaults, field_delim, data_column, bucket_boundaries, file_names,
                 skip_header_lines=_DEFAULT_SKIP_HEADER_LINES,
                 num_threads=_num_threads, batch_size=_batch_size, min_after_dequeue=_min_after_dequeue,
                 capacity=_capacity, used_for_test_data=False, name=_name):
        self.__file_names = file_names
        self.__field_delim = field_delim
        self.__record_defaults = record_defaults
        self.__skip_header_lines = skip_header_lines
        self.__data_column = data_column
        self.__bucket_boundaries = bucket_boundaries

        self.num_threads = num_threads

        self._used_for_test_data = used_for_test_data
        self._min_after_dequeue = min_after_dequeue
        self._batch_size = batch_size
        self._capacity = capacity
        self._name = name
        self.table = None

        self.vocabulary_size = 0

        self.shuffle_queue = tf.RandomShuffleQueue(capacity=self._capacity, min_after_dequeue=self._min_after_dequeue,
                                                   dtypes=[tf.int64, tf.int32], shapes=None)

    def get_data(self):
        return self.__load_batch(self.__file_names, record_defaults=self.__record_defaults,
                                 field_delim=self.__field_delim, data_column=self.__data_column,
                                 bucket_boundaries=self.__bucket_boundaries, skip_header_lines=self.__skip_header_lines,
                                 num_epochs=None, shuffle=True)

    @staticmethod
    def _split_file_to_path_and_name(file_name):
        file_path, tail = ntpath.split(file_name)
        file_path += '/'

        return file_path, tail

    def __generate_preprocessed_files(self, file_names, data_column, bucket_boundaries, field_delim=_CSV_DELIM):
        new_file_names = []
        for filename in file_names:
            file_path, tail = BaseDataLoader._split_file_to_path_and_name(filename)

            old_file_name = tail
            prefix = KagglePreprocessor.TEST_PREFIX if self._used_for_test_data else KagglePreprocessor.CLEAN_PREFIX
            file_name =  file_path + prefix + tail
            file = Path(file_name)
            new_file_names.append(file_name)

            print(file)
            if file.exists():
                try:
                    tf.os.remove(file_name)
                except OSError:
                    print("File not found %s" % file_name)

            self.__preprocess_file(file_path, old_file_name, field_delim, data_column, bucket_boundaries)

        return new_file_names

    def __preprocess_file(self, path, file_name, field_delim, data_column, bucket_boundaries):
        preprocessor = KagglePreprocessor(path, file_name, field_delim)
        preprocessor.read_file()
        preprocessor.apply_preprocessing(data_column)
        preprocessor.save_preprocessed_file()
        self.vocabulary_size = preprocessor.vocabulary_size

    def __load_batch(self, file_names, record_defaults, data_column, bucket_boundaries, field_delim=_CSV_DELIM,
                     skip_header_lines=0,
                     num_epochs=None, shuffle=True):

        original_file_names = file_names[:]
        file_names = self.__generate_preprocessed_files(file_names, data_column, bucket_boundaries,
                                                              field_delim=field_delim)

        filename_queue = tf.train.string_input_producer(
            file_names, num_epochs=num_epochs, shuffle=shuffle
        )

        example, label = self._read_file(filename_queue, record_defaults, field_delim, skip_header_lines)

        voca_path, voca_name = BaseDataLoader._split_file_to_path_and_name(
            original_file_names[0])  # TODO: will be break with multiple filenames
        voca_name = KagglePreprocessor.VOCABULARY_PREFIX + voca_name

        # load look up table that maps words to ids
        self.table = tf.contrib.lookup.index_table_from_file(vocabulary_file=voca_path + voca_name,
                                                        default_value=KagglePreprocessor.UNK_TOKEN_ID, num_oov_buckets=0)

        # convert to tensor of strings
        split_example = tf.string_split([example], " ")

        # determine lengths of sequences
        line_number = split_example.indices[:, 0]
        line_position = split_example.indices[:, 1]
        lengths = (tf.segment_max(data=line_position,
                                  segment_ids=line_number) + 1).sg_cast(dtype=tf.int32)

        # convert sparse to dense
        dense_example = tf.sparse_tensor_to_dense(split_example, default_value="")
        dense_example = self.table.lookup(dense_example)

        # get the enqueue op to pass to a coordintor to be run
        self.enqueue_op = self.shuffle_queue.enqueue([dense_example, label])
        dense_example, label = self.shuffle_queue.dequeue()

        # add queue to queue runner
        self.qr = tf.train.QueueRunner(self.shuffle_queue, [self.enqueue_op] * self.num_threads)
        tf.train.queue_runner.add_queue_runner(self.qr)

        # reshape from <unknown> shape into proper form after dequeue from random shuffle queue
        # this is needed so next queue can automatically infer the shape properly
        dense_example = dense_example.sg_reshape(shape=[1, -1])
        label = label.sg_reshape(shape=[1])

        _, (padded_examples, label_examples) = tf.contrib.training.bucket_by_sequence_length(lengths,
                                                                                             [dense_example, label],
                                                                                             batch_size=self._batch_size,
                                                                                             bucket_boundaries=bucket_boundaries,
                                                                                             dynamic_pad=True,
                                                                                             capacity=self._capacity,
                                                                                             num_threads=self._num_threads)

        # reshape shape into proper form after dequeue from bucket queue
        padded_examples = padded_examples.sg_reshape(shape=[self._batch_size, -1])
        label_examples = label_examples.sg_reshape(shape=[self._batch_size])

        return padded_examples, label_examples

    def _read_file(self, filename_queue, record_defaults, field_delim=_CSV_DELIM,
                   skip_header_lines=_DEFAULT_SKIP_HEADER_LINES):
        """
        Each class should implement this depending on the file format they want to read, default is csv
        :param filename_queue: 
        :return: 
        """

        reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
        key, value = reader.read(filename_queue)

        _, example, label = tf.decode_csv(value, record_defaults, field_delim)

        return example, label

    @abstractclassmethod
    def _preprocess_example(self, example):
        """
        Applies preprocessing where to the examples
        :param example: an example to preprocess_single_entry
        :return: the preprocessed example
        """

        return example
