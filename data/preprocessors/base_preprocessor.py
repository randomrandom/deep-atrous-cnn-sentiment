import collections

import pandas as pd
import re
from abc import abstractclassmethod


class BasePreprocessor(object):
    CLEAN_PREFIX = 'clean_'
    TEST_PREFIX = 'clean_test_'
    VOCABULARY_PREFIX = 'vocabulary_'
    UNK_TOKEN_ID = 1

    _METADATA_PREFIX = 'metadata_'
    _PAD_TOKEN = '<PAD>'
    _UNK_TOKEN = '<UNK>'
    _EOS_TOKEN = '<EOS>'
    _DEFAULT_TEST_SPLIT = .2

    _VOCABULARY_SIZE = 50000

    def __init__(self, path, filename, separator, test_split=_DEFAULT_TEST_SPLIT,
                 vocabulary_size=_VOCABULARY_SIZE, pad_token=_PAD_TOKEN,
                 unk_token=_UNK_TOKEN, eos_token=_EOS_TOKEN):
        self._regex = re.compile('[%s]' % re.escape(r"""#"$%&'()*+/:;<=>@[\]^_`{|}~"""))
        self._remove_space_after_quote = re.compile(r'\b\'\s+\b')
        self._add_space = re.compile('([.,!?()-])')
        self._remove_spaces = re.compile('\s{2,}')
        self._dictionary = {}

        self.path = path
        self.filename = filename
        self.separator = separator
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.vocabulary_size = vocabulary_size
        self.test_split = test_split
        self.data = None
        self.new_data = None

    def _build_dictionary(self, data, column_name):
        all_text = []

        for review in data[column_name]:
            all_text.extend(review.split())

        all_words = [(self.pad_token, -1), (self.unk_token, -1), (self.eos_token, -1)]

        assert all_words[BasePreprocessor.UNK_TOKEN_ID][0] == \
               self.unk_token, '<UNK> token id and actual position should match'

        all_words.extend(collections.Counter(all_text).most_common(self.vocabulary_size - 3))

        for word in all_words:
            if word[0] not in self._dictionary:
                self._dictionary[word[0]] = len(self._dictionary)

        word_column = 'Word'
        metadata = pd.DataFrame(data=all_words, columns=[word_column, 'Frequency'])
        self.vocabulary_size = len(self._dictionary)

        print('Built vocabulary with size: %d' % self.vocabulary_size)

        metadata.to_csv(self.path + self._METADATA_PREFIX + self.filename, sep=self.separator, index=False,
                        encoding='utf-8')
        print('Saved vocabulary to metadata file')
        metadata[word_column].to_csv(self.path + self.VOCABULARY_PREFIX + self.filename, sep=self.separator,
                                     index=False, encoding='utf-8')
        print('Saved vocabulary to vocabulary file')

    def save_preprocessed_file(self):
        assert self.new_data is not None, 'No preprocessing has been applied, did you call apply_preprocessing?'

        data_size = self.new_data.shape[0]
        train_size = (int)(data_size * (1 - self.test_split))

        self.new_data.iloc[:train_size, :].to_csv(self.path + self.CLEAN_PREFIX + self.filename, sep=self.separator,
                                                  index=False)
        self.new_data.iloc[train_size:, :].to_csv(self.path + self.TEST_PREFIX + self.filename, sep=self.separator,
                                                  index=False)
        print('Successfully saved preprocessed files')

    def apply_preprocessing(self, column_name):
        assert self.data is not None, 'No input data has been loaded'

        new_data = self.data.copy()
        new_data[column_name] = new_data[column_name].apply(lambda x: self.preprocess_single_entry(x))
        self._build_dictionary(new_data, column_name)

        self.new_data = new_data
        print('Applied preprocessing to input data')

    def preprocess_single_entry(self, entry):
        entry = self._regex_preprocess(entry)
        entry = self._custom_preprocessing(entry)

        return entry

    @abstractclassmethod
    def _custom_preprocessing(self, entry):
        """
        Apply custom preprocessing to single data entry. 
        :param entry: 
        :return: the entry after custom preprocessing
        """

        return entry

    def _regex_preprocess(self, entry):
        entry = self._add_space.sub(r' \1 ', entry)
        entry = self._regex.sub('', entry)
        entry = self._remove_space_after_quote.sub(r"'", entry)
        entry = self._remove_spaces.sub(' ', entry).lower().strip()

        return entry

    def read_file(self):
        self.data = pd.read_csv(self.path + self.filename, sep=self.separator)

        return self.data

    @staticmethod
    def read_vocabulary(file_path, separator):
        dictionary = pd.read_csv(file_path, sep=separator, header=None).to_dict()

        # remap value <> key to key <> value
        dictionary = {v: k for k, v in dictionary[0].items()}

        return dictionary
