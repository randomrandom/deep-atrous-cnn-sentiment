from data.preprocessors.base_preprocessor import BasePreprocessor


class KagglePreprocessor(BasePreprocessor):

    def __init__(self, buckets, *args):
        super().__init__(*args)
        self.buckets = buckets

    def _custom_preprocessing(self, entry):

        return entry

    def _pad_entry(self, entry):

        entry_no_space = entry.split()
        for i in range(len(self.buckets) - 1, -1, -1):
            bucket_size = self.buckets[i]

            if bucket_size > len(entry_no_space):
                entry_no_space = entry_no_space + [self.pad_token] * (bucket_size - len(entry_no_space) - 1)
                break

        entry = ' '.join(entry_no_space)

        return entry

    def apply_padding(self, column_name):
        self.new_data = self.new_data[column_name].apply(lambda x: self._pad_entry(x))