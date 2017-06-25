from data.preprocessors.base_preprocessor import BasePreprocessor


class KagglePreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _custom_preprocessing(self, entry):

        return entry

    def _pad_entry(self, entry, bucket_boundaries):

        entry_no_space = entry.split()
        for i in range(len(bucket_boundaries)):
            bucket_size = bucket_boundaries[i]

            if bucket_size > len(entry_no_space):
                entry_no_space = entry_no_space + [self.pad_token] * (bucket_size - len(entry_no_space) - 1)
                break

        entry = ' '.join(entry_no_space)

        return entry

    def apply_padding(self, column_name, bucket_boundaries):
        self.new_data[column_name] = self.new_data[column_name].apply(lambda x: self._pad_entry(x, bucket_boundaries))
