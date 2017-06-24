## How to get data

Follow these instructions in order to download the dataset

```
# Navigate to the data folder
cd data/datasets/kaggle_popcorn/challenge
wget https://drive.google.com/uc?id=0B8QhNqWxSRsIWEIyNkJzdGxZUU0&export=download .
```

## Preprocessing

Since the model doesn't waste memory on loading the full dataset in memory, it is not practical to perform advanced preprocessing via tensorflow methods. That's why all files must be preprocessed in advance before being fed to the model. 

The model currently support reading from TSV files in the following format:
```
id  sentiment   review
"string_id" int_sentiment   "string_review"
```
The entries are tab separated.

The model should automatically apply the needed preprocessing (applies regex, vocabulary generation, meta data, etc.) and generate "cleaned" files before training initiation.

Alternatively you can use the following predefined script

```
# Navigate back to the root directory
cd $PROJECT_DIR

# run the preprocessing script
python3.5 preprocess.py
```


