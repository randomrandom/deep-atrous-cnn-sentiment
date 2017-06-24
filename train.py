import sugartensor as tf
from model.model import *
from model.trainer import classifier_train

from data.kaggle_loader import KaggleLoader

BUCKETS = [100, 200, 300, 400, 500]
TRAIN_DATA_FILES = ['data/datasets/kaggle_popcorn_challenge/labeledTrainData.tsv']
TEST_DATA_FILES = ['data/datasets/kaggle_popcorn_challenge/testData.tsv']
NUM_LABELS = 2

data = KaggleLoader(BUCKETS, 15000, TRAIN_DATA_FILES)
validation = KaggleLoader(BUCKETS, 10000, TEST_DATA_FILES)

x, y = data.source, data.target
val_x, val_y = validation.source, validation.target

# session with multiple GPU support
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

emb = tf.sg_emb(name='emb', voca_size=data.vocabulary_size, dim=embedding_dim)
z_x = x.sg_lookup(emb=emb)
v_x = val_x.sg_lookup(emb=emb)

with tf.sg_context(name='model'):
    train_classifier = classifier(z_x, NUM_LABELS, data.vocabulary_size)

    # cross entropy loss with logit
    loss = train_classifier.sg_ce(target=y)

# with tf.sg_context(name='model', reuse=True):
#     test_classifier = classifier(v_x, NUM_LABELS, validation.vocabulary_size)
#
#     # accuracy evaluation (validation set)
#     acc = (test_classifier.sg_softmax()
#                     .sg_accuracy(target=val_y,name='val'))
#
#     # validation loss
#     val_loss = (test_classifier.sg_ce(target=val_y))

# train
classifier_train(sess=sess, log_interval=50, lr=2e-1, loss=loss, #eval_metric=[acc, val_loss],
        ep_size=data.num_batches, max_ep=150, early_stop=False, data=data)