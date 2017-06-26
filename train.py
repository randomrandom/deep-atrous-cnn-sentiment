from model.model import *
from model.trainer import classifier_train
from data.kaggle_loader import KaggleLoader


__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 32

BUCKETS = [100, 170, 240, 290, 340]
DATA_FILE = ['data/datasets/kaggle_popcorn_challenge/labeledTrainData.tsv']
NUM_LABELS = 2

data = KaggleLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)
validation = KaggleLoader(BUCKETS, DATA_FILE, used_for_test_data=True, batch_size=BATCH_SIZE)

x, y = data.source, data.target
val_x, val_y = validation.source, validation.target

# session with multiple GPU support
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# setup embeddings, preload pre-trained embeddings if needed
emb = None
if use_pre_trained_embeddings:
    embedding_matrix = data.preload_embeddings(embedding_dim, pre_trained_embeddings_file)
    emb = init_custom_embeddings(name='emb_x', embeddings_matrix=embedding_matrix)
else:
    emb = tf.sg_emb(name='emb', voca_size=data.vocabulary_size, dim=embedding_dim)

z_x = x.sg_lookup(emb=emb)
v_x = val_x.sg_lookup(emb=emb)

with tf.sg_context(name='model'):
    train_classifier = classifier(z_x, NUM_LABELS, data.vocabulary_size)

    # cross entropy loss with logit
    loss = train_classifier.sg_ce(target=y)

with tf.sg_context(name='model', reuse=True):
    test_classifier = classifier(v_x, NUM_LABELS, validation.vocabulary_size)

    # accuracy evaluation (validation set)
    acc = (test_classifier.sg_softmax()
                    .sg_accuracy(target=val_y,name='val'))

    # validation loss
    val_loss = (test_classifier.sg_ce(target=val_y))

# train
classifier_train(sess=sess, log_interval=50, lr=1e-3, loss=loss, eval_metric=[acc, val_loss],
        ep_size=data.num_batches, max_ep=150, early_stop=False, data=data)