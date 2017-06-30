from data.kaggle_loader import KaggleLoader
from model.model import *
from model.trainer import classifier_train

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 64

BUCKETS = [100, 170, 240, 290, 340]
DATA_FILE = ['data/datasets/kaggle_popcorn_challenge/labeledTrainData.tsv']
NUM_LABELS = 2

data = KaggleLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)
validation = KaggleLoader(BUCKETS, DATA_FILE, used_for_test_data=True, batch_size=BATCH_SIZE)

x, y = tf.split(data.source, tf.sg_gpus()), tf.split(data.target, tf.sg_gpus())
val_x, val_y = tf.split(validation.source, tf.sg_gpus()), tf.split(validation.target, tf.sg_gpus())

# session with multiple GPU support
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# setup embeddings, preload pre-trained embeddings if needed
emb = None
embedding_name = 'emb'

if use_pre_trained_embeddings:
    embedding_matrix = data.preload_embeddings(embedding_dim, pre_trained_embeddings_file)
    emb = init_custom_embeddings(name=embedding_name, embeddings_matrix=embedding_matrix, trainable=True)
else:
    emb = tf.sg_emb(name=embedding_name, voca_size=data.vocabulary_size, dim=embedding_dim)

data.visualize_embeddings(sess, emb, embedding_name)


# setup the model for training and validation. Enable multi-GPU support
@tf.sg_parallel
def get_train_loss(opt):
    with tf.sg_context(name='model'):
        z_x = opt.input[opt.gpu_index].sg_lookup(emb=emb)

        train_classifier = classifier(z_x, NUM_LABELS, data.vocabulary_size)

        # cross entropy loss with logit
        loss = train_classifier.sg_ce(target=opt.target[opt.gpu_index])

        return loss


@tf.sg_parallel
def get_val_metrics(opt):
    with tf.sg_context(name='model', reuse=True):
        tf.get_variable_scope().reuse_variables()

        v_x = opt.input[opt.gpu_index].sg_lookup(emb=emb)

        test_classifier = classifier(v_x, NUM_LABELS, validation.vocabulary_size)

        # accuracy evaluation (validation set)
        acc = (test_classifier.sg_softmax()
               .sg_accuracy(target=opt.target[opt.gpu_index], name='accuracy'))

        # validation loss
        val_loss = (test_classifier.sg_ce(target=opt.target[opt.gpu_index], name='validation'))

        return acc, val_loss


# train
classifier_train(sess=sess, log_interval=50, lr=1e-3, loss=get_train_loss(input=x, target=y)[0],
                 eval_metric=get_val_metrics(input=val_x, target=val_y)[0],
                 ep_size=data.num_batches, max_ep=10, early_stop=False, data=data)
