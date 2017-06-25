from model.model import *
from data.kaggle_loader import KaggleLoader


__author__ = 'georgi.val.stoyan0v@gmail.com'


batch_size = 1

BUCKETS = [100, 200, 300, 400, 500]
DATA_FILE = ['data/datasets/kaggle_popcorn_challenge/labeledTrainData.tsv']
NUM_LABELS = 2

data = KaggleLoader(BUCKETS, 20000, DATA_FILE) # TODO: determine dataset size dynamically

# setup input pipeline
x = tf.placeholder(dtype=tf.string, shape=batch_size)
preprocessed_x = data.build_eval_graph(x)

emb = tf.sg_emb(name='emb', voca_size=data.vocabulary_size, dim=embedding_dim)
z_x = preprocessed_x.sg_lookup(emb=emb)

# setup classifier
with tf.sg_context(name='model'):
    cls = classifier(z_x, NUM_LABELS, data.vocabulary_size)

# get positiveness / negativeness score
score = cls.sg_softmax()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # init session vars
    tf.sg_init(sess)
    sess.run(tf.tables_initializer())

    tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train'))

    print("Enter text: \n>", end='', flush=True)
    for line in tf.sys.stdin:

        sources = data.process_console_input(line)
        out_score = sess.run(score, {x: [sources]})

        print('> Sentiment score: ', out_score[0][1], "\n", end='\n>', flush=True)