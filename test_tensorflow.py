import sugartensor as tf
from kaggle_loader import KaggleLoader

data = KaggleLoader(['data/kaggle_popcorn_challenge/test.tsv'])
x, y = data.source, data.target

hello = tf.constant('The End!')

with tf.Session() as sess:
    initializer = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(initializer)
    coord = tf.train.Coordinator()
    qr = tf.train.QueueRunner(data.shuffle_queue, [data.enqueue_op] * data.num_threads)
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    threads = tf.train.start_queue_runners(coord=coord)
    try:
        print(sess.run([x, y]))
        print(sess.run([x, y]))
        print(sess.run([x, y]))

        print(hello.eval())
    except tf.errors.OutOfRangeError as ex:
        coord.request_stop(ex=ex)
    finally:
        coord.request_stop()
        coord.join(threads)
        coord.join(enqueue_threads)
