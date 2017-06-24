import sugartensor as tf
import numpy as np

__author__ = 'georgi.val.stoyan0v@gmail.com'

#
# hyper parameters
#

embedding_dim = 32  # embedding dimension
latent_dim = 64  # hidden layer dimension
num_blocks = 3  # dilated blocks
reg_type = 'l2'  # type of regularization used
default_dout = 0.2  # define the default dropout rate


# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):
    # default rate
    opt += tf.sg_opt(size=3, rate=1, causal=False, is_first=False, dout=0)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]

    with tf.sg_context(name='block_%d_%d' % (opt.block, opt.rate)):
        # reduce dimension
        input_ = (tensor
                  .sg_bypass(act='relu', ln=(not opt.is_first), name='bypass')  # do not
                  .sg_conv1d(size=1, dim=in_dim / 2, act='relu', ln=True, regularizer=reg_type, name='conv_in'))

        # 1xk conv dilated
        out = (input_
               .sg_aconv1d(size=opt.size, rate=opt.rate, dout=opt.dout, causal=opt.causal, act='relu', ln=True,
                           regularizer=reg_type, name='aconv'))

        # dimension recover and residual connection
        out = out.sg_conv1d(size=1, dim=in_dim, regularizer=reg_type, name='conv_out') + tensor

    return out


# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)


#
# cnn classifier graph ( causal convolution )
#

def classifier(x, num_classes, voca_size, test=False):

    with tf.sg_context(name='classifier'):
        dropout = 0 if test else default_dout
        res = x.sg_conv1d(size=1, dim=latent_dim, ln=True, regularizer=reg_type, name='decompressor')

        # loop dilated causal conv block
        for i in range(num_blocks):
            res = (res
                   .sg_res_block(size=3, block=i, rate=1, causal=False, is_first=True)
                   .sg_res_block(size=3, block=i, rate=2, causal=False, ))
                   #.sg_res_block(size=3, block=i, rate=4, causal=False, )
                   #.sg_res_block(size=3, block=i, rate=8, causal=False))

        in_dim = res.get_shape().as_list()[-1]
        res = res.sg_conv1d(size=1, dim=in_dim, dout=dropout, ln=True, regularizer=reg_type, name='conv_dout_final')

        # final fully convolution layer for softmax
        res = res.sg_conv1d(size=1, dim=num_classes, act='relu', ln=True, regularizer=reg_type, name='conv_relu_final')

        # perform max over time pooling
        res = res.sg_max(axis=[1])

    return res


def init_custom_embeddings(name, embeddings_matrix, summary=True,
                           trainable=False):
    '''
    Initializes the embedding vector with custom preloaded embeddings
    '''

    embedding = np.array(embeddings_matrix)
    w = tf.get_variable(name, shape=embedding.shape,
                        initializer=tf.constant_initializer(embedding),
                        trainable=trainable)

    if summary:
        tf.sg_summary_param(w)

    # 1st row should be zero and not be updated by backprop because of zero padding.
    # emb = tf.concat([tf.zeros((1, embedding.shape[1]), dtype=tf.sg_floatx), w], 0)
    emb = w

    return emb
