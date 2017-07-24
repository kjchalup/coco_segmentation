""" Deep convolutional segmentation on MS COCO.

Note: This network takes up almost 12GB of GPU memory.
This could be reduced by precomputing the VGG pool layer
weights for the images.

Krzysztof Chalupka, 2017.
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neural_networks.fcnn import FCNN, bn_relu_conv
# Import the VGG16 network model (clone this repository,
# and download the weights into the same directory as
# described in their README: https://github.com/machrisaa/tensorflow-vgg).
sys.path.append('/home/kchalupk/projects/tensorflow-vgg')
import vgg16

from mscoco import get_coco_batch

class SegmentNN(FCNN):
    """ A segmentation neural network.
    
    For a detailed description of the architecture,
    see https://github.com/kjchalup/coco_segmentation.
    """

    def __init__(self, **kwargs):
        self.vgg16 = vgg16.Vgg16()
        FCNN.__init__(self, **kwargs)
    
    def predict(self, x, sess):
        """ Compute the output for given data.

        Args:
            x (n_samples, h, w, c): Input data.
            sess: Tensorflow session.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        feed_dict = {self.x_tf: x, self.is_training_tf: False}
        y_pred = sess.run(tf.nn.sigmoid(self.y_pred), feed_dict)
        return y_pred

    def define_fcnn(self, n_filters=4, n_layers=4, **kwargs):
        """ Define the segmentation network. """
        y_pred = self.x_tf
        self.vgg16.build(self.x_tf)

        with tf.variable_scope('vgg_pool1_upscale'):
            pool1_up = self.vgg16.pool1
            pool1_up = tf.nn.relu(tf.layers.batch_normalization(pool1_up,
                center=True, scale=True, training=self.is_training_tf))
            pool1_up = tf.layers.conv2d_transpose(
                pool1_up, filters=n_filters, kernel_size=(5, 5), strides=(2, 2),
                padding='same', activation=None, reuse=self.reuse)

        with tf.variable_scope('vgg_pool3_upscale'):
            pool3_up = self.vgg16.pool3
            pool3_up = tf.nn.relu(tf.layers.batch_normalization(pool3_up,
                center=True, scale=True, training=self.is_training_tf))
            pool3_up = tf.layers.conv2d_transpose(
                pool3_up, filters=n_filters, kernel_size=(7, 7), strides=(8, 8),
                padding='same', activation=None, reuse=self.reuse)

        with tf.variable_scope('vgg_pool5_upscale'):
            pool5_up = self.vgg16.pool5
            pool5_up = tf.nn.relu(tf.layers.batch_normalization(pool5_up,
                center=True, scale=True, training=self.is_training_tf))
            pool5_up = tf.layers.conv2d_transpose(
                pool5_up, filters=n_filters, kernel_size=(11, 11),
                strides=(32, 32), padding='same', activation=None,
                reuse=self.reuse)

        with tf.variable_scope('vgg_concat'):
            y_pred = tf.concat(
                [pool1_up, pool3_up, pool5_up], axis=3)

        with tf.variable_scope('convolutional'):
            for layer_id in range(n_layers):
                with tf.variable_scope('layer{}'.format(layer_id)):
                    y_pred = bn_relu_conv(y_pred, self.is_training_tf,
                        n_filters=n_filters, stride=(1, 1), bn=True,
                        kernel_size=(5, 5), residual=self.res)

        with tf.variable_scope('output'):
            y_pred = bn_relu_conv(y_pred, self.is_training_tf,
                n_filters=1, stride=(1, 1), bn=True,
                kernel_size=(1, 1), residual=self.res)

        tf.summary.histogram('prediction', y_pred)

        return y_pred

    def define_loss(self):
        loss_pos = tf.reduce_mean(tf.nn.sigmoid(-.1 * 
            tf.boolean_mask(self.y_pred, self.y_tf)))
        loss_neg = tf.reduce_mean(tf.nn.sigmoid(.1 *
            tf.boolean_mask(self.y_pred, tf.logical_not(self.y_tf))))
        tf.summary.scalar('loss_pos', loss_pos)
        tf.summary.scalar('loss_neg', loss_neg)
        return loss_pos + loss_neg
        # If you want to use cross-entropy instead:
        #return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    labels=self.y_tf, logits=self.y_pred))


if __name__ == "__main__":
    """ Check that the network works as expected. Denoise MNIST. 
    Takes about a minute on a Titan X GPU.
    """
    im_shape = [224, 224] # Images will be reshaped to this shape.
    batch_size = 128 # Shouldn't be small because of batch normalization.

    def fetch_data(batch_size, data_type):
        ims, masks = get_coco_batch(category='person', batch_size=batch_size,
            im_size=im_shape, data_type=data_type)
        return ims, masks.astype(bool)

    # Define the graph.
    y_tf = tf.placeholder(tf.bool, [None, im_shape[0], im_shape[1], 1])
    fcnn = SegmentNN(x_shape = im_shape + [3], y_tf=y_tf, y_channels=1,
        res=False, save_fname='logs/weights')

    # Create a Tensorflow session and train the net.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())
        # UNCOMMENT THIS TO LOAD PRE-SAVED WEIGHRS
        #fcnn.saver.restore(sess, 'logs/weights')
        fcnn.saver.restore(sess, 'logs/vgg_segment_ce2')

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/training/')
        writer.add_graph(sess.graph)

        # Fit the net.
        #fcnn.fit(sess, fetch_data, epochs=10000,
        #    batch_size=batch_size, lr=0.1, writer=writer, summary=summary)

        # Test the network.
        np.random.seed(1)
        ims_ts, masks_ts = fetch_data(32, 'test')
        Y_pred = fcnn.predict(ims_ts, sess)

        # Compute the IoU over 1000 test points.
        iou = 0
        iou_correct = 0
        iou_recall = 0
        for test_batch in range(10):
            ims, masks = fetch_data(100, 'test')
            ypred = fcnn.predict(ims, sess)
            ious = ((ypred * masks).sum(axis=(1, 2)) /
                    (ypred.astype(bool) + masks).sum(axis=(1, 2)).astype(float))
            iou += (ypred * masks).sum() / float((ypred.astype(bool) + masks).sum())
            iou_correct += (ious > .5).sum()
            iou_recall += (ypred * masks).sum() / float(masks.sum())
            
        print('Avg IoU = {}. IoU > .5 fraction = {}, IoU recall = {}'.format(
            iou / 10., iou_correct / 1000., iou_recall / 10.))


    # Show some results.
    plt.figure(figsize=(24, 24))
    for im_id in range(32):
        plt.subplot2grid((8, 8), (2 * (im_id / 8), im_id % 8))
        plt.axis('off')
        plt.imshow(ims_ts[im_id], interpolation='nearest', vmin=0, vmax=1)

        plt.subplot2grid((8, 8), (2 * (im_id / 8) + 1, im_id % 8))
        plt.axis('off')
        plt.imshow(ims_ts[im_id], interpolation='nearest', vmin=0, vmax=1)
        plt.imshow(Y_pred[im_id].squeeze() > .5, cmap='gray', alpha=.8)

    plt.savefig('segmentation_results.png')
