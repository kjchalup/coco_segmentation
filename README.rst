`MS COCO`_ segmentation using deep fully-convolutional networks.
##############################

.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: http://www.gnu.org/licenses/gpl-3.0
    :alt: GPL 3.0 License

`MS COCO`_ is a challenging computer vision dataset that contains segmentation, bounding box, and caption annotations. There are often multiple instances of multiple object classes in each image (that's why it's called "common objects *in context*".) Objects occlude each other, and
are often either tiny, or zoomed-in on so much that only a part of the object is visible. Below are randomly picked examples of MS COCO images containing 'person', and the ground-truth segmentation masks. It's best to download the figure and inspect it full-res. All the images are rescaled to 224x224 pixels.

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/coco_examples.png
        :alt: Example MS COCO images of 'person'.
        :align: center

In red, I marked particularly problematic instances. Is a hand or shoes a 'person', for example? In addition, note that sometimes (first image, first row) a reflection of a 'person' counts as 'person'. Sometimes (fourth image, first row) it doesn't. Keeping this in mind, we should be suspicious should any algorithm achieve perfect agreement with the masks.

I wanted to build a (relatively) simple neural net that could approach the problem of segmenting out 'person' from such images. I was inspired by Ross Girschick's `recent results`_ on joint detection / classification /segmentation: whereas Ross shows encouraging results, his article quietly ignores problematic cases like the above. Since I wanted to keep things simple, I restricted the task to segemtnation of 'person'. 

The Neural Network
------------------
My approach is inspired by Ross's work, as well as the older `Fully Convolutional Networks for Semantic Segmentation`_ by Trevor Darrell et al. Both use a pre-trained feature extractor to build upon, Zisserman's `VGG`_ network. I downloaded `Tensorflow`_ weights for VGG from https://github.com/machrisaa/tensorflow-vgg and, after some thinking, set up the following architecture:

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/architecture.png
        :alt: Segmentation net architecture.
        :align: center

First, the network pushes an image through the first 13 convolutional layers of VGG. At layer *pool1*, the image is downsampled from 224x224 to 112x112 pixels. At *pool2*, to 56x56 pixels. At *pool3*, to 28x28 pixels. The receptive field sizes increase until at *pool3* each pixel looks at about a quarter of the original image.

Darell's work showed that a pixel-to-pixel segmentation can benefit from access to both low-level and high-level information. In the *upscale* layers, I use transpose convolutions to upscale each of *pool1*, *pool2* and *pool3* back to 224x224 images: 

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/upscale.png
        :alt: Segmentation net architecture.
        :align: center

Then, using a trick similar to `Inception`_'s bottleneck layers, I stack all these upscaled feature maps depth-wise and use 1x1 convolutions to reduce the depth in the *vgg_concat* layer. The output of *vgg_concat* layer is a stack of 224x224 feature maps that have access to low-level, mid-level and high-level VGG features. On top of this layer, I put four layers of *batch normalization* followed by *relu* nonlinearity followed by 5x5 convolution:

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/convlayers.png
        :alt: Segmentation net architecture.
        :align: center


The VGG layers are kept constant, all the remaining layers are trainable. The final ingredient is the loss function. The binary masks from MS coco could be compared to the output of the net (which is squashed through a sigmoid nonlinearity) using binary cross-entropy. However, I chose to create my own loss function that is easier to debug and track:

.. code-block:: python

    def define_loss(self):                         
        loss_pos = tf.reduce_mean(tf.nn.sigmoid( 
            -tf.boolean_mask(self.y_pred, self.y_tf)))
        loss_neg = tf.reduce_mean(tf.nn.sigmoid(
            tf.boolean_mask(self.y_pred, tf.logical_not(self.y_tf))))
        tf.summary.scalar('loss_pos', loss_pos)    
        tf.summary.scalar('loss_neg', loss_neg)    
        return loss_pos + loss_neg       

Note that this assumes self.y_pred is a boolean tensor. The *loss_pos* element is the (negated) average activation our neural net assigns to the positive ground-truth pixels. The smaller loss_pos, the closer our net gets to assigning *1* to 'person' pixels. The *loss_neg* element does the converse: the smaller loss_neg, the closer the net gets to assigning *0* to the background pixels.

By formulating the loss this way, I make the loss invariant to the percentage of 'person' pixels in an image: failing to find a tiny 'person' is penalized as much as failing to find 'person' taking up the whole image. In addition, we can track both elements in `Tensorboard`_:

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/loss.png
        :alt: Training curves.
        :align: center

These training loss curves show that the positive loss saturates at a lower level than negative loss -- that is, we can expect the network to have higher recall than precision. In addition, saturation of the training loss at a non-zero level suggests that we could use a larger network, or different hyperparameters (e.g. learning rate schedule) to get better results.

Results and Conclusion
-------
This network took up my whole Titan X GPU with 12GB of RAM. After the loss saturated I chose not to train further, as the results were satisfactory:

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/segmentation_results.png
        :alt: MS COCO segmentation results.
        :align: center

Some remarks regarding the results:
    * The **Intersection over Union (IoU)** is a standard measure of segmentation results. On test data, our algorithm achieves mean **IoU ~ .56** (after thresholding the nn output at .5). In addition, the **fraction of images with IoU greater than .5 is .58**. Pretty good!
    * The pos / neg loss discrepancy suggests that it should have greater recall than precision. Indeed: average **Intersection(ground truth, pred) / Area(ground truth)**  of our algorithm is **.85**. A reasonable idea would be to retrain the network, putting more weight on loss_neg to shrink the false positive area.
    * The network doesn't seem to have much trouble detecting small instances, or instances of only parts of 'person'.
    * The rectangular grid artifacts in some of the segmentation maps result from the transpose convolution upscaling. They could easily be smoothed post-hoc. A better solution would be to use larger transpose convolution filters. For example, the *pool3* layer is upscaled 32x and would ideally use filters of diameter larger than 32. Unforunately, a larger GPU would be necessary to store such large filters.
  
.. _Inception: https://arxiv.org/abs/1512.00567  
.. _VGG: https://arxiv.org/pdf/1409.1556.pdf
.. _recent results: https://arxiv.org/pdf/1703.06870.pdf
.. _MS COCO: http://mscoco.org/
.. _Fully Convolutional Networks for Semantic Segmentation: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
.. _numpy: http://www.numpy.org/
.. _scikit-learn: http://scikit-learn.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _Tensorboard: https://www.youtube.com/watch?v=eBbEDRsCmv4
.. _Keras: https://keras.io/
.. _nn.py: neural_networks/nn.py
.. _mtn.py: neural_networks/mtn.py
.. _gan.py: neural_networks/gan.py
.. _cgan.py: neural_networks/cgan.py
.. _fcnn.py: neural_networks/fcnn.py
.. _arXiv:1207.0580: https://arxiv.org/pdf/1207.0580.pdf)
.. _arXiv:1512.03385: https://arxiv.org/pdf/1512.03385.pdf
.. _arXiv:1505.00387: https://arxiv.org/pdf/1505.00387.pdf
.. _arXiv:1611.04076v2: https://arxiv.org/abs/1611.04076v2
.. _arXiv:1411.1784: https://arxiv.org/abs/1411.1784
