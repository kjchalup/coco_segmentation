`MS COCO`_ segmentation using deep fully-convolutional networks.
##############################

.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: http://www.gnu.org/licenses/gpl-3.0
    :alt: GPL 3.0 License

`MS COCO`_ is by all means a challenging computer vision dataset. There are often multiple instances of multiple object classes in each image (that's why it's called "common objects *in context*".) Objects occlude each other, and
are often either tiny, or zoomed-in on so much that only a part of the object is visible. Below are randomly picked examples of thirty-two MS COCO images containing 'person'. I rescaled the images to 224x224 pixels, and marked in red particularily controversial annotations of 'person':

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/coco_examples.png
        :alt: Example MS COCO images of 'person'.
        :align: center

I wanted to build a (relatively) simple neural net that could approach the problem of segmenting out 'person' from `MS COCO`_ images. I was inspired by `Ross Girschick`_'s recent results on joint detection / classification /segmentation on the dataset: segmentation on this dataset is possible! Since I wanted to keep things simple, I stuck just to segmentation of one (albeit probably the most difficult) class of instances, 'person'. 

The Neural Network
------------------
My approach is inspired by Ross's work, as well as the older `Fully Convolutional Networks for Semantic Segmentation`_ by Trevor Darrell et. al. Both use a pre-trained feature extractor to build upon, the `VGG`_ network. I downloaded `Tensorflow`_ weights for VGG from https://github.com/machrisaa/tensorflow-vgg and set up my segmentation net as follows:

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/architecture1.png
        :alt: Segmentation net architecture.
        :align: center

First, [describe the VGG layers].

On top of that [describe the fully conv layers].

The loss [describe the loss].

Results and Conclusion
-------
This network took up my whole Titan X GPU with 12GB of RAM. After several hours, the sigmoid loss saturated at about .2. I chose not to train further, as the results were satisfactory:

    .. image:: https://github.com/kjchalup/coco_segmentation/blob/master/results.png
        :alt: MS COCO segmentation results.
        :align: center

The results are not pretty, but are actually very good in terms of the IoU (intersection over union) of the segmentations w.r.t. the ground truth. [explain why not pretty].

It turns out training a reasonable segmentation network is not that hard! With some additional work (tailoring the architecture, running a larger net on multiple GPUs, smoothing the results etc) the network would likely approach human-level performance on this challenging dataset.
   
.. _numpy: http://www.numpy.org/
.. _scikit-learn: http://scikit-learn.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _TensorBoard: https://www.youtube.com/watch?v=eBbEDRsCmv4
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
