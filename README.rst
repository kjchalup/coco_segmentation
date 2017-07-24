Neural nets in `TensorFlow`_.
##############################

.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: http://www.gnu.org/licenses/gpl-3.0
    :alt: GPL 3.0 License

`TensorFlow`_ is a relatively low-level framework for building and executing
computational graphs. There are higher-level frameworks built on top of `TensorFlow`_ that implement neural
networks (e.g. `Keras`_).

This repository implements one such high-level framework.
In research, I need low-level control over the computational graph,
but I also often reuse basic neural net architectures. This makes it
inconvenient to use pre-existing frameworks (lack of low-level control) --
but I don't want to reimplement basic nn components each time.

Why would you use this repository?

    * You are a Tensorflow beginner, and want to see how to implement stuff. I certainly learned a lot looking at other people's tf code!
    * If you want to use my implementations in your own projects please do, though you'll probably learn more and get best results if you do your own coding.

Usage
-----
I tried to mimic the `scikit-learn`_ interface. You fit a network
using nn.fit, and predict with nn.predict. In some cases
there are other useful methods, e.g. GANs can gan.sample().
See individual module documentation for more details.

Each module's usage is exemplified in its __main__ part.
For example, `fcnn.py`_ contains a section which uses a Fully Convolutional
Neural Network (FCNN) with batch normalization and residual connections to denoise MNIST images:

.. code-block:: python

    [...] # Code that defines the FCNN.
    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    ims_tr = mnist.train.images.reshape(-1, 28, 28, 1)
    ims_ts = mnist.test.images.reshape(-1, 28, 28, 1)

    # Create a dataset of MNIST with random binary noise as inputs
    # and the original digits as outputs.
    X_tr = ims_tr + np.abs(np.random.randn(*ims_tr.shape) * .1)
    Y_tr = ims_tr
    X_ts = ims_ts + np.abs(np.random.randn(*ims_ts.shape) * .1)
    Y_ts = ims_ts

    # Define the graph.
    fcnn = FCNN(x_shape = ims_tr.shape[1:])

    # Create a Tensorflow session and train the net.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/fcnn')
        writer.add_graph(sess.graph)

        # Fit the net.
        fcnn.fit(X_tr, Y_tr, sess, epochs=100, writer=writer, summary=summary)

        # Predict.
        Y_pred = fcnn.predict(X_ts, sess).reshape(-1, 28, 28)
    
        [...] # More code that plots the results.

This trains an FCNN. The default settings set up a tiny network, with the advantage that it trains in less than a minute on a Titan X GPU, and good enough for testing the architecture. You can use `Tensorboard`_ to visualize the graph. For example, the image below illustrates the graph and zooms onto one specific batch norm -- residual layer:

    .. image:: https://github.com/kjchalup/neural_networks/blob/master/fcnn_graph.png
        :alt: Example FCNN graph.
        :align: center

Our FCNN indeed learned to denoise noisy MNIST pretty well:

    .. image:: https://github.com/kjchalup/neural_networks/blob/master/smoothmnist.png
        :alt: Example NN training output.
        :align: center
        
You can also use `Tensorboard`_ to visualize training and validation validation loss, and all kinds of other training stats:

    .. image:: https://github.com/kjchalup/neural_networks/blob/master/tr_loss.png
        :alt: Validation loss.
        :align: center
        
    .. image:: https://github.com/kjchalup/neural_networks/blob/master/val_loss.png
        :alt: Validation loss.
        :align: center

Implemented Methods
-------------------
At the moment, the reposity contains the following methods:
  
  * `nn.py`_: Multi-layer perceptron (MLP) with Dropout (`arXiv:1207.0580`_).
  * `nn.py`_: Residual Network (`arXiv:1512.03385`_).
  * `nn.py`_: Highway Network (`arXiv:1505.00387`_).
  * `gan.py`_: Least-Squares Generative Adversarial Network (`arXiv:1611.04076v2`_, in my experience the best GAN, though doesn't a convergence criterion like Wasserstein GANs).  
  * `cgan.py`_: Conditional Least-Squares Generative Adversarial Network (`arXiv:1411.1784`_)
  * `mtn.py`_: Multi-Task Networks (my own creation) -- learn from multiple datasets with related inputs but different output tasks.
  * `fcnn.py`_: Fully-convolutional neural nets.

Requirements
------------
Everything should work with Python 2 and 3.

    * `NumPy`_ >= 1.12
    * `TensorFlow`_ >= 1.0.0
   
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
