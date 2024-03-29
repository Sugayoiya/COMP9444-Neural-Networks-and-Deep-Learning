"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

You do not need to import any other libraries for this assignment.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create in part II.
"""

import tensorflow as tf

""" PART I """


def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af


def add_consts_with_placeholder():
    """ 
    Construct a TensorFlow graph that constructs 2 constants, 5.1, 1.0 and one
    TensorFlow placeholder of type tf.float32 that accepts a scalar input,
    and adds these three values together, returning as a tuple, and in the
    following order:
    (the resulting tensor, the constructed placeholder).
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.placeholder(dtype = tf.float32)
    a1 = tf.add(c1,c2)
    af = tf.add(a1,c3)
    return af, c3


def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """

    out_value = tf.maximum(in_value,0)

    return out_value


def my_perceptron(x):
    """
    Implement a single perception that takes four inputs and produces one output,
    using the RelU activation function you defined previously.

    Specifically, implement a function that takes a list of 4 floats x, and
    creates a tf.placeholder the same length as x. Then create a trainable TF
    variable that for the weights w. Ensure this variable is
    set to be initialized as all ones.

    Multiply and sum the weights and inputs following the peceptron outlined in the
    lecture slides. Finally, call your relu activation function.
    hint: look at tf.get_variable() and the initalizer argument.
    return the placeholder and output in that order as a tuple

    Note: The code will be tested using the following init scheme
        # graph def (your code called)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # tests here

    """ 
    i = tf.placeholder(dtype = tf.float32, shape = [x,])
    w = tf.get_variable('w', initializer = tf.ones_initializer(),shape = [x,], trainable = True)
    out = my_relu(tf.reduce_sum(tf.multiply(w,i)))

    return i, out


""" PART II """
fc_count = 0  # count of fully connected layers. Do not remove.


def input_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")


def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")




def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    # X = tf.placeholder(tf.float32, [None, 784])
    # Y = tf.placeholder(tf.float32, [None, 10])

    w = tf.Variable(tf.zeros([X.shape[1],layersize]))
    b = tf.Variable(tf.zeros([layersize]))
    logits = tf.add(tf.matmul(X,w) , b)
    preds = tf.nn.softmax(logits)
    batch_xentropy = -(Y * tf.log(preds + 1e-8) )
    batch_loss = tf.reduce_mean(batch_xentropy)

    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    # X = tf.placeholder(tf.float32, [None, 784])
    # Y = tf.placeholder(tf.float32, [None, 10])

    # Connection weights for the first layer
    w1 = tf.Variable(tf.random_normal([784, 784], stddev=0.03), name='w1')
    # Biases for the first layer
    b1 = tf.Variable(tf.random_normal([784]), name='b1')
    # and Connection weights for the second layer
    w2 = tf.Variable(tf.random_normal([784, 10], stddev=0.03), name='w2')
    # Biases for the second layer
    b2 = tf.Variable(tf.random_normal([10]), name='b2')
    # intermediate
    hidden = tf.add(tf.matmul(X,w1),b1)
    hidden = tf.nn.relu(hidden)

    logits = tf.add(tf.matmul(hidden,w2),b2)
    preds = tf.nn.softmax(logits)

    batch_xentropy = -(Y * tf.log(preds + 1e-8))
    batch_loss = tf.reduce_mean(batch_xentropy)

    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="same"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """
    conv1 = tf.layers.conv2d(X, convlayer_sizes[0], filter_shape, padding = padding, activation = tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, convlayer_sizes[1], filter_shape, padding = padding, activation= tf.nn.relu)
    # print(conv1.shape,'\n',conv2.shape)

    fc_size = int(conv2.shape[1]*conv2.shape[2]*conv2.shape[3])
    # print(fc_size)

    w1 = tf.Variable(tf.truncated_normal( [fc_size,fc_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1,shape = [fc_size]))
    fc = tf.reshape(conv2,[-1,fc_size])
    fc_out = tf.nn.relu(tf.add(tf.matmul(fc,w1),b1))
    # print(fc_out.shape)

    # w = tf.Variable(tf.truncated_normal( [1024,outputsize], stddev=0.1))
    # b = tf.Variable(tf.constant(0.1, shape = [outputsize]))

    # logits = tf.add(tf.add(tf.matmul(w,fc_out),b))
    # preds = tf.nn.softmax(logits)

    # batch_xentropy = -(Y * tf.log(preds + 1e-8))
    # batch_loss = tf.reduce_mean(batch_xentropy)

    w, b, logits, preds, batch_xentropy, batch_loss = onelayer(fc_out,Y)

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
