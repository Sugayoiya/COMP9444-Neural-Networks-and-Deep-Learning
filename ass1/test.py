import tensorflow as tf 

def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """

    out_value = tf.maximum(in_value,0)

    return out_value

if __name__ == "__main__":

    i = tf.placeholder(dtype = tf.float32, shape = [4,])
    w = tf.get_variable('w', initializer = tf.ones_initializer(),shape = [4,], trainable = True)
    
    init = tf.global_variables_initializer()
    ini = my_relu(tf.multiply(w,i))
    ii = tf.Session()
    print(ii.run(init))
    print(ii.run(ini,feed_dict={i:[1,2,3,4]}))