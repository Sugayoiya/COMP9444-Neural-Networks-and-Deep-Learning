import tensorflow as tf
import re,string

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 256 # Maximum length of a review to consider
EMBEDDING_SIZE = 50 # Dimensions for each word vector

# 128 200 iter:14000 acc:0.84

# stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
#                   'there', 'about', 'once', 'during', 'out', 'very', 'having',
#                   'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
#                   'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
#                   'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
#                   'each', 'the', 'themselves', 'below', 'are', 'we',
#                   'these', 'your', 'his', 'through', 'don', 'me', 'were',
#                   'her', 'more', 'himself', 'this', 'down', 'should', 'our',
#                   'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
#                   'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
#                   'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
#                   'yourselves', 'then', 'that', 'because', 'what', 'over',
#                   'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
#                   'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
#                   'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
#                   'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
#                   'how', 'further', 'was', 'here', 'than'})
stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than','m','ll'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # print('before:',review)
    # print()

    review = review.lower().replace("<br />", " ")
    # print('after replace lower and <br />:',review)
    # print()
    
    # strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    # review = re.sub(strip_special_chars,"",review.lower())
    # print('remain only words:',review)
    # print()

    # review = review.translate(None, string.punctuation)
    # print('remain only words:',review)
    # print()
    s = review
    # punctuations = set(string.punctuation)
    remove_punc = re.compile('[%s]' % re.escape(string.punctuation))
    review = remove_punc.sub(' ',s)
    # print('remain only words:',review)
    # print()

    remove_stop_words = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape,list(stop_words))))
    review = remove_stop_words.sub("",review)
    # print('after stop_words:',review)
    # print()

    review = re.sub(' +',' ',review)
    # print('finally:',review)

    processed_review = review.split(' ')

    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    labels = tf.placeholder(tf.float32, [ BATCH_SIZE, 2],name="labels")
    input_data = tf.placeholder(tf.float32,[BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name= "input_data")

    dropout_keep_prob = tf.placeholder_with_default(.75,shape=(),name="dropout_keep_prob")

    # embedding = tf.convert_to_tensor(glove_embeddings_arr, dtype=tf.float32)
    # embeds = tf.Variable(tf.zeros([batch_size, len(glove_embeddings_arr[0])], dtype=tf.float32))
    # embeds = tf.nn.embedding_lookup(embedding, input_data)

    lstm1 = tf.contrib.rnn.LSTMCell(64)
    lstm2 = tf.contrib.rnn.LSTMCell(MAX_WORDS_IN_REVIEW)
    lstm_list = [lstm1,lstm2]

    # lstm = tf.contrib.rnn.BasicLSTMCell(MAX_WORDS_IN_REVIEW)
    # drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob = dropout_keep_prob)
    # cell = tf.contrib.rnn.MultiRNNCell([drop])

    # cell = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob = dropout_keep_prob)
    
    multi_rnn = tf.contrib.rnn.MultiRNNCell(lstm_list)
    multi_rnn = tf.contrib.rnn.DropoutWrapper(cell=multi_rnn, output_keep_prob = dropout_keep_prob)

    output, _ = tf.nn.dynamic_rnn(multi_rnn,input_data,dtype=tf.float32)
    output = tf.transpose(output, [1,0,2])
    # output = tf.reduce_mean(output, axis=1)

    last = tf.gather(output, int(output.get_shape()[0])-1)

    weight = tf.Variable(tf.truncated_normal(shape=[MAX_WORDS_IN_REVIEW,2],stddev=0.01))
    bias = tf.Variable(tf.constant(0.1,shape=[2]))
    logits = tf.matmul(last,weight)+bias
    prediction = tf.nn.softmax(logits)

    predict_labels = tf.argmax(logits,1)
    real_labels = tf.argmax(labels,1)
    correct_predict = tf.equal(predict_labels,real_labels)
    Accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32),name = 'accuracy')

    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels=labels)
    loss = tf.reduce_mean(xentropy,name = 'loss')

    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)



    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
