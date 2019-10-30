
#### how to start

ready to begin, download the necessary files in Assignment2.zip (http://www.cse.unsw.edu.au/~cs9444/18s2/hw2/Assignment2.zip)

Unzip this archive by typing

`unzip Assignment2.zip`

You should then see the following files:

>data/  	  
>>Directory containing the training and evaluation datasets.

>implementation.py  	 
>>This is a skeleton file for your code. The assignment should be completed by modifying this file.

>runner.py	  
>>This file is a wrapper around implementation.py and contains a large amount of pre-implemented functionality. An unedited version of this file must be used to generate your final model.

>glove.6B.50d.txt	  
>>This is a text file containing the embedding vectors used in this assignment. Word embeddings have been shown to improve the performance of many NLP models by converting words from character arrays to vectors that contain semantic information of the word itself. In this assignment, we use GloVe embeddings, you can read more about them here.

#### dataset

The training dataset contains a series of movie reviews scraped from the IMDB website. There are no more than 30 reviews for any one specific movie. The "data" directory contains two sub-directories, "train" and "validate". Each of these contains two sub-directories, **"pos"** and **"neg"**. These directories contain the raw reviews in plain text form. The "train" directory contains 12500 positive and 12500 negative reviews; the "validate" directory contains 1000 positive and 1000 negative reviews. Each review is confined to the first line of its associated text file, with no line breaks.

For evaluation, we will run your model against a third dataset "test" that has not been made available to you. If contains additional reviews in the same format. For this reason you should be very careful to avoid overfitting - your model could report 100% training accuracy but completely fail on unseen reviews. There are various ways to prevent this such as judicious use of dropout, splitting the data into a training and validation set, etc.

#### Code Overview

>runner.py

This file allows for three modes of operation, "train", "eval" and "test". The first two can be used during development, while "test" will be used by us for marking.

**"train"** calls functions to load the data, and convert it to embedded form. It then trains the model defined in implementation.py, performs tensorboard logging, and saves the model to disk every 10000 iterations. These model files are saved in a created checkpoints directory, and should consist of a checkpoint file, plus three files ending in the extentions .data-00000-of-00001, .index and .meta It also prints loss values to stdout every 50 iterations. While an unedited version of this file must be used to train your final model, during development you are encouraged to make modifications. You may wish to display validation accuracy on your tensorboard plots, for example.

**"eval"** evaluates the latest model checkpoint present in the local checkpoints directory and prints the final accuracy to the console. You should not modify this code.

You may note that in both train and eval mode, the data is first fed through the **load_data()** method, which in turn calls the **preprocess(review)** function that you will define. This is to ensure your preprocessing is consistent across all runs. In otherwords, whatever transformations you apply to the data during training will also be applied during evaluation and testing.

Further explanation of the functionality of this file appears in comments.

>implemention.py

This is where you should implement your solution. This file contains two functions: **preprocess()**, and **define_graph()**

**preprocess(review)** is called whenever a review is loaded from text, prior to being converted into embedded form. You can do anything here that is manipulation at a string level, e.g.

* removing stop words
* stripping/adding punctuation
* changing case
* word find/replace
* paraphrasing

Note that this shouldn't be too complex - it's main purpose is to clean data for your actual model, not to be a model in and of itself.
**define_graph()** is where you should define your model. You will need to define **placeholders**, for the input and labels, Note that the input is not strings of words, but the strings after the embedding lookup has been applied (i.e. arrays of floats). To ensure your model is sufficiently general (so as to achieve the best test accuracy) you should experiment with regularization techniques such as dropout. This is where you must also provide the correct names for your placeholders and variables.

There are two variables which you should experiment with changing. **BATCH_SIZE** defines the size of the batches that will be used to train the model in runner.py and **MAX_WORDS_IN_REVIEW** determines the maximum number for words that are considered in each sample. Both may have a significant effect on model performance.


#### Visualizing Progress

In addition to the output of  runner.py, you can view the progress of your models using the tensorboard logging included in that file. To view these logs, run the following command from the source directory:
`tensorboard --logdir=./tensorboard`

* open a Web browser and navigate to  http://localhost:6006
* you should be able to see a plot of the loss and accuracies in TensorBoard under the "scalars" tab

Make sure you are in the same directory from which runner.py is running.
