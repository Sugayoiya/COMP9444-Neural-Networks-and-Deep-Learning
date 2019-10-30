## COMP9444-Neural-Networks-and-Deep-Learning

cuda9.0 on ubuntu 18.04 install reference: 

https://blossomnoodles.github.io/2018/04/30/ubuntu-18.04-cuda-installation.html

change 
```
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}
```
to
```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\ 
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

then

`source ~/tensorflow/bin/activate`


### assignment 1：Basic TensorFlow and Digit Recognition

#### Introduction
This assignment is split into two parts. Part 1 contains **basic introductory exercises using TensorFlow's core API**. In Part 2, you will be **implementing a single layer network, a two layer network and a convolutional network** to classify **handwritten digits**. We will work with the MNIST dataset, a common dataset used to evaluate Machine Learning models.


### assignment 2：Recurrent Networks and Sentiment Classification

#### Introduction
You should now have a good understanding of the internal dynamics of TensorFlow and how to implement, train and test various network architectures. In this assignment we will **develop a classifier able to detect the sentiment of movie reviews**. Sentiment classification is an active area of research. Aside from improving performance of systems like Siri and Cortana, sentiment analysis is very actively utilized in the finance industry, where sentiment is required for automated trading on news snippits and press releases.

### assignment 3：Deep Reinforcement Learning

#### Introduction
In this assignment we will implement a **Deep Reinforcement Learning algorithm on a classic control task in the OpenAI AI-Gym Environment**. Specifically, we will implement Q-Learning using a Neural Network as an approximator for the Q-function.


