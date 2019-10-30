#### Preliminaries

We will be using the AI-Gym environment provided by OpenAI to test our algorithms. AI Gym is a toolkit that exposes a series of high-level function calls to common environment simulations used to benchmark RL algorithms. All AI Gym function calls required for this assignment have been implemented in the skeleton code provided, however it would be a good idea to understand the basic functionality by reading through the getting started guide.
You can install AI Gym by running

`pip install gym`

or, if you need admin priveliges to install python packages:

`sudo -H pip install gym`

This will install all environments in the "toy_text", "algorithmic", and "classic_control" categories. We will only be using the CartPole environment in "classic_control". If you want to only install the classic_control environments (to save disk space or to keep your installed packages at a minimum) you can run:

`pip install 'gym[classic_control]'`
(prepend sudo -H if needed)

To test if the required environments have been installed correctly, run (in a python interpreter):

```import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()
```
You should then be able to see a still-frame render of the CartPole environment.

Next, download the starter code from src.zip(http://www.cse.unsw.edu.au/~cs9444/18s2/hw3/src.zip)

All functionality should be implemented in this file:

Neural_QTrain.py

#### Code Structure

##### Placeholders:
* state_in takes the current state of the environment, which is represented in our case as a sequence of reals.
* action_in accepts a one-hot action input. It should be used to "mask" the q-values output tensor and return a q-value for that action.
* target_in is the Q-value we want to move the network towards producing. Note that this target value is not fixed - this is one of the components that seperates RL from other forms of machine learning.

##### Network Graph:
* You can define any type of graph you like here, cnn, dense, lstm etc. It's important to consider what is the constraint in this problem - is a larger network necessarily better?
* q_values: Tensor containing Q-values for all available actions i.e. if the action space is 8 this will be a rank-1 tensor of length 8
* q_action: This should be a rank-1 tensor containing 1 element. This value should be the q-value for the action set in the action_in placeholder
* Loss/Optimizer Definition You can define any loss function you feel is appropriate. Hint: should be a function of target_in and q_action. You should also make careful choice of the optimizer to use.

##### Main Loop:
* Move through the environment collecting experience. In the naive implementation, we take a step and collect the reward. We then re-calcuate the Q-value for the previous state and run an update based on this new Q-value. This is the "target" referred to throughout the code.

#### Implementation Steps

* Step 1: Basic Implementation

The first thing you should do is complete the specified TODO sections and implement a basic version of neural-q-learning that is able to perform some learning. Following the provided structure, we perform one update immediately after each step taken in the environment. This results in slow and brittle learning. A correct implementation will first achieve 200 reward in around 1000 episodes depending on hyperparameter choices. In general you will observe sporadic and unstable learning.

Details of the CartPole environment are availible here.

* Step 2: Batching

An easy way to speed up training is to collect batches of experiences then compute the update step on this batch. This has the effect of ensuring the Q-values are updated over a larger number of steps in the environment, however these steps will remain highly correlated. Depending on you network and hyperparameters, you should see a small to medium improvement in accuracy.

* Step 3: Experience Replay

To ensure batches are decorrelated, save the experiences gathered by stepping through the environment into an array, then sample from this array at random to create the batch. This should significantly improve the robustness and stability of learning. At this point you should be able to achieve 200 average reward within the first 200 episodes.

* Step 4: Extras

Once these steps have been implemented, you are free to include any extra features you feel will improve the algorithm. Use of a target network, hyperparameter tuning, and altering the Bellman update may all be benificial.

* Step 5: Report

As with Assignment 2, you are required to submit a simple 1-page pdf or text file outlining and justfying your design choices. This report will be used to assign marks in accordance with the marking scheme if we are unable to run your code.


#### result

`python3 Neural_QTrain.py`

```
Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5061 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
episode: 100 epsilon: 0.5464400100343645 Evaluation Average Reward: 198.3
episode: 200 epsilon: 0.3349594990296074 Evaluation Average Reward: 200.0
episode: 300 epsilon: 0.2068508575689736 Evaluation Average Reward: 200.0
episode: 400 epsilon: 0.12924642991313134 Evaluation Average Reward: 200.0
episode: 500 epsilon: 0.08223596189843865 Evaluation Average Reward: 200.0
episode: 600 epsilon: 0.05375841016954478 Evaluation Average Reward: 200.0
episode: 700 epsilon: 0.03650755122854545 Evaluation Average Reward: 200.0
episode: 800 epsilon: 0.026057490878016322 Evaluation Average Reward: 200.0
episode: 900 epsilon: 0.01972715325812182 Evaluation Average Reward: 200.0
episode: 1000 epsilon: 0.01589242187498466 Evaluation Average Reward: 200.0
```

