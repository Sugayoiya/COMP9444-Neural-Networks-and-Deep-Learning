import gym
import tensorflow as tf
import numpy as np
import random
# import datetime

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99 # discount factor
INITIAL_EPSILON = 0.9 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 200 # decay period
batch_size = 200    
replay_size = 10000
hidden_size = 100

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
w1 = tf.Variable(tf.truncated_normal([STATE_DIM, hidden_size]))
b1 = tf.Variable(tf.constant(0.01,shape = [ hidden_size]))
w2 = tf.Variable(tf.truncated_normal([ hidden_size,ACTION_DIM]))
b2 = tf.Variable(tf.constant(0.01,shape = [ACTION_DIM]))

hidden_layer = tf.nn.relu(tf.matmul(state_in,w1)+b1)

# TODO: Network outputs
q_values = tf.matmul(hidden_layer,w2)+b2
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

replay_buffer = []
reward_list = []
train_flag = 1

# starttime = datetime.datetime.now()
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()
    reward_ = 0
    # Update epsilon once per episode
    epsilon -= (epsilon-FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        reward_ += reward
        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        
        replay_buffer.append((state,action,reward,next_state,done))
        if len(replay_buffer) > replay_size:
            replay_buffer.pop(0)
        
        # target = reward + GAMMA*np.max(nextstate_q_values)

        # Update
        state = next_state
        if train_flag == 1:
            if(len(replay_buffer)>batch_size):
                minibatch = random.sample(replay_buffer,batch_size)
                state_batch = [data[0] for data in minibatch]
                action_batch = [data[1] for data in minibatch]
                reward_batch = [data[2] for data in minibatch]
                next_state_batch = [data[3] for data in minibatch]

                target_batch = []
                q_value_batch = q_values.eval(feed_dict={state_in:next_state_batch})

                for i in range(0,batch_size):
                    sample_done = minibatch[i][4]
                    if sample_done:
                        target_batch.append(reward_batch[i])
                    else:
                        target_ = reward_batch[i] + GAMMA*np.max(q_value_batch[i])
                        target_batch.append(target_)
                
                # Do one training step
                session.run([optimizer], feed_dict={
                    target_in: target_batch,
                    action_in: action_batch,
                    state_in: state_batch
                })
                    
        if done:
            break
    reward_list.append(reward_)
    # print('episode:',episode,reward_)

    if len(reward_list)<10:
        train_flag = 1
    elif sum(reward_list[-10:])/10>=195.0:
        train_flag = 0


    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)
        # endtime = datetime.datetime.now()
        # print('time:',endtime-starttime)
        # starttime = endtime

env.close()
