
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import random

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# In[2]:


os.chdir(r"D:\Shiny\Machine Learning\Assignment 4")


# In[3]:


sm_value = pd.read_csv("SM_value.csv")
sm_policy = pd.read_csv("SM_policy.csv")


# In[4]:


lm_value = pd.read_csv("LM_value.csv")
lm_policy = pd.read_csv("LM_policy.csv")


# In[6]:


lm_value.head()


# In[56]:


smv_step = sm_value.iloc[:,1]
smv_reward = sm_value.iloc[:,2]
smv_time = sm_value.iloc[:,3]
smp_step = sm_policy.iloc[:,1]
smp_reward = sm_policy.iloc[:,2]
smp_time = sm_policy.iloc[:,3]


# In[7]:


lmv_step = lm_value.iloc[:,1]
lmv_reward = lm_value.iloc[:,2]
lmv_time = lm_value.iloc[:,3]
lmp_step = lm_policy.iloc[:,1]
lmp_reward = lm_policy.iloc[:,2]
lmp_time = lm_policy.iloc[:,3]


# In[8]:


n = np.size(lmv_step)


# In[9]:


np.size(lmp_step)


# In[10]:


def compare_time(n,x,y,title):
    
    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Iteration")
    plt.ylabel("Time (in milliseconds)")
    plt.plot(n, x, '-', color="b", label="Value Iteration")
    plt.plot(n, y, '-', color="r", label="Policy Iteration")
    plt.legend(loc="best")
    plt.show()
    
def compare_reward(n,x,y, title):
    
    plt.figure()
    plt.title("Model Reward: " + title)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.plot(n, x, '-', color="b", label="Value Iteration")
    plt.plot(n, y, '-', color="r", label="Policy Iteration")
    plt.legend(loc="best")
    plt.show()


def compare_step(n,x,y, title):
    
    plt.figure()
    plt.title("Model Step: " + title)
    plt.xlabel("Iteration")
    plt.ylabel("Step")
    plt.plot(n, x, '-', color="b", label="Value Iteration")
    plt.plot(n, y, '-', color="r", label="Policy Iteration")
    plt.legend(loc="best")
    plt.show()


# In[14]:


k = (np.linspace(.01, 1.0, 200)*n).astype('int')  


# In[15]:


k


# In[18]:


compare_time(k,lmv_time, lmp_time,'Large Maze')


# In[17]:


compare_reward(k, lmv_reward, lmp_reward, 'Large Maze')   
compare_step(k, lmv_step, lmp_step, 'Large Maze')


# In[65]:


smq_1 = pd.read_csv("SMQ_0.1.csv")
smq_2 = pd.read_csv("SMQ_0.3.csv")
smq_3 = pd.read_csv("SMQ_0.5.csv")
smq_4 = pd.read_csv("SMQ_0.7.csv")
smq_5 = pd.read_csv("SMQ_0.9.csv")


# In[19]:


lm_1 = pd.read_csv("LM_0.1.csv")
lm_2 = pd.read_csv("LM_0.3.csv")
lm_3 = pd.read_csv("LM_0.5.csv")
lm_4 = pd.read_csv("LM_0.7.csv")
lm_5 = pd.read_csv("LM_0.9.csv")


# In[20]:


lm_1.head()


# In[21]:


lm1_step = lm_1.iloc[:,1]
lm1_reward = lm_1.iloc[:,2]
lm1_time = lm_1.iloc[:,3]
lm2_step = lm_2.iloc[:,1]
lm2_reward = lm_2.iloc[:,2]
lm2_time = lm_2.iloc[:,3]
lm3_step = lm_3.iloc[:,1]
lm3_reward = lm_3.iloc[:,2]
lm3_time = lm_3.iloc[:,3]
lm4_step = lm_4.iloc[:,1]
lm4_reward = lm_4.iloc[:,2]
lm4_time = lm_4.iloc[:,3]
lm5_step = lm_5.iloc[:,1]
lm5_reward = lm_5.iloc[:,2]
lm5_time = lm_5.iloc[:,3]


# In[22]:


def compare_time(n,x,y,z,m,l,title):
    
    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Iteration")
    plt.ylabel("Time (in milliseconds)")
    plt.plot(n, x, '-', color="k", label="Learning rate = 0.1")
    plt.plot(n, y, '-', color="b", label="Learning rate = 0.3")
    plt.plot(n, z, '-', color="r", label="Learning rate = 0.5")
    plt.plot(n, m, '-', color="g", label="Learning rate = 0.7")
    plt.plot(n, l, '-', color="m", label="Learning rate = 0.9")
    plt.legend(loc="best")
    plt.show()
    
def compare_reward(n,x,y,z,m,l, title):
    
    plt.figure()
    plt.title("Model Reward: " + title)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.plot(n, x, '-', color="k", label="Learning rate = 0.1")
    plt.plot(n, y, '-', color="b", label="Learning rate = 0.3")
    plt.plot(n, z, '-', color="r", label="Learning rate = 0.5")
    plt.plot(n, m, '-', color="g", label="Learning rate = 0.7")
    plt.plot(n, l, '-', color="m", label="Learning rate = 0.9")
    plt.legend(loc="best")
    plt.show()


def compare_step(n,x,y,z,m,l, title):
    
    plt.figure()
    plt.title("Model Step: " + title)
    plt.xlabel("Iteration")
    plt.ylabel("Step")
    plt.plot(n, x, '-', color="k", label="Learning rate = 0.1")
    plt.plot(n, y, '-', color="b", label="Learning rate = 0.3")
    plt.plot(n, z, '-', color="r", label="Learning rate = 0.5")
    plt.plot(n, m, '-', color="g", label="Learning rate = 0.7")
    plt.plot(n, l, '-', color="m", label="Learning rate = 0.9")
    plt.legend(loc="best")
    plt.show()


# In[23]:


n= np.size(lm1_step )


# In[70]:


n= np.size(smq1_step )


# In[24]:


n


# In[30]:


k = (np.linspace(.001, 1.0, 1000)*n).astype('int')  


# In[31]:


k


# In[32]:


compare_time(k,lm1_time, lm2_time,lm3_time,lm4_time,lm5_time,'Large Maze')
compare_reward(k, lm1_reward, lm2_reward,lm3_reward,lm4_reward,lm5_reward, 'Large Maze')   
compare_step(k, lm1_step, lm2_step,lm3_step,lm4_step,lm5_step, 'Large Maze')


# In[1]:


from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/q2ZOEFAaaI0?showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')


# In[2]:


import numpy as np
import gym
import random


# In[3]:


from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
)


# In[4]:


env = gym.make("FrozenLakeNotSlippery-v0")


# In[5]:


#env = gym.make("FrozenLake-v0")


# In[6]:


action_size = env.action_space.n
state_size = env.observation_space.n


# In[7]:


qtable = np.zeros((state_size, action_size))


# In[34]:


total_episodes = 200        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 0.1          # Exploration rate
max_epsilon = 0.1            # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001             # Exponential decay rate for exploration prob
#I find that decay_rate=0.001 works much better than 0.01


# In[35]:


rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards =total_rewards + reward
        
        # Our new state is state
        state = new_state
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
        
    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)
print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)
print(epsilon)


# In[22]:


env.reset()
env.render()
print(np.argmax(qtable,axis=1).reshape(4,4))

