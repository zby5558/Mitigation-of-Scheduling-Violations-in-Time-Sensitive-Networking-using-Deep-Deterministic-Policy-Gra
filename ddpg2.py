# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:15:46 2021

@author: zby09
"""
#this network works for considering misbehaviors and the legnth of windows
from collections import deque
import gym
import numpy as np
import os
import pickle
import random
import tensorflow as tf
from tensorflow.keras import Input
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
import simulation1
tf.compat.v1.disable_eager_execution()

class DDPGTrainer():
    def __init__(#parameters for DDPG model
        self, 
        n_features, 
        n_actions, 
        sample_size=128, 
        tau=0.99, 
        gamma=0.9, 
        epsilon=1.0, 
        epsilon_decay=0.995, 
        epsilon_min=0.01, 
        a_lr=0.002, 
        c_lr=0.002
    ):
        self.tau = tau
        self.memory_buffer = deque(maxlen=4000)
        self.sample_size = sample_size
        self.gamma = gamma 
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.n_features = n_features
        self.n_actions = n_actions
        
        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
    def build_model(self):
        s_input = Input([self.n_features])
        a_input = Input([self.n_actions])
        
        # layers for the actor network
        x = Dense(units=150, activation='relu')(s_input) 
        x = Dense(units=200, activation='relu')(x)
        x = Dense(units=300, activation='tanh')(x)  
        x = Dense(units=200, activation='tanh')(x)
        x = Dense(units=100, activation='linear')(x)
        x = Dense(units=200, activation='tanh')(x)
        action = Dense(units=self.n_actions, activation='softmax')(x)
        #action = Lambda(lambda x: x * self.n_actions)(x)
        actor = tf.keras.models.Model(inputs=s_input, outputs=action)
        
        # layers for the critic network
        x = K.concatenate([s_input, a_input], axis=-1)
        x = Dense(100, activation='relu')(x)
        x = Dense(200, activation='relu')(x)
        x = Dense(300, activation='tanh')(x)
        x = Dense(200, activation='tanh')(x)
        x = Dense(100, activation='linear')(x)
        q_a_value = Dense(1, activation='linear')(x)
        critic = tf.keras.models.Model(inputs=[s_input, a_input], outputs=q_a_value)
        
        actor.add_loss(-critic([s_input, action])) # the loss of the critic network
        ### train the critic network
        critic.trainable = False
        actor.compile(optimizer=tf.keras.optimizers.Adam(self.a_lr))
        critic.trainable = True
        
        ### train the actor network
        actor.trainable = False
        critic.trainable = True # 由于actor的计算图用到critic部分，actor.trainable变化会影响critic.trainable
        critic.compile(optimizer=tf.keras.optimizers.Adam(self.c_lr), loss='mse')
        actor.trainable = True
        return actor, critic
    
    def OU(self, x, mu=0, theta=0.15, sigma=0.2):
        return theta * (mu - x) + sigma * np.random.randn(1) # shape: [1]

    def choose_action(self, state):
        action = self.actor.predict(state)[0] # shape: []
        noise = max(self.epsilon, 0) * self.OU(action)
        action = np.clip(action + noise, 0.01, 1) # shape: [1]
        return action

    def store(self, state, action, reward, next_state, done):#store the information of each iteration
        sample = (state, action, reward, next_state, done)
        self.memory_buffer.append(sample)

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_model(self):
        print("replay")
        samples = random.sample(self.memory_buffer, self.sample_size)
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])

        next_actions = self.target_actor.predict(next_states)
        q_a_next = self.target_critic.predict([next_states, next_actions]) # q_a_next.shape: [self.sample_size, 1]
        y = rewards + self.gamma * q_a_next[:, 0] * ~dones  # y.shape: [self.sample_size]
        
        self.critic.fit([states, actions], y[:, None], verbose=0) 
        self.actor.fit(states, verbose=0)
        
    def update_target_model(self):
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_target_weights[i] * self.tau + (1 - self.tau) * actor_weights[i]
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_target_weights[i] * self.tau + (1 - self.tau) * critic_weights[i]
        self.target_actor.set_weights(actor_target_weights)
        self.target_critic.set_weights(critic_target_weights)
        
    def save(self, checkpoint_path='pendulum'):
        self.ckpt_manager.save()
        with open(f'{checkpoint_path}/epsilon.pkl', 'wb') as f:
            pickle.dump(self.epsilon, f)
        
    def load(self, checkpoint_path='pendulum'):
        ckpt = tf.train.Checkpoint(
            actor=self.actor,
            critic=self.critic,
            target_actor=self.target_actor,
            target_critic=self.target_critic,
            actor_optimizer = self.actor.optimizer,
            critic_optimizer = self.critic.optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
        
        if os.path.exists(f'{checkpoint_path}/epsilon.pkl'):
            with open(f'{checkpoint_path}/epsilon.pkl', 'rb') as f:
                self.epsilon = pickle.load(f)
                
        if self.ckpt_manager.latest_checkpoint:
            status = ckpt.restore(self.ckpt_manager.latest_checkpoint)
            status.run_restore_ops() 
            
session = tf.compat.v1.InteractiveSession()
model = DDPGTrainer(56, 3)
model.load("withLength1") # load the model
flowID = [1,2,3,4,5]
flowOffsets = [10,20,30,40,50]
hyperperiod = 100
GCL = [30,50,20]
bitVector = [0,1,0]
flowToWindow = [[1],[1],[1],[1],[1]]
transTime = [10,10,10,10,10]
initial_state = []
total_rewards = []
done = False
for i in range(56):
    initial_state.append(0)

for episode in range(4000):
    TSN = simulation1.TSN_env(hyperperiod,GCL,flowOffsets,flowID,flowToWindow,bitVector,10,transTime)#reset()
    next_state = initial_state.copy()
    reward_sum = 0
    for step in range(10):
        state = next_state
        action = model.choose_action(np.array(state)[None])
        #print("action: "+str(action))
        actionList = []
        t = TSN.hyperperiod
        for i in range(len(action)-1):
            actionList.append(int(action[i]/np.sum(action)*TSN.hyperperiod))
            t = t-int(action[i]/np.sum(action)*TSN.hyperperiod)
        actionList.append(t)
        print("actionList: "+str(actionList))
        TSN.setGCL(actionList)#update GCL for the next hyperperiod
        reward, state = TSN.running()# running for next hyperperiod
        reward_sum += reward
        model.store(np.array(state), np.array(action), reward, np.array(next_state), done)
        #print("state: "+str(state))
        #print("next_state: "+str(next_state))

        if len(model.memory_buffer) > model.sample_size:
            model.update_model()
            model.update_target_model()
            model.update_epsilon()
    print(f'episode{episode} total reward: {reward_sum}')
    print("episode: "+str(episode))
    total_rewards.append(reward_sum)
model.save("withLength1")

session.close()