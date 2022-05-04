import time
import numpy as np
import math
from util import *
from map import *
import sys
import os
import time
import math
import copy
import argparse
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import scipy.signal
import os
from datetime import datetime
import matplotlib.pyplot as plt

class CPPEnv(object):
    def __init__(self,util):
        self.util = util
        self.vpsState = [0]*len(self.util.viewpoints)
        self.voxelState = [0]*len(self.util.map.grids)
        self.occupiedCount = len(np.nonzero(self.voxelState)[0])
        self.vpIdx = 0

    def reset(self,vpIdx=0):
        self.vpIdx = vpIdx
        self.vpsState = [0]*len(self.util.viewpoints)
        self.voxelState = [0]*len(self.util.map.grids)
        self.occupiedCount = len(np.nonzero(self.voxelState)[0])
        vp = self.util.viewpoints[vpIdx]
        self.vpsState[vpIdx] += 1
        for grid in vp.gridCover:
            self.voxelState[grid.id] = 1

        coverage = self.coverage()
        nbvps = self.util.neighbors(vpIdx)
        obs = [vp.location[0], vp.location[1], vp.location[2], self.vpsState[vpIdx]]
        for i in nbvps:
            v = self.util.viewpoints[i]
            obs.append(v.location[0])
            obs.append(v.location[1])
            obs.append(v.location[2])
            obs.append(self.vpsState[i])

        return vp, nbvps, obs, coverage

    def step(self, vpIdx):
        # move to new vp and update the status
        dist = vpDistance(self.util.viewpoints[self.vpIdx], self.util.viewpoints[vpIdx])
        self.vpIdx = vpIdx

        self.vpsState[vpIdx] += 1
        vp = self.util.viewpoints[vpIdx]
        for grid in vp.gridCover:
            self.voxelState[grid.id] = 1

        # calcuate reward
        occupiedCount_new = len(np.nonzero(self.voxelState)[0])
        reward = 100.0*(occupiedCount_new - self.occupiedCount)/len(self.util.map.grids) - 0.1 - 10*(self.vpsState[self.vpIdx]-1)
        self.occupiedCount = occupiedCount_new

        # return
        coverage = self.coverage()
        nbvps = self.util.neighbors(vpIdx)
        obs = [vp.location[0],vp.location[1],vp.location[2],self.vpsState[vpIdx]]
        for i in nbvps:
            v = self.util.viewpoints[i]
            obs.append(v.location[0])
            obs.append(v.location[1])
            obs.append(v.location[2])
            obs.append(self.vpsState[i])

        # print(vpIdx, nbvps, reward, obs, coverage)
        return vp, nbvps, reward, obs, coverage

    def coverage(self):
        return float(self.occupiedCount) / float(len(self.util.map.grids))

"""
Replay Buffer, strore experiences and calculate total rewards, advanteges
the buffer will be used for update the policy
"""
class ReplayBuffer:
    def __init__(self, input_shape, action_size, size=1000):
        self.obs_buf = np.zeros((size, input_shape), dtype=np.float32) # states
        self.act_buf = np.zeros((size, action_size), dtype=np.float32) # action, based on stochasitc policy with teh probability
        self.rew_buf = np.zeros(size, dtype=np.float32) # step reward
        self.pred_buf = np.zeros((size, action_size), dtype=np.float32) # prediction: action probability, output of actor net
        self.val_buf = np.zeros(size, dtype=np.float32) # value of (s,a), output of critic net
        self.adv_buf = np.zeros(size, dtype=np.float32) # advantege Q(s,a)-V(s)
        self.ret_buf = np.zeros(size, dtype=np.float32) # ep_return, total reward of episode
        self.ptr, self.idx = 0, 0 # buffer ptr, and current trajectory start index

    def store(self, state, action, reward, prediction, value):
        #print("storing", state[0].shape, action.shape, reward, prediction.shape, value.shape)
        self.obs_buf[self.ptr]=state
        self.act_buf[self.ptr]=action
        self.rew_buf[self.ptr]=reward
        self.pred_buf[self.ptr]=prediction
        self.val_buf[self.ptr]=value
        self.ptr += 1

    def size(self):
        return self.ptr

    """
    For each epidode, calculating the total reward and advanteges with specific
    gamma and lamada
    """
    def ep_update(self, gamma=0.99, lamda=0.95):
        """
        magic from rllab for computing discounted cumulative sums of vectors
        input: vector x: [x0, x1, x2]
        output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
        """
        def discount_cumsum(x,discount):
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        ep_slice = slice(self.idx,self.ptr)
        rews = np.append(self.rew_buf[ep_slice],0)
        vals = np.append(self.val_buf[ep_slice],0)
        # rewards-to-go, which is targets for the value function
        self.ret_buf[ep_slice] = discount_cumsum(rews,gamma)[:-1]
        # General Advantege Estimation
        deltas = rews[:-1]+gamma*vals[1:]-vals[:-1]
        self.adv_buf[ep_slice] = discount_cumsum(deltas,gamma*lamda)
        self.idx = self.ptr

    def get(self):
        s = slice(0,self.ptr)
        # normalize advantage batch-wise
        advs = self.adv_buf[s]
        normalized_advs = (advs-np.mean(advs))/(np.std(advs)+1e-10)
        data = dict(states=self.obs_buf[s], actions=self.act_buf[s],
                    returns=self.ret_buf[s], predictions=self.pred_buf[s],
                    advantages=normalized_advs)
        self.ptr, self.idx = 0, 0
        return data


"""
Agent NN
"""
def mlp_net(inputs_dim, outputs_dim, outputs_activation='softmax'):
    inputs = keras.Input(shape=inputs_dim)
    x = layers.Dense(128, activation = 'relu')(inputs)
    x = layers.Dense(64, activation = 'relu')(x)
    x = layers.Dense(32, activation = 'relu')(x)
    outputs = layers.Dense(outputs_dim, activation = outputs_activation)(x)
    return keras.Model(inputs=inputs, outputs=outputs)

"""
loss print call back
"""
class PrintLoss(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        print("epoch index", epoch+1, "loss", logs.get('loss'))

"""
Actor net
"""
class Actor_Model:
    def __init__(self, input_shape, action_size, clip_ratio, lr, beta):
        self.clip_ratio = clip_ratio
        self.beta = beta # hyperparameter that controls the influence of entropy loss
        self.action_size = action_size
        self.actor = self.build_model(input_shape, action_size, lr)
        self.loss_printer = PrintLoss()

    def build_model(self, input_shape, action_size, lr):
        model = mlp_net(inputs_dim=input_shape, outputs_dim=action_size, outputs_activation="softmax")
        model.compile(loss=self.ppo_loss, optimizer=keras.optimizers.Adam(learning_rate=lr))
        print(model.summary())
        return model

    """
    The key part of the PPO-clip
    policy ratio is define as r = pi(a|s) / pi_old(a|s)
    loss = min(r*AF, clip(r, 1-e, 1+e)*AF), where 'e' is the clip ratio,
    and AF is the advantage function AF(s,a)=Q(s,a)-V(s)
    """
    def ppo_loss(self, y_true, y_pred):
        # y_true: np.hstack([advantages, predictions, actions])
        advs,o_pred,acts = y_true[:,:1],y_true[:,1:1+self.action_size],y_true[:,1+self.action_size:]
        # print(y_pred, advs, picks, acts)
        prob = y_pred*acts
        old_prob = o_pred*acts
        ratio = prob/(old_prob + 1e-10)
        p1 = ratio*advs
        p2 = K.clip(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advs
        # total loss = policy loss + entropy loss (entropy loss for promote action diversity)
        loss = -K.mean(K.minimum(p1,p2)+self.beta*(-y_pred*K.log(y_pred+1e-10)))
        return loss

    def predict(self, state):
        digits = self.actor.predict(state)
        #print("actor prediction", digits)
        return digits

    def fit(self,states,y_true,epochs,batch_size):
        self.actor.fit(states, y_true, epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size,callbacks=[self.loss_printer])

    def save(self, path):
        self.actor.save_weights(path)

    def load(self, path):
        self.actor.load_weights(path)

"""
Critic net
"""
class Critic_Model:
    def __init__(self, input_shape, lr):
        self.critic = self.build_model(input_shape, lr)
        self.loss_printer = PrintLoss()

    def build_model(self, input_shape, lr):
        model = mlp_net(inputs_dim=input_shape, outputs_dim=1,outputs_activation="linear")
        model.compile(loss="mse",optimizer=keras.optimizers.Adam(learning_rate=lr))
        print(model.summary())
        return model

    def predict(self,state):
        digits = self.critic.predict(state)
        #print("critic prediction", digits)
        return digits

    def fit(self,states,y_true,epochs,batch_size):
        self.critic.fit(states, y_true, epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size,callbacks=[self.loss_printer])

    def save(self, path):
        self.critic.save_weights(path)

    def load(self, path):
        self.critic.load_weights(path)

"""
A PPO agent class using images as input
"""
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_size,
        clip_ratio=0.2,
        lr_a=1e-3,
        lr_c=3e-3,
        beta=1e-3
    ):
        self.name = 'ppo_agent'
        self.action_size = action_size
        self.Actor = Actor_Model(state_dim,action_size,clip_ratio,lr_a,beta)
        self.Critic = Critic_Model(state_dim,lr_c)

    def action(self, state):
        pred = np.squeeze(self.Actor.predict(state), axis=0)
        act = np.random.choice(self.action_size,p=pred) # index of actions
        val = np.squeeze(self.Critic.predict(state), axis=0)
        # print("prediction, action, value:", pred, act, val)
        return pred, act, val

    def train(self, data, batch_size, iter_a=80, iter_c=80):
        states = data['states']
        actions = np.vstack(data['actions'])
        predictions = np.vstack(data['predictions'])
        advantages = np.vstack(data['advantages'])
        returns = np.vstack(data['returns'])
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        # training Actor and Crtic networks
        print("training Actor network...")
        self.Actor.fit(states, y_true, iter_a, batch_size)
        print("training Actor network...")
        self.Critic.fit(states, returns, iter_c, batch_size)


def draw(fig, env, agent, map):
    ax = fig.add_subplot(111)
    map.plotMap(ax)
    plt.draw()

    vp, nbvps, obs, coverage = env.reset(0)

    vp.plotView(ax,type=2)
    plt.draw()
    plt.pause(0.5)

    epReturn, epLength, dist = 0.0, 0.0, 0.0
    for step in range(30):
        pred, act, val = agent.action([obs])
        vpIdx = nbvps[act]
        nvp, nbvps, r, n_obs, coverage = env.step(vpIdx)

        vp.plotView(ax,type=2)
        plt.draw()
        plt.pause(0.5)

        dist += vpDistance(vp, nvp)
        epReturn += r
        epLength += 1
        vp = nvp
        obs = n_obs
        if coverage == 1.0:
            break;

    plt.text(-3,-5,"Done with {:.2f} meters traveling {} vps.".format(dist,int(epLength+1)))
    plt.text(-3,-7,"Total reward {:.2f}".format(epReturn))
    plt.draw()
    plt.pause(1)
    ax.cla()
    plt.clf()


def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cn', type=float, default=0.9) # control parameter for neighbors choice
    parser.add_argument('--ad', type=int, default=8) # action dimenstion, how many neighbors
    parser.add_argument('--max_ep', type=int, default=10000)

    return parser.parse_args()


if __name__ == "__main__":
    args = getParameters()

    map = GridMap()
    map.makeMap(30,30,1)
    vps = generateViewPoints(gridMap = map,fov = (60.0,60.0), resolution=1.0)
    util = ViewPointUtil(map,vps,nb=args.ad,overlapRatio=args.cn)

    model_dir = os.path.join(sys.path[0],'saved',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    print("model is saved to", model_dir)
    summary_writer = tf.summary.create_file_writer(model_dir)
    summary_writer.set_as_default()

    # create a environment
    buffer_cap = 600
    train_freq = 500
    env = CPPEnv(util)
    action_size = args.ad
    max_step = 30
    state_dim = 4*(action_size+1) # x,y,z of the vp and its neighborhood vps
    agent = PPOAgent(state_dim=state_dim, action_size=action_size, clip_ratio=0.2, lr_a=1e-3, lr_c=3e-3, beta=0.01)
    buffer = ReplayBuffer(input_shape=state_dim, action_size=action_size, size=buffer_cap)

    fig = plt.figure(figsize=(map.height/5,map.width/5)) # inch

    success_counter = 0
    for ep in range(args.max_ep):
        vpIdx = 0 #np.random.randint(len(vps))
        vp, nbvps, obs, coverage = env.reset(vpIdx)
        epReturn, epLength = 0, 0
        for step in range(max_step):
            pred, act, val = agent.action([obs])
            vpIdx = nbvps[act]
            vp, nbvps, r, n_obs, coverage = env.step(vpIdx)
            buffer.store(obs,tf.one_hot(act,action_size).numpy(),r,pred,val)
            obs = n_obs
            epReturn += r
            epLength += 1
            if coverage == 1.0:
                success_counter += 1
                break;

        tf.summary.scalar("episode total reward", epReturn, step=ep+1)
        print("Episode:{},EpReturn:{:.4f},EpLength:{},Success:{}".format(ep+1, epReturn, epLength, success_counter))

        buffer.ep_update(gamma=0.99, lamda=0.97)
        size = buffer.size()
        if size >= train_freq or (ep+1) == args.max_ep:
            print("ppo training with ",size," experiences...")
            agent.train(data=buffer.get(), batch_size=size, iter_a=120, iter_c=120)

        if ep % 100 == 0:
            draw(fig, env, agent, map)
