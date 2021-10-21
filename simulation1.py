    # -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:45:46 2021

@author: zby09
"""
import queue
import numpy as np
import math
class TSN_env():
#This project only simulates the highest priority queue in TSN switches
    def __init__(
        self, 
        hyperperiod, 
        GCL,
        flowOffsets,#arrival time of frames in this switch
        flowID, #flowID
        flowToWindow, #The assigned window for each flow, it is a list of list. If the element list of flowToWindow is a list with more than one elements, it means that that flow send more than one frame in each hyperperiod
        bitVector,#a vector of 1 and 0. bitVector[i] is 1 means that the gate is open in GCL[i]
        terminate,# not used currently
        transTime#transmission time of each frame belonging to different flows
    ):
        self.hyperperiod = hyperperiod
        self.bitVector = bitVector
        self.GCL = GCL
        self.flowOffsets = flowOffsets
        self.flowID = flowID
        self.hpindex = 0
        self.queue = queue.Queue()
        self.seqNum = []
        self.flowToWindow = flowToWindow
        self.misbehaviors = [] #misbehavior[i] stores the misbehavior informantion of hyperperiod i. Each element is a dictionary
        self.terminate = terminate
        self.clock = 0
        self.transTime = transTime
        self.busyPeriod = 0
        self.original_GCL = self.GCL.copy()
        self.penalty = 6 # penalty for frames not sent in its assigned hyperperiod, which are the frame in the queue at the end of the hyperperiod. 
        for i in range(len(flowID)):
            self.seqNum.append(0)
    def checkGate(self):#check the gate status
        sum = [0]
        for i in range(1,len(self.GCL)+1):
            sum.append(0)
            sum[i] = sum[i-1]+self.GCL[i-1]
        for i in range(len(self.GCL)):
            if(self.clock<sum[i+1]):
                return  self.bitVector[i], i
        
    def getHyperperiod(self):
        return self.hyperperiod
    def getFlow(self):
        return self.flow
    def updateHpindex(self):
        self.hpindex = self.hpindex+1
    def enQueue(self,ID,seq):
        temp = str(ID)+":"+str(seq)
        #print("enqueue: "+temp)
        self.queue.put(temp)    
    def updateClock(self):
        self.clock = self.clock+1   
    def setGCL(self,GCL):
        self.GCL = GCL.copy()
    def getMisbehaviors(self,packet,windex):
        #print(packet)
        temp = packet.split(":")
        ID = int(temp[0])
        seq = int(temp[1])
        ind = self.flowID.index(ID)
        numPackets = len(flowToWindow[ind])
        expectedHpindex = int(seq/numPackets)-1
        expectedWindex = flowToWindow[ind][seq%numPackets]
        delta = self.hpindex*len(self.GCL)+windex-expectedHpindex*len(self.GCL)-expectedWindex
        while(len(self.misbehaviors)<self.hpindex+1):
            self.misbehaviors.append({})
        if(delta in self.misbehaviors[self.hpindex]):
            self.misbehaviors[self.hpindex][delta] = self.misbehaviors[self.hpindex][delta]+1
        else:
            self.misbehaviors[self.hpindex][delta] = 1
        return self.transTime[ind]
    def getGCL(self):
        return self.GCL
    def getRewardAndState(self):# return the rewards and the state for the reinforcement learning model
        reward = 0;
        currentMis = []
        if(len(self.misbehaviors)>=self.hpindex):
            currentMis = self.misbehaviors[self.hpindex-1]           
        state = []
        for i in range(50):
            state.append(0)
        for key in currentMis:
            reward = reward-max(0,key)*currentMis[key]
            if(key>=0):
                state[key] = currentMis[key]
        reward = reward-self.queue.qsize()*self.penalty
        rewardForGateLength = 0
        total_length = 0
        wait_time = 0
        total_wait = 0;
        for i in range(len(self.GCL)):
            if(self.bitVector[i] == 1):
                total_length = total_length+self.GCL[i]
                total_wait = wait_time+total_wait
            wait_time = wait_time=self.GCL[i]
                
                #print(total_length)
        #print(reward)
        reward = reward*max(1,math.sqrt(total_length/np.sum(self.transTime)))-max(0,total_length-np.sum(self.transTime))-total_wait
        print(total_wait)
        #eward = reward+total_length/10
        state.append(self.queue.qsize())#The last element is to record the number of packets not being sent in the current hyperperiod
        state.append(total_length)
        state = state+self.GCL
        state.append(total_length)
        state.append(np.sum(self.transTime))
        return reward, state
        
    def running(self):
        while(self.clock<=self.hyperperiod):
            #print(self.clock)
            if(self.clock == self.hyperperiod):
                self.updateHpindex()
                self.clock = 0
                if(self.busyPeriod>self.hyperperiod):
                    self.busyPeriod = self.busyPeriod-self.hyperperiod
                else:
                    self.busyPeriod = 0
                break
            else:
                gateState, windex = self.checkGate()
                if(gateState and self.clock >= self.busyPeriod):
                    a = np.array(self.flowOffsets)
                    inds = np.where(a == self.clock)[0]
                    if(self.queue.empty() == False):
                        packet = self.queue.get()
                        transT = self.getMisbehaviors(packet,windex)
                        self.busyPeriod = self.clock+transT
                        for i in range(len(inds)):                      
                            self.seqNum[int(inds[i])] = self.seqNum[int(inds[i])]+1
                            packet = str(self.flowID[inds[i]])+":"+str(self.seqNum[inds[i]])
                            self.enQueue(self.flowID[inds[i]], self.seqNum[inds[i]])
                    else: 
                        
                        for i in range(len(inds)):
                            self.seqNum[inds[i]] = self.seqNum[inds[i]]+1
                            packet = str(self.flowID[int(inds[i])])+":"+str(self.seqNum[int(inds[i])])
                            if(i == 0):
                                transT = self.getMisbehaviors(packet,windex)
                                self.busyPeriod = self.clock+transT
                            else:
                                self.enQueue(self.flowID[int(inds[i])], self.seqNum[int(inds[i])])
                else:
                    a = np.array(self.flowOffsets)
                    inds = np.where(a == self.clock)[0]
                    for i in range(len(inds)):                      
                        self.seqNum[int(inds[i])] = self.seqNum[int(inds[i])]+1
                        packet = str(self.flowID[inds[i]])+":"+str(self.seqNum[inds[i]])
                        self.enQueue(self.flowID[inds[i]], self.seqNum[inds[i]])                        
                self.updateClock()
        reward, state = self.getRewardAndState()
        return reward, state
                                
def getActions(m,s, granularity,i,actions,action, csum):
    if(csum>s):
        return
    if(i == m-1):
        action.append(s-csum)
        actions.append(action)
        return
    if(csum>s):
        return
    for j in range(int(s/granularity)+1):
        temp = action.copy()
        temp.append(j*granularity)
        getActions(m,s, granularity,i+1,actions,temp, csum+j*granularity)
        
action = []        
actions = []              
getActions(3,100,5,0,actions,action,0)                        
        
flowID = [1,2,3,4,5]
flowOffsets = [10,20,30,40,50]
hyperperiod = 100
GCL = [0,98,2]
bitVector = [0,1,0]
flowToWindow = [[1],[1],[1],[1],[1]]
transTime = [10,10,10,10,10]
TSN = TSN_env(hyperperiod,GCL,flowOffsets,flowID,flowToWindow,bitVector,10,transTime)
reward, state = TSN.running()
'''rewards = []
for i in range(10):
    reward, state = TSN.running()
    rewards.append(reward)'''
