# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random   #to take random samples from different batches during experience replay
import os       #to load & save the model
import torch    #neural network using pytorch
import torch.nn as nn
import torch.nn.functional as F    #activation/loss functions are part of this module
import torch.optim as optim       #optimizer
import torch.autograd as autograd   #put tensor (advanced matrix) into a variable which contains gradient
from torch.autograd import Variable #converts torch tensor to the variable that contains tensor and gradient


# Creating the architecture of the Neural Network
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        #input_size = 5 (3 sensor, +&- orientation)
        #nb_action = 3 (straight, left, right)
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)   #full connection between neurons of input layer to hidden layer with 30 neurons 
        self.fc2 = nn.Linear(30, nb_action)    #full connection between neurons of hidden layer to output layer which has 3 action to perform
    
    def forward(self, state):
        #Activates the Neural network and return Q-value for each state
        x = F.relu(self.fc1(state))  #x represents hidden neurons which are activated using Relu function
        q_values = self.fc2(x)
        return q_values
    
#Implementing Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity #maximum no. transition we can have in memory of events
        self.memory = []         #contains last 100 events in agent's memory
    
    def push(self, event):
        #used for appending a new event in the memory 
        #event is tuple of 4 elements - last state(st), new state(st+1), last action (at), last reward(rt)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)
    
# Implementing Deep Q Learning
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma   #discount factor or delay coefficient
        self.reward_window = []
        self.model = Network(input_size, nb_action)  #Neural Network object
        self.memory = ReplayMemory(100000)  #capacity = 100000 transitions in the memory then we will sample from this memory to get small no. of transition on which model will learn
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)   #lr is learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*20) # probs is probability, Temperature parameter T increases the probability or certainity of action to take = 100
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)  #output of neural network
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)   #Temporal Difference loss
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) #improves back probagation . retain_variables = True free some memory
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        #compute the mean of all rewards on sliding reward window
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        #Save the brain/model of the car for later use whenever we quit the application 
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")

