#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:16:32 2018

@author: helgi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import torch
from torch.autograd import Variable
#import Backgammon
import BG_Competition
#import flipped_agent
device = torch.device('cpu')


# load to global memory
w1 = torch.load('C_w1_trained_99000.pth')
w2 = torch.load('C_w2_trained_99000.pth')
b1 = torch.load('C_b1_trained_99000.pth')
b2 = torch.load('C_b2_trained_99000.pth')
nx = 24 * 2 * 6 + 4

Xw1 = torch.load('X_w1_trained_99.pth')
Xw2 = torch.load('X_w2_trained_99.pth')
Xb1 = torch.load('X_b1_trained_99.pth')
Xb2 = torch.load('X_b2_trained_99.pth')
Xnx = 24 * 2 * 6 + 4

'''
def one_hot_encoding(board, nSecondRoll):
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1)
    # where are the zero, single, double, ... discs
    for i in range(0,5):
        oneHot[i*24+np.where(board[1:25] == i)[0]-1] = 1
    # anything above 4 should be also labelled
    oneHot[5*24+np.where(board[1:25] >= 5)[0]-1] = 1
    # now repeat the process but for other player "-1"
    for i in range(0,5):
        oneHot[6*24+i*24+np.where(board[1:25] == -i)[0]-1] = 1
    # anything above 4 should be also labelled
    oneHot[6*24+5*24+np.where(board[1:25] <= -5)[0]-1] = 1
    # now add the jail and home bits
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = nSecondRoll
    return oneHot
'''

def one_hot_encoding(board):
    one_hot = []
    for i in range(1,len(board)):
        #create a vector with all possible quantities
        one_hot_place = np.zeros( (2 * 15) + 1 )
        
        if(board[i] == 0):    
            place_in_vector = 0
        elif (board[i] > 0):
            place_in_vector = int(board[i])
        else:
            place_in_vector = 15 + -1*int(board[i])
        
        one_hot_place[place_in_vector] = 1
        one_hot.extend(one_hot_place)
    return one_hot


def chooseMove(board, possible_moves, possible_boards, player):
    va = np.zeros(len(possible_moves))
    xa = np.zeros((len(possible_moves),868))
    for i in range(0,len(possible_moves)):
        xa[i,:] = one_hot_encoding(possible_boards[i])
    x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = device))
    h = torch.mm(w1,x) + b1
    h_sigmoid = h.sigmoid()
    #pi = torch.mm(theta,h_sigmoid).softmax(1)
    #xtheta_mean = torch.sum(torch.mm(h_sigmoid,torch.diagflat(pi)),1)
    #xtheta_mean = torch.unsqueeze(xtheta_mean,1)
    #m = torch.multinomial(pi,1)
    y = torch.mm(w2,h_sigmoid)+ b2
    va = y.sigmoid().detach().cpu()
    bestMove = np.argmax(va)
    return possible_boards[bestMove],possible_moves[bestMove]

def flip_board(board_copy):
    #flips the game board and returns a new copy
    idx = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
    12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])
    flipped_board = -np.copy(board_copy[idx])
        
    return flipped_board

def flip_move(move):
    if len(move)!=0:
        for m in move:
            for m_i in range(2):
                m[m_i] = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                                12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])[m[m_i]]        
    return move

# this epsilon greedy policy uses a feed-forward neural network to approximate the after-state value function
def action(board, dice, oplayer, i = 0):

    flippedplayer = -1
    if (flippedplayer == oplayer): # view it from player 1 perspective
        board = flip_board(np.copy(board))
        player = -oplayer # player now the other player +1
        playerTemp = oplayer
    else:
        player = oplayer
        playerTemp = oplayer
    possible_moves, possible_boards = BG_Competition.legal_moves(board, dice, player)
    na = len(possible_boards)
    if (na == 0):
        return []
    
    '''
    xa = np.zeros((na,nx+1))
    va = np.zeros((na))
    for j in range(0, na):
        xa[j,:] = one_hot_encoding(possible_boards[j],i)
    x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = device))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu()
    action = possible_moves[np.argmax(va)]
    if (flippedplayer == oplayer): # map this move to right view
        action = flipped_agent.flip_move(action)
    '''
    after_state,action = chooseMove(board, possible_moves, possible_boards, player)
    
    move = action
    if playerTemp == -1: move = flip_move(action)
    if playerTemp:
        return move
    
    return action
