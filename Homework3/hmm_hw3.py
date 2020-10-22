# -*-coding:Latin-1 -*

# Machine Learning 2 Homework 3
print("\nMachine Learning 2 Homework 3")
# By Kutlu Toren & Louis Devèze
print("By Kutlu Toren & Louis Deveze")
# Implementation of a Spam filter using Naive Bayes implementation
print("Implementation of Hidden Markov Model\n")

# --------------------
# READ ME
# To run the program just enter the command line python hmm_hw3.py 
#
#


# --------------------
# importing packages
import numpy as np


# This function calculate Xj(t) with respect to Xj(t-1), aij, bjk, and dimentions for the output matrix predefined as 4
def calculateXjt(aij,bjk,xjt1,column,T=4,J=4):
    xjt=np.zeros(J)
    for n in range(0,J):
        middleSum=0
        for m in range(0,T):
            #Uncomment bellow line to see the calculation of each probability in state time matrix
            #print("X{}= {} a{}{} = {}  , b{}{} = {}       {}".format(m,xjt1[m],m,n,aij[m,n],m,n ,bjk[n,column],   xjt1[m]*aij[m,n]*bjk[n,column]))
            middleSum=middleSum+xjt1[m]*aij[m,n]*bjk[n,column]
        print("Z{}(t) = {}".format(n,middleSum))
        xjt[n]=middleSum
    
    return xjt

#This function constructs the matrix of Xj(t) values using calculateXjt function.
# Xjt is the return variable which is an matrix; aij,bjk, initial xj(t=0) and the vector of observed states taken as input  
def forwardAlgorithm(aij,bjk,xj0,x_observed):

    xjt=xj0
    print("for t = 1")
    xjt=np.vstack([xjt, calculateXjt(aij,bjk,xj0,x_observed[0])])

    #This loop appends the xjt matrix (vector in beginning) with xj(t+1)
    for t in range(1,4):
        print("\nfor t = {}".format(t+1))
        xjt=np.vstack([xjt, calculateXjt(aij,bjk,xjt[t,:],x_observed[t])])

    #transpose the xjt to have state in rows and time in columns
    xjt= np.transpose(xjt)

    return xjt


# This function utilize calculateXjt and from the maximum value in each t, determine the most probable hidden states for that observation
# Path is the return variable which is an matrix; aij,bjk, initial xj(t=0) and the vector of observed states taken as input
# Different than Forward Algorithm there is no need to store xjt values. The index of the maximum value stored in path.
def decodingAlgorithm(aij,bjk,xj0,x_observed):
    path =[] 
    xjt=xj0  
    print("Argmaxj'X0(t) = {} at index {}\n".format(max(xjt),np.argmax(xjt)))
    path.append(np.argmax(xjt))
    
    xjt= calculateXjt(aij,bjk,xj0,x_observed[0])
    print("Argmaxj'X1(t) = {} at index {}\n".format(max(xjt),np.argmax(xjt)))
    path.append(np.argmax(xjt))

    #This loop appends the xjt matrix (vector in beginning) with xj(t+1)
    for t in range(1,4):
        xjt=calculateXjt(aij,bjk,xjt,x_observed[t])
        print("Argmaxj'X{}(t) = {} at index {}\n".format(t+1,max(xjt),np.argmax(xjt)))
        path.append(np.argmax(xjt))
    
    return path



#  ******* Start of initialization of parameters

#initiate aij and bjk as per example

#We supposed x = {x1, x3, x2, x0}, z(0) = z1
x_observed= [1,3,2,0] # this will be used to iterate over columns of bjk

xj0= np.array([0,1,0,0]) #This is the initial xj(t=0) considering z(0)=z(1)

I=4 #defining row number I for aij
J=4 #defining column number J for aij and row number J for bjk
K=5 #defining column number J for bkj

aij=np.array([[1  ,0  ,0  ,0  ],
              [0.2,0.3,0.1,0.4],
              [0.2,0.5,0.2,0.1],
              [0.7,0.1,0.1,0.1]])

bjk=np.array([[1  ,0  ,0  ,0  ,0  ],
              [0  ,0.3,0.4,0.1,0.2],
              [0  ,0.1,0.1,0.7,0.1],
              [0  ,0.5,0.2,0.1,0.2]])


#Control of the initial definition
#print(type(aij))
#print(type(bjk))
#print("aij is: \n {} ".format(aij))
#print("bjk is: \n {} ".format(bjk))

# ******* end of initialization of parameters


#Encoding Problem
xjt= forwardAlgorithm(aij,bjk,xj0,x_observed)

np.set_printoptions(suppress=True) # this line is to avoid seeing in scientific notation
print("Resulting state time matrix for the example is: \n {} \n\n P(x|M) = {}\n".format(xjt,xjt[0,4]))
    
#Decoding Problem 

path = decodingAlgorithm(aij,bjk,xj0,x_observed)

print("The path is: "),
for i in path:
    print("Z{} ".format(i)),
print("\n")

