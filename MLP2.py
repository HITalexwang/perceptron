# -*- coding: cp936 -*-
import os,sys,time,random
import numpy as np

class MLP(object):
    def __init__(self,input,n_in,n_hidden,n_out,W1=None,W2=None):
        self.input=input
        #self.W1=np.zeros([n_in,n_hidden])
        if W1 is None:
            self.W1=np.random.random((n_in,n_hidden))
        else:
            self.W1=W1
        #self.b1=np.zeros(n_hidden,)
        if W2 is None:
            self.W2=np.random.random((n_hidden,n_out))
        else:
            self.W2=W2
        #self.b2=np.zeros(n_out,)
        
    def hidden(self):
        """
        W1=(n_in,n_out) len(b)=n_out
        """
        lin_output=np.dot(self.input,self.W1) #+self.b1
        self.hiddens=self.sigmoid(lin_output)
        
    def output(self):
        """
        hidden=(n_example,n_in),W2=(n_in,n_out)
        """
        self.outputs=self.sigmoid(np.dot(self.hiddens,self.W2))
        self.y_pred=np.argmax(self.outputs,axis=1) #max according to line

    def sigmoid(self,p):
        return 1.0/(1+np.exp(-p))

def train(max_iter,eta,targets,inputs):
    #input=[[1,3],[4,5]]
    claf=MLP(inputs,2,4,4)
    claf.hidden()
    claf.output()
    time=max_iter
    new_deltao=1
    old_deltao=0
    deltao=np.array([[-100]])
    while((time>0)&(abs(new_deltao-old_deltao)>0.001)):
        old_deltao=sum(sum(deltao))
        deltao=(targets-claf.outputs)*claf.outputs*(1.0-claf.outputs)
        deltah=claf.hiddens*(1.0-claf.hiddens)*(np.dot(deltao,np.transpose(claf.W2)))
        new_deltao=sum(sum(deltao))
        
        updateW1=np.zeros((np.shape(claf.W1)))
        updateW2=np.zeros((np.shape(claf.W2)))

        updateW1=eta*(np.dot(np.transpose(inputs),deltah[:,:]))
        updateW2=eta*(np.dot(np.transpose(claf.hiddens),deltao))
        claf.W1 +=updateW1
        claf.W2 +=updateW2
        claf.hidden()
        claf.output()
        #print old_deltao,new_deltao
        time-=1
    #print time,claf.W1
    #print claf.W2
    #print time,claf.outputs
    #print time,claf.y_pred
    return (claf.W1,claf.W2)

def test(inputs,W1,W2):
    claf=MLP(inputs,2,4,4,W1,W2)
    claf.hidden()
    claf.output()
    for i in range(len(inputs)):
        print inputs[i],'is in quadrant',claf.y_pred[i]+1
    #print claf.y_pred

def getdata():
    inputs=[]
    targets=[]
    t=[]
    inp=[]
    for i in range(200):
        inp=[random.randint(-10,10),random.randint(-10,10)]
        if inp[0]>0:
            if inp[1]>0:
                targets.append([1,0,0,0])
            else:
                targets.append([0,0,0,1])
        else:
            if inp[1]>0:
                targets.append([0,1,0,0])
            else:
                targets.append([0,0,1,0])
        inputs.append(inp)
    return (inputs,targets)
        
        
if __name__=="__main__":
    """
    a=np.array([[1,2],[3,4],[5,6]])
    Z=np.sum(a,axis=1)
    ZZ=np.tile(Z,(2,1))
    print np.transpose(ZZ)
    """
    (inputs,targets)=getdata()
    inpu=[]
    for t in targets:
        if t[0]==1:
            inpu.append(0)
        if t[1]==1:
            inpu.append(1)
        if t[2]==1:
            inpu.append(2)
        if t[3]==1:
            inpu.append(3)
    #targets=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    #inputs=[[30,40],[-1,2],[-3,-10],[10,-20]]
    (W1,W2)=train(1000,0.1,targets,inputs)
    test([[11,-3],[8,12],[-100,-3],[-31,5]],W1,W2)
    
