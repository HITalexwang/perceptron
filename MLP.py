import os,sys,time
import numpy as np

class HiddenLayer(object):
    def __init__(self,input,n_in,n_out,W=None,b=None):
        """
        W=(n_in,n_out) len(b)=n_out
        """
        self.input=input
        if W is None:
            W=np.zeros([n_in,n_out])
        if b is None:
            b=np.zeros(n_out,)
        self.W=W
        self.b=b

        lin_output=np.dot(input,self.W)+self.b
        self.output=self.act(lin_output)
        self.params=[self.W,self.b]
        
    def act(self,lin):
        return 1.0/(1+np.exp(-lin))

class LogisticRegression(object):
    def __init__(self,input,n_in,n_out):
        self.W=np.zeros([n_in,n_out])
        self.b=np.zeros(n_out,)
        self.p_y_given_x=self.sigmoid(np.dot(input,self.W)+self.b)
        self.y_pred=np.argmax(self.p_y_given_x,axis=1) #max according to line
        self.params=[self.W,self.b]
        self.n_out=n_out

    def sigmoid(self,p):
        """
        Z=np.sum(p,axis=1)
        ZZ=np.tile(Z,(self.n_out,1))
        ZZ=np.transpose(ZZ)
        return np.divide(p,ZZ)
        """
        return 1.0/(1+np.exp(-p))

class MLP(object):
    def __init__(self,input,n_in,n_hidden,n_out):
        self.hiddenLayer=HiddenLayer(input,n_in,n_hidden)
        self.logReg=LogisticRegression(self.hiddenLayer.output,n_hidden,n_out)
        self.outputs=self.logReg.p_y_given_x
        self.hidden=self.hiddenLayer.output
        self.W1=self.hiddenLayer.W
        self.W2=self.logReg.W



if __name__=="__main__":
    hl=HiddenLayer(1,2,3)
    """
    a=np.array([[1,2],[3,4],[5,6]])
    Z=np.sum(a,axis=1)
    ZZ=np.tile(Z,(2,1))
    print np.transpose(ZZ)
    """
    
