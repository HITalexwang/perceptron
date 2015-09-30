import random

def getdata(file):
    input=open(file)
    data=[]
    for line in input:
        d=[]
        fields=line.strip().split()
        for i in fields:
            d.append(int(i))
        data.append(d)
    return data

#randomly create data seperated by 2x+y=5, >5 is +1,<=5 is -1
def getdata_random(n):
    data=[]
    for line in range(n):
        d=[]
        x=random.uniform(-100,100)
        y=random.uniform(-100,100)
        d.append(x)
        d.append(y)
        if 2*x+y>5:
            d.append(1)
        else:
            d.append(-1)
        data.append(d)
    return data

class perceptron(object):
    def __init__(self,w,b,data=None,eta=None):
        self.w=w
        self.b=b
        if data is not None:
            self.data=data
        if eta is not None:
            self.eta=eta
    
    def GD(self):
        flag=0
        for line in self.data:
            if line[2]*(self.w[0]*line[0]+self.w[1]*line[1]+self.b)<=0:
                self.w[0]+=self.eta*line[2]*line[0]
                self.w[1]+=self.eta*line[2]*line[1]
                self.b+=self.eta*line[2]
                flag=1
        return (flag,self.w,self.b)

    def pred(self,x,y):
        pred=self.w[0]*x+self.w[1]*y+self.b
        if pred>0:
            print "(",x,",",y,")'s label is: +1"
        else:
            print "(",x,",",y,")'s label is: -1"

def train(data,maxiter,eta):  
    flag=1
    time=maxiter
    w=[0,0]
    b=0
    classifier=perceptron(w,b,data,eta)
    while(flag)&(time>0):
        (flag,w,b)=classifier.GD()
        #print w,b
        time-=1
    if time==0:
        print "did not find the best solution"
        print w,b
    else:
        print "after iterations:",maxiter-time
        print "the best solution is:",w,b
    return (w,b)
    
if __name__=="__main__":
    #data=getdata("test.txt")
    data=getdata_random(5000)
    (w,b)=train(data,10000,0.01)
    classifier=perceptron(w,b)
    classifier.pred(1.1,2)
    classifier.pred(2.3,3)
    
