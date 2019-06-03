#initial variables
alpha=1    #learning rate
conv=0.000001    #convergence
P=1.           #initial value of all the parameters(theta)
ITE=400         #maximum number of iterations

def par_calc(train,valid,a,conv,P,ITE):
    t_x=np.array(train.iloc[:,:-1])
    k=np.array([[1.] for i in range(train.shape[0])])
    t_x=np.hstack((k,t_x))
#    print(t_x[0:5,:])
#    print(t_x.shape)
    t_y=np.array(train.iloc[:,11:])
#    print(t_y[0:5,:])
    v_x=np.array(valid.iloc[:,:-1])
    k=np.array([[1.] for i in range(valid.shape[0])])
    v_x=np.hstack((k,v_x))
    v_y=np.array(valid.iloc[:,11:])
    p=np.array([[P] for i in range(t_x.shape[1])])
#    print(p)
    n=t_x.shape[0]
    error=(np.sum((np.dot(t_x,p)-t_y)**2))/(2*n)
#error calculation
    ite=0
    while True:
#        print(p)
        if ite==ITE:
            break
        ite+=1
        j=(np.dot(t_x,p)-t_y)
        prev_error=error
        error=np.sum(j**2)/(2*n)
        if ite>1 and abs(prev_error-error)<0.5:
#            print(prev_error-error,ite)
            plt.scatter(ite,error)
        print("ITE:",ite," Error:",error)
        if error>prev_error:
            sys.exit("OVERSHOOT STARTED")
        if abs(prev_error-error)<conv and ite>1:
            break
    #parameter update
        for i in range(t_x.shape[1]):
#            print(j[0:5,:])
#            print(np.transpose(t_x[0:5,i]))
#            print(np.multiply(j[0:5,:],(np.ndarray.transpose(t_x[0:5,i]))))
            grad=0
            for J in range(t_x.shape[0]):
                grad+=j[J,0]*t_x[J,i]
                
            p[i,0]=p[i,0]-(a/n)*grad
#validation error
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()
#    print(p.shape,"P")
#    print(v_x.shape,"V_X")
    v_error=np.sum(((np.dot(v_x,p))-v_y)**2)/n
#    print(p)
    return (np.dot(t_x,p)),t_y
#    return p,v_error
            
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
df = pd.read_csv('/home/abhinav/Desktop/Assignment2_AT/winequality-red.csv', delimiter=',')
df=df.sample(frac=1) #randomizing the sample
#k=df.isnull().sum()
#print(k)

#Feature scaling
col = df.columns
mean=df.mean()
for i in col:
    if i=='quality':
        continue
    df[i]=df[i].apply(lambda x:(x-mean[i])/(df[i].max()-df[i].min()))
#     df[i]=df[i].apply(lambda x:(x-df[i].min())/(df[i].max()-df[i].min()))

#checking for individual variable relation with output 
'''for i in col:
    plt.scatter(df[i],df.iloc[:,11])
    plt.ylabel("Quality")
    plt.xlabel(i)
    plt.show()'''

#split in test and train
train_x=df.iloc[:-160,:]
test_x=df.iloc[1598-160+1:,:]

pred_y,v_y = par_calc(train_x,test_x,alpha,conv,P,ITE)

#results
from sklearn.metrics import r2_score
R2=r2_score(v_y,pred_y)
print("R2: ", R2)
acc=0
total=v_y.shape[0]
uni=df['quality'].unique()
for i in range(total):
    dist=1000
    for j in uni:
        if abs(j-pred_y[i])<dist:
            dist=abs(j-pred_y[i])
            k=j
    pred_y[i]=k
    if pred_y[i]==v_y[i,0]:
        acc+=1
print("Accuracy:",acc*100/total,"%")


    
