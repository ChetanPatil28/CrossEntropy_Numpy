import torch

def inbuilt_CE(y,yhat,cw,reduce=False):
    print('Loss in Pytorch is',nn.CrossEntropyLoss(torch.FloatTensor(cw),reduce=reduce)(yhat, y))
    
def numpy_CE(y,yhat,cw,reduce=False):
    y,yhat,cw=y.numpy(),yhat.numpy(),cw.numpy()  
    L=[]
    n,c,d1,d2=yhat.shape
    loss=np.empty_like(y)
    for i in range(n):
        for k in range(d1):
            for l in range(d2):
                clas=y[i,k,l]
                num=np.exp(yhat[i,clas,k,l])
                denom=np.exp(yhat[i,:,k,l]).sum()
                ce=-np.log(num/denom)*cw[clas]
                L.append(ce)
    loss=np.asarray(L).reshape(y.shape)
    if reduce==False:
        print("Loss in Numpy is ",loss)
    else:
        print("Loss in Numpy is ",loss.sum())


yhat=torch.randn(2,3,1,2) ## (N,C,d1,d2.....dk)
y=torch.LongTensor([[[1,0]],[[1,2]]])
class_weights=torch.tensor([7,8,4]).float()

inbuilt_CE(y,yhat,class_weights,reduce=False)
print('\n\n')
numpy_CE(y,yhat,class_weights,reduce=False)        