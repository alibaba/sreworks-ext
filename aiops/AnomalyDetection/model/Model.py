#modify coherence and lossFunc MCNet2
#modify attRes -> gateIncrease procedure, introduce optimal transport here MCNet3
#modify attRes gateIncrease ratios during training process MCNet4
#add layerNorm MCNet5
#constrain transport matrix P MCNet6
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import random
import math
from optimialTrans import opt
from eval_methods import pot_eval,searchThreshold
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class myDataset(Dataset):
    def __init__(self,datas,winlen,labels=None,type="train"):
        super(myDataset,self).__init__()
        if type=="train" or type=="validate":
            self.x=self.split(datas,winlen)
        else:
            self.x,self.y=self.splitTest(datas,winlen,labels)
        self.type=type

    def split(self,datas,winlen):
        xs=[]
        for i in range(len(datas)-winlen):
            xs.append(datas[i:i+winlen])
        return xs

    def splitTest(self,datas,winlen,labels):
        xs=[]
        ys=[]
        for i in range(len(datas)-winlen):
            xs.append(datas[i:i+winlen])
            ys.append(labels[i:i+winlen])
        return xs,ys

    def __getitem__(self, item):
        item = item % self.__len__()
        if self.type=="train" or self.type=="validate":
            return self.x[item]
        else:
            return (self.x[item],self.y[item])


    def __len__(self):
        return len(self.x)


class DecompositionBlock(nn.Module):
    def __init__(self, pars):
        super(DecompositionBlock,self).__init__()
        #self.conv=nn.Conv1d(pars.fealen,pars.fealen,pars.klen,1)
        self.conv=nn.AvgPool1d(pars.klen,1)
        self.klen=pars.klen
    def forward(self,x):#x:batchsize,winlen,fealen
        batch,winlen,fealen=x.shape
        x=x.permute(0,2,1)#batchsize,fealen,winlen
        xPadding=torch.cat([x[:,:,:self.klen-1],x],dim=-1)
        xconv=self.conv(xPadding)
        xdiff=x-xconv
        xconv=xconv.permute(0,2,1)
        xdiff=xdiff.permute(0,2,1)#batchsize,winlen,fealen
        return xdiff,xconv


class GaussianCurve(nn.Module):
    def __init__(self,rows,cols,center,pars):
        super(GaussianCurve,self).__init__()
        self.sigmas=nn.Parameter(torch.ones(rows,1))
        self.rows=rows
        self.cols=cols
        self.center=center
        self.device=pars.device
    def forward(self):
        xs = torch.arange(0, self.cols)
        xs = xs.repeat(self.rows, 1).to(self.device)
        if isinstance(self.center,list):
            centers=torch.tensor(self.center,dtype=torch.float)
            centers=centers.repeat(self.cols,1)
            centers=centers.permute(1,0)
        else:
            centers=torch.ones((self.rows,self.cols))*self.center
        centers=centers.to(self.device)
        gauss=torch.pow(xs-centers,2)#rows,cols
        gauss=gauss.permute(1,0)
        gauss/=-2*torch.pow(self.sigmas,2)
        gauss=torch.exp(gauss)/self.sigmas#cols,rows
        gausSum=gauss.sum(dim=0)
        gauss/=gausSum
        gauss=gauss.permute(1,0)
        return gauss


class gate(nn.Module):
    def __init__(self,pars):
        super(gate,self).__init__()
        self.atts=nn.ModuleList([nn.MultiheadAttention(pars.patchSize, 1, batch_first=True,kdim=pars.patchSize,vdim=1) for i in range(pars.fealen)])
        #self.att1 = nn.MultiheadAttention(pars.patchSize, 1, batch_first=True,kdim=pars.patchSize,vdim=1)
        #self.att2 = nn.MultiheadAttention(pars.patchSize, 1, batch_first=True, kdim=pars.patchSize, vdim=1)
        self.activ=nn.Sigmoid()
        #self.relu=nn.ReLU()
        self.patchSize=pars.patchSize
        #self.posCurve=GaussianCurve(1, pars.fealen,0,pars)
        #self.negCurve=GaussianCurve(1, pars.fealen,pars.fealen-1,pars)
        self.attCurve=GaussianCurve(pars.winlen,pars.winlen,[i for i in range(pars.winlen)],pars)
        self.softmax=nn.Softmax(dim=-1)
        self.device=pars.device
        self.activ2=nn.LeakyReLU()
        self.Wx=nn.Linear(pars.fealen,pars.fealen)
        self.scaler=nn.Parameter(torch.ones(pars.fealen))
        self.bias=nn.Parameter(torch.ones(pars.fealen))
        if pars.omit:
            self.cost=pars.exeTime[1:]
        else:
            self.cost=pars.exeTime
        #self.u=nn.Parameter(torch.ones(pars.fealen))
        #self.v=nn.Parameter(torch.ones(pars.fealen))

    def getK(self):
        intervals=torch.tensor(self.cost)
        X,Y=torch.meshgrid(intervals,intervals)
        epsilon=0.03**2
        K = torch.exp(-torch.pow(X - Y, 2) / epsilon)
        K=K.to(self.device)
        return K

    def getC(self):
        intervals = torch.tensor(self.cost)
        X, Y = torch.meshgrid(intervals, intervals)
        C=X-Y
        C=C.to(self.device)
        return C

    def forward(self,x,ratios):#x:batchsize,winlen,fealen
        x=x.permute(0,2,1)#batch,fealen,winlen
        batchSize,fealen,winlen=x.shape
        xPadding=torch.cat([x[:,:,:self.patchSize],x],dim=-1)
        xExtend=xPadding.unfold(2,self.patchSize+1,1)#batch,fealen,blockNum,patchsize+1
        _,_,blockNum,_=xExtend.shape
        attWeight=[]
        for i,att in enumerate(self.atts):
            _,attWei=att(xExtend[:,i,:,:-1],xExtend[:,i,:,:-1],xExtend[:,i,:,-1:])
            attWeight.append(attWei)
        attWeight=torch.stack(attWeight,dim=0)#fealen,batch,blockNum,blockNum
        attWeight=attWeight.permute(1,0,2,3).reshape(batchSize*fealen,blockNum,blockNum)
        #xExtend=xExtend.reshape(batchSize*fealen,blockNum,self.patchSize+1)
        #_,attWeight=self.att1(xExtend[:,:,:-1],xExtend[:,:,:-1],xExtend[:,:,-1:])#batchSize*fealen,blockNum,1
        attWeightSave=attWeight.clone()
        attWeightSave=attWeightSave.reshape(batchSize,fealen,winlen,winlen)
        attWeight=attWeight*(1-self.attCurve()) #batch*fealen,winlen (tarlen),winlen (sourLen)
        attWeight=self.softmax(attWeight).permute(1,0,2)#tarLen, batch*fealen, sourLen
        #xEmbed=self.linear(xExtend)
        #xEmbed=self.activ2(xEmbed)
        #xExtend = xExtend.reshape(batchSize * fealen, blockNum, self.patchSize + 1)
        value=xExtend[:,:,:,-1:].permute(0,2,3,1)#batch,blocknum,1,fealen
        #value=self.Wx(value).permute(0,3,1,2)
        value=(self.scaler*value+self.bias).permute(0,3,1,2)
        value=value.reshape(batchSize*fealen,blockNum)
        attRes=(attWeight*value).sum(dim=-1) #tarLen, batch*fealen
        attRes=attRes.permute(1,0)#batch*fealen,tarLen
        #attGate,_=self.att2(xExtend[:,-1,:-1],xExtend[:,:-1,:-1],xExtend[:,:-1,-1])#batchSize*fealen,1,1
        attRes=attRes.reshape(batchSize,fealen,blockNum)#batch,fealen,winlen
        attRes=attRes.permute(0,2,1)#batch,winlen,fealen
        #K=self.getK()
        #C=self.getC()
        #P=torch.matmul(torch.diag(self.u),K)
        #P=torch.matmul(P,torch.diag(self.v))
        #gateIncrease=torch.matmul(attRes,P.transpose(1,0))
        #cost=torch.sum(P*C)
        #presIncrease=ratios[0]*attRes+ratios[1]*gateIncrease
        return attRes,attWeightSave,10 #batch,winlen,fealen


class OPT(nn.Module):
    def __init__(self,pars):
        super(OPT,self).__init__()
        self.P=nn.Parameter(torch.diag(torch.ones(pars.fealen)))
        self.device=pars.device
        self.fealen=pars.fealen
        self.softmax=nn.Softmax(dim=-1)
        if pars.omit:
            self.cost=pars.exeTime[1:]
        else:
            self.cost=pars.exeTime

    def getC(self):
        intervals = torch.tensor(self.cost,dtype=torch.float)
        intervals/=intervals.sum()
        X, Y = torch.meshgrid(intervals, intervals)
        C=X-Y
        mask=C<0.
        C[mask]=0.
        C=C.to(self.device)
        return C

    def forward(self,x):#x:batch,winlen,fealen
        C=self.getC()
        P=self.softmax(self.P)
        tx=torch.matmul(x,P)
        batchSize,winlen,fealen=x.shape
        x=x.repeat(1,1,self.fealen).reshape(batchSize,winlen,fealen,fealen)
        cost=torch.mean(P.transpose(1,0)*C*x)
        return tx,cost


class filter(nn.Module):
    def __init__(self,pars):
        super(filter,self).__init__()
        self.curve=GaussianCurve(pars.winlen,pars.winlen,[i for i in range(pars.winlen)],pars)
        self.active=nn.Softmax(dim=-1)
    def forward(self,weight):#batch,winlen,winlen
        curve=1-self.curve()
        coherence=weight*curve#batch,winlen,winlen
        coherence=coherence.sum(dim=-1)
        coherence=self.active(coherence)
        return coherence #batch,winlen


class MCNetModule(nn.Module):
    def __init__(self,pars):
        super(MCNetModule,self).__init__()
        self.decomposition=DecompositionBlock(pars)
        self.gate=gate(pars)
        self.opt=OPT(pars)
    def forward(self,x,ratios):
        xdiff,xconv=self.decomposition(x)
        seasonalTrend,attWeight,_=self.gate(xconv,ratios)
        #recons,cost=self.opt(seasonalTrend)
        return seasonalTrend,attWeight #batch,winlen,fealen


class MCNet(nn.Module):
    def __init__(self,pars):
        super(MCNet,self).__init__()
        self.decompositions=nn.ModuleList([MCNetModule(pars) for i in range(pars.moduleNum)])
        self.device=pars.device
        self.filter=filter(pars)
        self.sigma=pars.sigma
        self.layerNorm=nn.LayerNorm(pars.fealen)
        self.softmax=nn.Softmax(dim=-1)
        self.opt=OPT(pars)
    def forward(self,x,type="test",iterations=0):
        if type=="train":
            ratios=(self.sigma**iterations,1-self.sigma**iterations)
        else:
            ratios=(0,1)
        batch,winlen,fealen=x.shape
        xPres=x
        attWeights=torch.zeros(batch,fealen,winlen,winlen).to(self.device)
        recons=torch.zeros(x.shape).to(self.device)
        reconSingles=[]
        reconAggregs=[]
        for count,decomp in enumerate(self.decompositions):
            recon,attWei=decomp(xPres,ratios)
            reconSingles.append(recon.cpu().detach().numpy())
            recons+=recon
            reconAggregs.append(recons.cpu().detach().numpy())
            attWeights+=attWei #batch,fealen,winlen,winlen
            #rate=0.6**count
            xPres=x-recons
            #xPres=self.layerNorm(xPres)
        attWeights=(attWeights.sum(dim=1)).reshape(batch,winlen,winlen)
        recons=self.softmax(recons)
        recons,costs=self.opt(recons)
        recons=self.softmax(recons)
        coherence=self.filter(attWeights)
        reconSingles=np.stack(reconSingles,axis=0)
        reconAggregs=np.stack(reconAggregs,axis=0)
        return recons,coherence,costs,reconSingles,reconAggregs


def lossFunc(recons,x,coherence,costs,pars):#recons,x:batch,winlen,fealen; coherence:batch,winlen
    #print("recons",recons.mean(),recons.std())
    #print("x",x.mean(),x.std())
    error=torch.pow(x-recons,2)
    #print("error",error.shape,error.mean())
    error=error.sum(dim=-1)
    error*=coherence
    error=error.sum(dim=-1)
    error=error.mean()
    return (1-pars.lamda)*error+pars.lamda * costs #error


def train(dataloader,model,loss_fn,parameters,optimizer,iterations):#optimizer!
    size = len(dataloader.dataset)
    num_batches=len(dataloader)
    model.train()
    for batch, x in enumerate(dataloader):
        x = x.to(parameters.device)
        xoriginal=x.clone()
        recons,coherence,costs,_,_ = model(x,"train",iterations)
        loss = loss_fn(recons,xoriginal,coherence,costs,parameters)
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),2)
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(x)
            # plot(model,X,y,pLen)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate(dataloader,model,loss_fn,parameters):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(parameters.device)
            recons,coherence,costs,_,_ = model(x,"validate")
            loss = loss_fn(recons,x,coherence,costs,parameters)
            test_loss+=loss
    test_loss /= num_batches

    print(f"Validate Error: \n  Avg loss: {test_loss:>8f} \n")
    return test_loss

def getAnomScore3(x,recons,pars):
    scores=x-recons
    scores=scores.sum(axis=-1)
    scores=list(scores)
    return scores

def getAnomScore2(x,recons,pars):
    if pars.omit:
        intervals = np.array(pars.exeTime[1:], dtype=np.float64)
    else:
        intervals = np.array(pars.exeTime, dtype=np.float64)
    scores=(x-recons)*intervals
    scores=scores.sum(axis=-1)
    scores=list(scores)
    return scores

def getAnomScoreDivide(x,recons,pars):
    if pars.omit:
        intervals = np.array(pars.exeTime[1:], dtype=np.float64)
    else:
        intervals=np.array(pars.exeTime,dtype=np.float64)
    scores = (x - recons) * intervals
    return scores

def getAnomScore1(x,recons,pars):
    scores=[]
    if pars.omit:
        intervals=np.array(pars.exeTime[1:],dtype=np.float64)
    else:
        intervals=np.array(pars.exeTime,dtype=np.float64)
    [Y,X]=np.meshgrid(intervals, intervals)
    cost=(Y-X)
    intervals/=intervals[-1]
    #print(intervals)
    for xslice,recon in zip(x,recons):
        P=opt(x.shape[1],intervals,recon.cpu().detach().numpy(),xslice.cpu().detach().numpy())
        score=(P*cost).sum()
        scores.append(score)
    return scores


def slideWindow(scores,coherences,pars):
    winlen=pars.slidewinlen
    scores=scores[:winlen]+scores
    coherences=np.hstack((coherences[:winlen],coherences))
    nscores=[]
    coherence=torch.tensor(coherences)
    for i in range(len(scores)-winlen):
        weight=torch.softmax(coherence[i:i+winlen],dim=0).numpy()
        nscores.append(scores[i]-np.sum(scores[i:i+winlen]*weight))
    return nscores


def reconAdjust(recons,x):
    print(recons.shape)
    recons=(recons-recons.mean(axis=0,keepdims=True))/(recons.std(axis=0,keepdims=True)+0.00001)
    recons=(recons*x.std(axis=0,keepdims=True))+x.mean(axis=0,keepdims=True)
    return recons

def plotSeries(series,colors):
    print(series.shape)
    series=series.transpose(1,0)
    for sery, color in zip(series,colors):
        plt.plot(sery,color=color)

def test(dataloader,model,loss_fn,parameters):
        num_batches = len(dataloader)
        print(num_batches)
        model.eval()
        test_loss = 0
        labels=[]
        reconSeq=[]
        origiSeq=[]
        diffSeq=[]
        coherences=[]
        labelsShow=[]
        reconSigs=[]
        reconAggs=[]
        with torch.no_grad():
            for x,y in dataloader:
                x, y = x.to(parameters.device), y.to(parameters.device)
                labelsShow.append(y[:, -1])
                y = y == 1
                labels.append(y[:, -2:].any(dim=-1))
                recons,coherence,costs,reconSig,reconAgg = model(x,"test")
                #torch.true_divide(torch.abs(x-xu),xstd).sum(dim=(-1))
                test_loss += loss_fn(recons,x,coherence,costs,parameters)
                reconSeq.append(recons[:,-1,:])
                origiSeq.append(x[:,-1,:])
                diffSeq.append(x[:,-1,:]-recons[:,-1,:])
                coherences.append(coherence[:,-1])
                reconSigs.append(reconSig[:,:,-1])
                reconAggs.append(reconAgg[:,:,-1])
        test_loss /= num_batches
        reconSeq=torch.cat(reconSeq,dim=0).cpu().detach().numpy()
        origiSeq=torch.cat(origiSeq,dim=0).cpu().detach().numpy()
        reconSeq=reconAdjust(reconSeq,origiSeq)
        diffSeq=torch.cat(diffSeq,dim=0).cpu().detach().numpy()
        coherences=torch.cat(coherences,dim=0).cpu().detach().numpy()
        reconSigs=np.concatenate(reconSigs,axis=1)
        reconAggs=np.concatenate(reconAggs,axis=1)
        scores=getAnomScore2(origiSeq,reconSeq,parameters)
        divideScores=getAnomScoreDivide(origiSeq,reconSeq,parameters)
        print("score shape",len(scores))
        print("recon shape",reconSeq.shape)
        print("coherences shape", coherences.shape)
        #scores=slideWindow(scores,coherences,parameters)
        scores=np.array(scores)
        labels=torch.cat(labels,dim=0).cpu().detach().numpy()
        labelsShow=torch.cat(labelsShow,dim=0).cpu().detach().numpy()
        pot_result = searchThreshold(-scores, labels)
        precision = pot_result['pot-precision']
        recall = pot_result['pot-recall']
        F1 = pot_result['pot-f1']
        threshold = pot_result['pot-threshold']
        accuracy = torch.true_divide(pot_result['pot-TP'] + pot_result['pot-TN'],
                                     pot_result['pot-TP'] + pot_result['pot-TN']
                                     + pot_result['pot-FP'] + pot_result['pot-FN']).item()
        if parameters.needplot:
            """plt.figure(figsize=(10, 7))
            plt.subplot(7, 1, 1)
            plt.plot(reconSeq)
            plt.subplot(7,1,2)
            plt.plot(origiSeq)
            plt.subplot(7,1,3)
            if parameters.omit:
                avgSeq=(origiSeq*parameters.exeTime[1:]).sum(axis=-1)
            else:
                avgSeq=(origiSeq*parameters.exeTime).sum(axis=-1)
            plt.plot(avgSeq)
            plt.subplot(7,1,4)
            plt.plot(scores)
            plt.axhline(y=-threshold, color='grey', linestyle='--')
            plt.subplot(7,1,5)
            plt.plot(coherences)
            plt.subplot(7,1,6)
            plt.plot(divideScores)
            plt.subplot(7,1,7)
            plt.plot(labelsShow)
            plt.show()"""
            colors=["#14517C","#2F7fC1","#E7EFFA","#9673CD","#F3D266","#D8383A","#F7E1ED","#F8F3F9","#C497B2","#A9B8C6","#934B43","#9394E7","#D76364","#EF7A6D","#F1D77E","#B1CE46","#5F97D2","#9DC3E7"]
            colors=colors[:parameters.fealen]
            #colors = plt.cm.magma(np.linspace(0, 1, parameters.fealen))
            plt.figure(figsize=(10,9))
            plt.subplot(4,1,1)
            #plt.plot(reconSeq,color=colors)
            plotSeries(reconSeq,colors)
            plt.xlabel("Time slots")
            #plt.ylabel("Reconstruction series")
            plt.title("Reconstruction series")

            plt.subplot(4,1,2)
            #plt.plot(origiSeq,color=colors)
            plotSeries(origiSeq, colors)
            plt.title("Original series")
            plt.xlabel("Time slots")
            #plt.ylabel("Original series")

            plt.subplot(4,1,3)
            lbound=np.min(scores)
            ubound=np.max(scores)
            y_ceil=np.ones(scores.shape)*ubound
            y_bottom=np.ones(scores.shape)*lbound
            x=np.arange(0,len(scores))
            labelMask=labelsShow!=1.
            y_ceil[labelMask]=y_bottom[labelMask]
            plt.plot(scores,label="Anomaly score",color="#8E8BFE")
            plt.axhline(y=-threshold, color='grey', linestyle='--',label="Threshold")
            plt.fill_between(x,y_bottom,y_ceil,color="#FEA3E2",alpha=0.5,edgecolor="none",label="Anomaly")
            plt.legend()
            plt.title("Anomaly score")
            plt.xlabel("Time slots")
            #plt.ylabel("Anomaly score")

            plt.subplot(4,1,4)
            plt.plot(coherences[parameters.klen:],color="#8E8BFE",label="weight")
            lbound = np.min(coherences)
            ubound = np.max(coherences)
            y_ceil = np.ones(scores.shape) * ubound
            y_bottom = np.ones(scores.shape) * lbound
            y_ceil[labelMask] = y_bottom[labelMask]
            plt.fill_between(x, y_bottom, y_ceil, color="#FEA3E2", alpha=0.5, edgecolor="none", label="Anomaly")
            plt.title("Weight in loss function")
            plt.xlabel("Time slots")
            plt.legend()
            plt.tight_layout()
            plt.savefig('MCVisualize.pdf', bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(10, 9))
            for i in range(parameters.moduleNum):
                plt.subplot(parameters.moduleNum, 3, i*3+1)
                plotSeries(origiSeq,colors)
                plt.subplot(parameters.moduleNum,3,i*3+2)
                plotSeries(reconAggs[i], colors)
                plt.subplot(parameters.moduleNum,3,i*3+3)
                plotSeries(reconSigs[i],colors)
            plt.tight_layout()
            plt.savefig('LayerAtt.pdf', bbox_inches='tight')
            plt.show()


        print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")
        print("precision:%.6f, recall:%.6f, F1 score:%.6f, accuracy:%.6f\n" % (precision, recall, F1, accuracy))
        print("average score:%f"%np.mean(scores))
        return test_loss, precision, recall, F1, accuracy

def loadModel(path,parameters):
    model = MCNet(parameters)
    model.load_state_dict(torch.load(path))
    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def getSeed():
    seed=int(time.time()*1000)%(2**32-1)
    return seed


def detectAnomaly(trainData,valDatas,testDatas,testLabels,args,dataset):
    trainData=torch.tensor(trainData,dtype=torch.float)
    valDatas=torch.tensor(valDatas,dtype=torch.float)
    testDatas=torch.tensor(testDatas,dtype=torch.float)
    testLabels=torch.tensor(testLabels,dtype=torch.float)
    seed = 1282028438#getSeed()#
    setup_seed(seed)
    loadMod = args.loadMod != 0
    needTrain = args.needTrain != 0
    trainDataset=myDataset(trainData,args.winlen)
    valDataset=myDataset(valDatas,args.winlen,type="validate")
    testDataset=myDataset(testDatas,args.winlen,testLabels,"test")
    trainDataLoader=DataLoader(trainDataset,batch_size=args.batchSize,shuffle=True)
    valDataLoader=DataLoader(valDataset,batch_size=args.batchSize,shuffle=True)
    testDataLoader=DataLoader(testDataset,batch_size=args.batchSize,shuffle=False)
    dirName = "MCNet_"+str(dataset)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    modelPath = dirName + "/MCNet_"+str(seed)+".pth"
    if not loadMod:
        model=MCNet(args).to(args.device)
    else:
        model=loadModel(modelPath,args).to(args.device)
    loss_fn=lossFunc
    epochs = 3
    best_loss = 9999999999
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.1)
    trainStart=time.time()
    if needTrain:
        last_loss = 999999999
        count = 0
        torch.save(model.cpu().state_dict(), modelPath)
        model = model.to(args.device)
        print("Saved PyTorch Model State to " + modelPath)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(trainDataLoader,model,loss_fn,args,optimizer,t)
            test_loss=validate(valDataLoader,model,loss_fn,args)
            if math.isnan(test_loss):
                break
            if last_loss < test_loss:
                count += 1
            else:
                count = 0
            if count >= 2 or math.isnan(test_loss):
                break
            last_loss = test_loss
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.cpu().state_dict(), modelPath)
                model = model.to(args.device)
                print("Saved PyTorch Model State to " + modelPath)
    trainEnd=time.time()
    trainCost=trainEnd-trainStart
    print("trainCost:",trainCost)
    model = loadModel(modelPath, args).to(args.device)
    inferStart = time.time()
    test_loss, precision, recall, F1, accuracy = test(testDataLoader, model, loss_fn, args)
    inferEnd = time.time()
    with open(dirName + "/res" + str(seed) + ".csv", "w") as f:
        f.write("%f,%f,%f,%f\n" % (precision, recall, F1, accuracy))
    with open(dirName + "/config" + str(seed) + ".txt", "w") as f:
        f.write(str(args))
    return [precision, recall, F1, accuracy]