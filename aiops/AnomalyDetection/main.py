#main for flourier10 SMD group 2
#epoch=1
from model.Model import detectAnomaly
from utils.utils import normaliseData,readMutshangData
import numpy as np
import argparse
import torch
import random



parser = argparse.ArgumentParser(description='[VAE]')
parser.add_argument('--fealen',type=int, required=False, default=17,help='feature length')
parser.add_argument('--winlen',type=int,required=False,default=4,help='window length')#4/40
parser.add_argument('--moduleNum',type=int,required=False,default=6,help="diff layers")#6/6
parser.add_argument('--loadMod',type=int,required=False,default=0,help='load model from file')
parser.add_argument('--device',type=str,required=False,default='cpu',help='train on which device')
parser.add_argument('--batchSize',type=int,required=False,default=100,help='batch size')
parser.add_argument('--needTrain',type=int,required=False,default=1,help='need train or just inference')
parser.add_argument('--lr',type=float,required=False,default=0.001,help='learning rate')
parser.add_argument('--patchSize',type=int,required=False,default=2,help='attention patch length')#2/3
parser.add_argument('--klen',type=int,required=False,default=2,help='convolution kernel length')#2/5
parser.add_argument('--lamda',type=float,required=False,default=0.01,help='loss function weight')
parser.add_argument("--sigma",type=float,required=False,default=0.8,help='the ratio between attRes and gateIncrease')
parser.add_argument("--slidewinlen",type=int,required=False,default=2,help='the sliding window average length')#3/2
parser.add_argument('--exeTime',type=list,required=False,default=[5,15,25,35,55,90,130,170,210,255,305,355,405,455])
parser.add_argument('--edges',type=list,required=False,default=[0,10,20,30,40,70,110,150,190,230,280,330,380,430])
parser.add_argument('--invalid',type=list,required=False,default=[31])
parser.add_argument('--omit',type=int,required=False,default=1,help='whether omit the first column of data')
parser.add_argument('--needplot',type=int,required=False,default=0)
args = parser.parse_args()

#mutshang data
args.edges=[0,5,10,20,30,40,70,110,150,190,230,280,330,380,430,900,1200,9000]
exeTime=[]
for i in range(len(args.edges)-1):
    exeTime.append((args.edges[i]+args.edges[i+1])/2)
exeTime[-1]=1230*60
args.exeTime=exeTime
args.omit=False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def divide(datas,ratio):
    leng=len(datas)
    len1=int(ratio*leng)
    return datas[:len1],datas[len1:]

def getSeed():
    return  random.randint(0,10000)

seed = getSeed()
setup_seed(seed)

datas,labels=readMutshangData("./datas/mustangData.npy","./datas/mustangLabel.npy")
print(datas.shape,labels.shape)
print(datas.shape,labels.shape)
dataset="MUTSTANG"

print(datas.shape)
results=[]

for i in range(len(datas)):
    if (datas[i]==0).all():
        continue
    datasetID=dataset+str(i)+"_"
    tr_val,test=divide(datas[i],0.8)
    _,testLabels=divide(labels[i],0.8)
    trains,valdatas=divide(tr_val,0.9)
    if (testLabels[args.winlen-2:]<=0).all() or (datas[i]==0.).all():
        continue

    trains=normaliseData(trains,args.omit)
    valdatas=normaliseData(valdatas,args.omit)
    testdatas=normaliseData(test,args.omit)
    result=detectAnomaly(trains,valdatas,testdatas,testLabels,args,datasetID)
    results.append([i]+result)
results=np.array(results)
avrg=results.mean(axis=0)
print("results:",results)
print()
print("average:",avrg)
np.savetxt("results.csv",results,fmt='%f',delimiter=',',newline='\n')
np.savetxt("average.csv",avrg, fmt='%f', delimiter=',')

