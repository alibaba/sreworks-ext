import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
def readMutshangData(datapath,labelpath):
    datas=np.load(datapath)[:14000]
    labels=np.load(labelpath)[:14000,0]
    print("label shape",labels.shape)
    datas=datas.reshape(35,400,-1)
    labels=labels.reshape(35,400)
    np.save("mutshangSplitData.npy",datas)
    np.save("mutshangSplitLabel.npy",labels)
    return datas,labels
def readPublicData(path,path2=""):
    datas = np.genfromtxt(path, delimiter=',')
    datas/=datas.sum(axis=-1,keepdims=True)
    datalen,fealen=datas.shape
    datas=datas.reshape(2,int(datalen/2),fealen)
    print(datas.shape)
    return datas

def listds(start,end):
    dss=[]
    pres=start
    count=start%100
    while pres<=end:
        dss.append(pres)
        if count<24:
            count+=1
            pres+=1
        else:
            count=0
            pres=pres//100
            day=pres%100
            if day+1<=31:#cross day
                pres+=1
                pres*=100
            else:
                pres=pres//100
                month=pres%100
                if month+1<=12:#cross month
                    pres+=1
                    pres*=100
                    pres+=1
                    pres*=100
                else:#cross year
                    pres=pres//100
                    pres+=1#year+1
                    pres*=100
                    pres+=1#month=1
                    pres*=100
                    pres+=1#day=1
                    pres*=100#time=0
    return dss

def completeData(datas,rangeSeq,dss):
    ndatas=np.zeros((len(dss),len(rangeSeq)))
    for i,time in enumerate(dss):
        for j,ranges in enumerate(rangeSeq):
            result = datas[:, 5] == time
            result = datas[np.where(result)]
            result2 = result[:, 3] == ranges
            result2 = result[np.where(result2)]
            if len(result2)==0:
                ndatas[i][j]=0.
            else:
                ndatas[i][j]=result2[0,2]
    return ndatas

def readData(path1,path2=""):
    datas = pd.read_csv(path1)
    datas = datas.sort_values('ds')
    datas = datas.groupby('cluster')
    rangeSeq = ['[0-10)', '[10-20)', '[20-30)', '[30-40)'
        , '[40-70)', '[70,110)', '[110,150)', '[150,190)'
        , '[190,230)', '[230-280)', '[280-330)', '[330-380)', '[380-430)','[430-*)']
    dss = listds(2023121600, 2023122710)
    groups = []
    for data in datas:
        completed = completeData(data[1].to_numpy(), rangeSeq, dss)
        groups.append(completed)
        print(groups[-1].shape)
    groups = np.stack(groups)
    if path2!="":
        labels=np.load(path2)
    else:
        labels=np.zeros((groups.shape[0],groups.shape[1]))
    return groups,labels


def readData2(path1,path2=""):
    datas = pd.read_csv(path1)
    datas = datas.sort_values('ds')
    datas = datas.groupby('cluster')
    rangeSeq = ['[0-10)', '[10-20)', '[20-30)', '[30-40)'
        , '[40-70)', '[70,110)', '[110,150)', '[150,190)'
        , '[190,230)', '[230-280)', '[280-330)', '[330-380)', '[380-430)','[430-*)']
    dss = listds(2023121600, 2024010210)
    groups = []
    for data in datas:
        completed = completeData(data[1].to_numpy(), rangeSeq, dss)
        groups.append(completed)
        print(groups[-1].shape)
    groups = np.stack(groups)
    if path2!="":
        labels=np.load(path2)
    else:
        labels=np.zeros((groups.shape[0],groups.shape[1]))
    return groups,labels

def insertNormality(datas,quickRatio,quickNumRatio,quickTimeRange,labels,exeTime,edges):
    exeTime=exeTime[1:]
    edges=edges[:-1]
    #exeTime = np.array([5, 15, 25, 35, 55, 90, 130, 170, 210, 255, 305, 355, 405])
    #edges = [0,10, 20, 30, 40, 70, 110, 150, 190, 230, 280, 330, 380]
    for i, data in enumerate(datas):
        length, interval = data.shape
        changeNum = math.ceil(length * quickRatio)
        positions = [random.randint(0, length - 1) for i in range(changeNum)]
        labels[i, positions] = -1.
        for position in positions:
            quickNum = np.random.random(interval - 1)
            quickNum[0] = 0
            quickNum /= quickNum.sum()
            quickNum *= quickNumRatio
            quickTime = np.random.randint(quickTimeRange[0], quickTimeRange[1], size=interval - 1)
            presTime = np.maximum(exeTime - quickTime,0)
            slowDur = random.randint(2, 5)
            for k in range(min(slowDur, length - position)):
                labels[i, position + k] = -1.
                for j in range(interval - 1):
                    transfer = min(datas[i, position + k, j], quickNum[j])
                    datas[i, position + k, j] = max(datas[i, position + k, j] - quickNum[j], 0)
                    target = len(edges)-1
                    for edge in reversed(edges):
                        if presTime[j] >= edge:
                            break
                        target -= 1
                    datas[i, position + k, target] += transfer
    return datas,labels

def insertAnomaly(datas,anomalyRatio,slowNumRatio,slowTimeRange,labels,exeTime,edges):
    #labels=np.zeros((datas.shape[0],datas.shape[1]))
    exeTime=exeTime[:-1]
    edges=edges[1:]
    #exeTime=np.array([5,15,25,35,55,90,130,170,210,255,305,355,405])
    #edges=[10,20,30,40,70,110,150,190,230,280,330,380,430]
    for i,data in enumerate(datas):
        length,interval=data.shape
        anomalyNum=math.ceil(length*anomalyRatio)
        positions=[random.randint(0,length-1) for i in range(anomalyNum)]
        for position in positions:
            slowNum=np.random.random(interval-1)
            slowNum[-1]=0
            slowNum/=slowNum.sum()
            slowNum*=slowNumRatio
            slowTime=np.random.randint(slowTimeRange[0],slowTimeRange[1],size=interval-1)
            presTime=exeTime+slowTime
            slowDur = random.randint(2, 5)
            for k in range(min(slowDur,length-position)):
                flag = True
                for j in range(interval - 1):
                    transfer=min(datas[i,position+k,j],slowNum[j])
                    datas[i,position + k, j] = max(datas[i,position + k, j] - slowNum[j], 0)
                    target = 0
                    if transfer>0:
                        flag=False
                    for edge in edges:
                        if presTime[j] < edge:
                            break
                        target += 1
                    datas[i,position + k, target] += transfer
                if not flag:
                    labels[i,position+k] = 1.
    return datas,labels

def insertAnomaly2(datas,anomalyRatio,slowNumRatio,slowTimeRange,exeTime,edges):
    labels=np.zeros(datas.shape[0])
    exeTime = exeTime[:-1]
    edges=edges[1:]
    #exeTime=np.array([5,15,25,35,55,90,130,170,210,255,305,355,405])
    #edges=[10,20,30,40,70,110,150,190,230,280,330,380,430]
    length,interval=datas.shape
    anomalyNum=math.ceil(length*anomalyRatio)
    positions=[random.randint(0,length-1) for i in range(anomalyNum)]
    labels[positions]=1.
    for position in positions:
        slowNum=np.random.random(interval-1)
        slowNum[-1]=0
        slowNum/=slowNum.sum()
        slowNum*=slowNumRatio
        slowTime=np.random.randint(slowTimeRange[0],slowTimeRange[1],size=interval-1)
        presTime=exeTime+slowTime
        slowDur=random.randint(2,8)
        for k in range(min(slowDur,length-position)):
            labels[position+k]=1.
            for j in range(interval-1):
                datas[position+k,j]=max(datas[position+k,j]-slowNum[j],0)
                target=0
                for edge in edges:
                    if presTime[j]<edge:
                        break
                    target+=1
                datas[position+k,target]+=slowNum[j]
    return datas,labels


def normaliseData(datas,omit=True):
    if omit:
        ndata=datas[:,1:]
    else:
        ndata=datas[:,:]
    ndata=np.array(ndata,dtype=np.float64)
    ndataSum=ndata.sum(axis=-1)+0.0001
    ndata=ndata.transpose(1,0)
    ndata/=ndataSum
    ndata=ndata.transpose(1,0)
    return ndata


def generateSin(start,end,num,amplitude,frequency,phase,biase):
    time = np.linspace(start, end, num)
    sin_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)+biase
    return sin_wave

def generateSin2(start,end,num,amplitude,frequencys,phase,biase):
    num*=10
    sin_waves=[]
    step = 100
    time = np.linspace(start, end, num)
    for i in range(0,num,step):
        frequency=np.random.uniform(frequencys[1],frequencys[2])
        ntime=np.linspace(time[i],time[i+step-1],math.ceil(step*frequencys[0]/frequency))
        sin_wave = amplitude * np.sin(2 * np.pi * frequencys[0] * ntime + phase)+biase
        sin_waves.append(sin_wave)
    sin_waves=np.concatenate(sin_waves,axis=0)
    return sin_waves


def syntheticData2(clusterNum,length,fealen,noise=False):
    datas = []
    componsitionNum=2
    for i in range(clusterNum):
        waves = []
        for j in range(fealen):
            frequency = random.randint(6, 10)  # 1/24+random.gauss(0,0.005)#random.randint(6,10)
            amplitude = random.random() * 10
            bias = random.randint(1, 10)
            phase = random.gauss(0, 1) * 2 * math.pi
            wave = generateSin(0, length, length, amplitude, random.uniform(1/100,1/200), phase, bias) + amplitude
            for k in range(componsitionNum):
                wave+=generateSin(0,length,length,amplitude*0.2,1/24,phase*random.random(),bias)
            # frequency=1/40+random.gauss(0,0.005)#random.randint(6,10)
            # wave2=generateSin(0,length,length,amplitude,frequency,phase,bias)+amplitude
            # wave=wave1+wave2
            if noise:
                wave += np.random.normal(0, amplitude * 0.03, length)
            waves.append(wave)
        datas.append(waves)
    datas = np.array(datas)  # batch,fealen,winlen
    dataSum = datas.sum(axis=-2)  # batch,winlen
    datas = datas.transpose((1, 0, 2))  # fealen,batch,winlen
    datas /= dataSum
    datas = datas.transpose((1, 2, 0))
    labels = np.zeros((datas.shape[0], datas.shape[1]))
    return datas, labels

def syntheticData(clusterNum,length,fealen,noise=False,std=0.06):
    datas=[]
    for i in range(clusterNum):
        waves=[]
        for j in range(fealen):
            frequency=random.randint(6,10)#1/24+random.gauss(0,0.005)#random.randint(6,10)
            amplitude=random.random()*10
            bias=random.randint(1,10)
            phase=random.gauss(0,1)*2*math.pi
            wave=generateSin(0,length,length,amplitude,frequency,phase,bias)+amplitude
            #frequency=1/40+random.gauss(0,0.005)#random.randint(6,10)
            #wave2=generateSin(0,length,length,amplitude,frequency,phase,bias)+amplitude
            #wave=wave1+wave2
            if noise:
                wave+=np.random.normal(0,amplitude*std,length)
            waves.append(wave)
        datas.append(waves)
    datas=np.array(datas)#batch,fealen,winlen
    dataSum = datas.sum(axis=-2)#batch,winlen
    datas = datas.transpose((1, 0, 2))  # fealen,batch,winlen
    datas/=dataSum
    datas=datas.transpose((1,2,0))
    labels=np.zeros((datas.shape[0],datas.shape[1]))
    return datas,labels


def syntheticData3(clusterNum,length,fealen,noise=False,frequencyRatio=0.3):
    datas=[]
    for i in range(clusterNum):
        waves=[]
        for j in range(fealen):
            frequency=random.uniform(0.04,0.1)#1/24+random.gauss(0,0.005)#random.randint(6,10)
            frequencys=[frequency,frequency,frequency*(1+frequencyRatio)]
            amplitude=random.random()*10
            bias=random.randint(1,10)
            phase=random.gauss(0,1)*2*math.pi
            wave=generateSin2(0,length,length,amplitude,frequencys,phase,bias)+amplitude
            wave=wave[:length]
            frequency = random.uniform(0.08, 0.1)  # 1/24+random.gauss(0,0.005)#random.randint(6,10)
            frequencys = [frequency, frequency, frequency * (1 + frequencyRatio)]
            amplitude = random.random() * 10
            wave+=(generateSin2(0,length,length,amplitude,frequencys,phase,bias)+amplitude)[:length]
            frequency = random.uniform(0.01, 0.04)  # 1/24+random.gauss(0,0.005)#random.randint(6,10)
            frequencys = [frequency, frequency, frequency * (1 + frequencyRatio)]
            amplitude = random.random() * 10
            wave+=(generateSin2(0,length,length,amplitude,frequencys,phase,bias)+amplitude)[:length]
            #wave=generateSin(0,length,length,amplitude,frequency,phase,bias)+amplitude
            #frequency=1/40+random.gauss(0,0.005)#random.randint(6,10)
            #wave2=generateSin(0,length,length,amplitude,frequency,phase,bias)+amplitude
            #wave=wave1+wave2
            if noise:
                wave+=np.random.normal(0,amplitude*0.06,length)
            waves.append(wave)
        datas.append(waves)
    datas=np.array(datas)#batch,fealen,winlen
    dataSum = datas.sum(axis=-2)#batch,winlen
    datas = datas.transpose((1, 0, 2))  # fealen,batch,winlen
    datas/=dataSum
    datas=datas.transpose((1,2,0))
    labels=np.zeros((datas.shape[0],datas.shape[1]))
    return datas,labels
