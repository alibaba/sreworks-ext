import numpy as np
import math
import random

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
