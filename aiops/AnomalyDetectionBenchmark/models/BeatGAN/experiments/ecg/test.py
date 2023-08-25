import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import os
import csv
from options import Options
import pdb
from model import BeatGAN
import numpy as np
from  torch.utils.data import DataLoader,TensorDataset
from data import load_data, load_data2
from spot import SPOT
from metrics.combine_all_scores import combine_all_evaluation_scores
# from dcgan import DCGAN as myModel


device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")

def load_case(path, opt):
    test_samples = np.load(path)
    opt.nc = test_samples.shape[1]
    samples = []
    for i in range(0, test_samples.shape[0], 320):
        if i + 320 >= test_samples.shape[0]:
            break
        samples.append(test_samples[i : i + 320, :])

    samples = np.array(samples)
    test_samples = samples
    test_samples = np.transpose(test_samples,(0,2,1))
    test_y = np.zeros([test_samples.shape[0], 1])
    test_dataset = TensorDataset(torch.Tensor(test_samples), torch.Tensor(test_y))

    return DataLoader(dataset=test_dataset,  # torch TensorDataset format
                      batch_size=opt.batchsize,
                      shuffle=False,
                      num_workers=0,
                      drop_last=False)


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):

    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

opt = Options().parse()
train_path = os.path.join(opt.dataroot, 'train.py')
test_path = os.path.join(opt.dataroot, 'test.py')
label_path = os.path.join(opt.dataroot, 'test_label.npy')
normal_dataloader = load_case(train_path, opt)
abnormal_dataloader = load_case(test_path, opt)
model=BeatGAN(opt,None,device)

model_base_path = 'output/beatgan/ecg/model/beatgan_{0}_G.pkl'
model.G.load_state_dict(torch.load(model_base_path.format(opt.dataname),map_location=device))

model.G.eval()
with torch.no_grad():
    lossT, loss = [], []
    for i, data in enumerate(normal_dataloader):
        test_x=data[0]
        fake_x, _ = model.G(test_x)
        batch_input = test_x.cpu().numpy()
        batch_output = fake_x.cpu().numpy()
        batch_heat = (batch_input - batch_output)**2
        batch_heat = np.transpose(batch_heat,(0,2,1))
        score = np.sum(batch_heat, axis=2)
        score = score.flatten()
        lossT += list(score)
        
    for i, data in enumerate(abnormal_dataloader):
        test_x=data[0]
        fake_x, _ = model.G(test_x)
        batch_input = test_x.cpu().numpy()
        batch_output = fake_x.cpu().numpy()
        batch_heat = (batch_input - batch_output)**2
        batch_heat = np.transpose(batch_heat,(0,2,1))
        score = np.sum(batch_heat, axis=2)
        score = score.flatten()
        loss += list(score)

    
    labels = np.load(label_path)[:len(loss)].flatten()
    lms = 0.9999

    s = SPOT(1e-3)  # SPOT object
    s.fit(lossT, loss)  # data import
    s.initialize(level=0.999, min_extrema=False, verbose=False)  # initialization step
    
    ret = s.run(dynamic=False)  # run
    pot_th = np.mean(ret['thresholds']) * 1.0
    print(pot_th)
    pred, p_latency = adjust_predicts(loss, labels, pot_th, calc_latency=True)
    
    loss = np.array(loss)
    scores = combine_all_evaluation_scores(labels, pred, loss, opt.dataname)
    print(scores)
    with open('result.csv','a',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=scores.keys())
        writer.writerow(scores)
