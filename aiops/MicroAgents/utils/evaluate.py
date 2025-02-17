import os
from utils.json import parse_json_markdown
import json
from utils.load import load_data, save_data
import functools

def read_results(file_path, dataset):
    res = []
    file_list = os.listdir(file_path)
    for file in file_list:
        if file.endswith('.txt') and dataset in file:
            with open(os.path.join(file_path, file), 'r') as f:
                lines = "".join(list(f.readlines()))
                lines = lines.replace('Likelihood', 'Possibility')
                lines = lines.replace('Probability', 'Possibility')
                lines = lines.replace('"Type"', '"Abnormal Type"')
                lines = lines.replace('"Abnormal_Type"', '"Abnormal Type"')
                lines = lines.replace('AbnormalType', 'Abnormal Type')
                lines = lines.replace('Possiblity', 'Possibility')
                try:
                    cur_res = parse_json_markdown(lines)
                    if 'CAUSES' in cur_res:
                        cur_res['CAUSE'] = cur_res['CAUSES']
                    if 'CAUSE' not in cur_res:
                        print('cause not provided,', os.path.join(file_path, file))
                   
                    labels = file.split('_', 3)
                    system_name = labels[0]
                    number = labels[1]
                    podname = labels[2]
                    inject_type = labels[3].split('.')[0]
                    
                    cur_res = {
                        'system_name': system_name,
                        'number':number,
                        'podname':podname,
                        'inject_type':inject_type,
                        'res':cur_res
                    }
                    res.append(cur_res)
                except Exception as e:
#                     res = json.loads(lines)
                    # print(lines)
                    print(e)
                    print(file)
                    res.append(None)
    return res

def comp(x,y):
    if isinstance(x['Possibility'], list):
        x['Possibility'] = x['Possibility'][0]
    if isinstance(y['Possibility'], list):
        y['Possibility'] = y['Possibility'][0]
    if  x['Possibility'].lower() == y['Possibility'].lower():
        return 0
    elif 'high' in x['Possibility'].lower():
        return 1
    elif 'high' in y['Possibility'].lower():
        return -1
    elif 'medium' in x['Possibility'].lower():
        return 1
    elif 'medium' in y['Possibility'].lower():
        return -1
    else:
        return 0


def evaluate(file_path, dataset, k):
    res = read_results(file_path, dataset)

    
    correct_pod_num = 0
    correct_type_num = 0
    pod_gt = []
    type_gt = []
    pod_pred = []
    type_pred = []

    all_num = 0
    
    for cur_res in res:

        if cur_res is None:
            all_num+=1
            continue
        # print(cur_res['number'], cur_res['res'].keys())
        if 'CAUSE' not in cur_res['res']:
            all_num+=1

            print(cur_res)

            continue
        try:
            cur_res['res']['CAUSE'].sort(key=functools.cmp_to_key(comp), reverse=True)
        except:
            print('key error', file_path, cur_res['res']['CAUSE'])
        preds = cur_res['res']['CAUSE'][0:k]
        podname_gt = cur_res['podname']
        inject_type_gt = cur_res['inject_type']
        number = cur_res['number']

        if dataset == 'TrainTicket' and inject_type_gt == 'exception':
            continue
        
        all_num += 1
        pod_gt.append(podname_gt)
        type_gt.append(inject_type_gt)
        
        flag = False
        for pred in preds:
            # if 'Abnormal Type' not in pred:
            #     print(number, podname_gt, inject_type_gt, pred)
            if podname_gt in pred['Module']:
                correct_pod_num += 1
                pod_pred.append(pred['Module'])

                if '/' in pred['Abnormal Type']:
                    pred['Abnormal Type'] = pred['Abnormal Type'].split('/')[0]
                if isinstance(pred['Abnormal Type'], list):
                    pred['Abnormal Type'] = pred['Abnormal Type'][0]
                if 'network' in inject_type_gt.lower() and 'network' in pred['Abnormal Type'].lower():
                    correct_type_num += 1
                    type_pred.append(pred['Abnormal Type'])
                    flag = True
                    break
                elif 'cpu' in inject_type_gt.lower() and 'cpu' in pred['Abnormal Type'].lower():
                    correct_type_num += 1
                    type_pred.append(pred['Abnormal Type'])
                    flag = True
                    break
                elif 'exception' in inject_type_gt.lower() and 'code' in pred['Abnormal Type'].lower():
                    correct_type_num += 1
                    type_pred.append(pred['Abnormal Type'])
                    flag = True
                    break
                elif 'return' in inject_type_gt.lower() and 'code' in pred['Abnormal Type'].lower():
                    correct_type_num += 1
                    type_pred.append(pred['Abnormal Type'])
                    flag = True
                    break
                elif 'code' in inject_type_gt.lower() and 'code' in pred['Abnormal Type'].lower():
                    correct_type_num += 1
                    type_pred.append(pred['Abnormal Type'])
                    flag = True
                    break
                else:
                    type_pred.append(None)
                flag = True
                break
        if not flag:
            print(number, podname_gt, inject_type_gt)
       
    print('correct_pod_num', correct_pod_num, 'all_num', all_num)
    
    return correct_pod_num/all_num, correct_type_num/all_num
    
    


