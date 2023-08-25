import os

base_path = "../datasets/holo/"
data_types = [base_path + 'filllinear', base_path + 'fillmean', base_path + 'fillzero']

for data_type in data_types:
    datas = os.listdir(data_type)
    for data in datas:
        path = os.path.join(data_type, data)
        os.system(f"bash train.sh {path}")
        os.system(f"bash test.sh {path} {data}")