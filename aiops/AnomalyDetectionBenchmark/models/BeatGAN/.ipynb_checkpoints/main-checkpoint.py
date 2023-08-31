import os

#base_path = "../datasets/holo/"
#data_types = [base_path + 'filllinear', base_path + 'fillmean', base_path + 'fillzero']
parser.add_argument('--dataset', metavar='-d', type=str, required=False, default='MSL', help='dataset name')
parser.add_argument('--instance', metavar='-i', type=str, required=False, default='15', help='instance number')
parser.add_argument('--holo_datafolder', type=str, default='../../datasets/holo/fillzero_std',help='holo_datafolder')
parser.add_argument('--public_datafolder', type=str, default='../../datasets/public/',help='public_datafolder')
parser.add_argument('--holo_result_save_path', type=str, default='../../result/holo_result.csv',help='holo_result_save_path')
parser.add_argument('--public_result_save_path', type=str, default='../../result/public_result.csv',help='public_result_save_path')
config = parser.parse_args()

public_datafolder = config.public_datafolder
public_datasets = config.dataset
holo_datafolder = config.holo_datafolder
holo_datasets = "instance"+config.instance
holo_result_file = config.holo_result_save_path
pub_result_file = config.public_result_save_path

if config.dataset != 'holo':
    is_pub = True
    save_path = pub_result_file
    datafolder = public_datafolder
    dataset = public_datasets
    path = os.path.join(public_datafolder, public_datasets)
else:
    is_pub = False
    save_path = holo_result_file
    datafolder = holo_datafolder
    dataset = holo_datasets
    path = os.path.join(holo_datafolder, holo_datasets)

#data_paths = [os.path.join(public_datafolder, public_datasets), os.path.join(holo_datafolder, holo_datasets)]

os.system(f"bash train.sh {path}")
os.system(f"bash test.sh {path} {dataset} {save_path}")