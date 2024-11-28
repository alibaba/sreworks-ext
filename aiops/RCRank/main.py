from utils.load_data import load_dataset_valid as load_dataset_tensor_valid
from utils.load_data import load_dataset as load_dataset_tensor
from model.train_test import train as train_opt_all
from utils.config import Args, ArgsPara, TrainConfig

    
def train(config, train_dataloader, test_dataloader, valid_dataloader, train_len, test_len, valid_len, train_dataset, plan_args, para_args):
    train_opt_all(config.select_model, config.use_fuse_model, batch_size=config.batch_size, train_dataloader=train_dataloader, test_dataloader=test_dataloader, valid_dataloader=valid_dataloader,  train_len=train_len, test_len=test_len, valid_len=valid_len,  train_dataset=train_dataset, use_metrics=config.use_metrics, use_log=config.use_log, model_path_dir=config.model_path, model_name=config.model_name, lr=config.lr, use_margin_loss=config.use_margin_loss, use_label_loss=config.use_label_loss, use_weight_loss=config.use_weight_loss, use_threshold_loss=config.use_threshold_loss, margin_loss_type=config.margin_loss_type, epoch=config.epoch, opt_threshold=config.opt_threshold, margin_loss_margin=config.margin_loss_margin, plan_args=plan_args, para_args=para_args)

if __name__ == '__main__':  
    valid_dataloader = None
    valid_len = 0
    
    batch_size = 8
    device = "cuda:1"
    use_valid_dataset = True
    data_path = "./data/data.pickle"
    
    if use_valid_dataset:
        train_dataloader, test_dataloader, valid_dataloader, train_len, test_len, valid_len, train_dataset = load_dataset_tensor_valid(data_path, batch_size=batch_size, device=device)
    else:
        train_dataloader, test_dataloader, train_len, test_len, train_dataset = load_dataset_tensor(data_path, batch_size=batch_size, device=device)
    
    config = TrainConfig()
    config.batch_size = batch_size
    config.lr = 1e-4
    plan_args = Args()
    plan_args.device = device
    para_args = ArgsPara()
    threshold =  config.opt_threshold
    standred_threshold = (threshold  - train_dataset.opt_labels_train_mean) / (train_dataset.opt_labels_train_std + 1e-6)
    para_args.std_threshold = standred_threshold.to(plan_args.device)
    ts_weight = 7
    para_args.ts_weight = ts_weight
    margin_weight = ts_weight
    para_args.margin_weight = margin_weight
    
    model_name = "GateComDiffPretrainModel" 
    config.model_name = model_name
    config.use_fuse_model = True
    config.use_metrics = True
    config.use_log = True
    
    config.use_margin_loss = True
    config.use_threshold_loss = True
    
    para_args.ts_weight = ts_weight
    para_args.margin_weight = margin_weight
    
    
    eta = 0.07
    config.margin_loss_margin = eta
    config.model_path = f"res/{config.model_name} confidence eta{eta}/"
    train(config, train_dataloader, test_dataloader, valid_dataloader, train_len, test_len, valid_len, train_dataset, plan_args, para_args)
        
    