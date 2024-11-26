class TrainConfig:
    batch_size = 32
    use_fuse_model = False
    select_model = "all_fuse"
    use_metrics = False
    use_log = False
    model_path = None
    model_name = None
    lr = 3e-4
    use_margin_loss = False
    use_label_loss = False
    use_weight_loss = False
    use_threshold_loss = False
    margin_loss_type = "MarginLoss"
    epoch = 50
    opt_threshold = 0.1

class Args:
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    device = 'cuda:0'
    input_emb = 1063
    use_sample = True
    
class ArgsPara:
    mul_label_weight = 1.0
    ts_weight = 1.0
    pred_type = "pred_opt"
    
