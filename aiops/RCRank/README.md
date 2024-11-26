# RCRank

This is the official PyTorch code for RCRank.

We propose RCRank, the first method to utilize a multimodal approach for identifying and ranking the root causes of slow queries by estimating their impact. We employ a pre-training method to align the multimodal information of queries, enhancing the performance and training speed of root cause impact estimated. Based on the aligned pre-trained embedding module, we use Cross-modal fusion of feature modalities to ultimately estimate the impact of root causes, identifying and ranking the root causes of slow queries.

## Installation
First clone the repository using Git.

Next, some data can be downloaded from this [link](https://drive.google.com/file/d/1L26JZDH6TJdleJkGaPbWjNSQm9E3CuO2/view?usp=drive_link). Please place the data files into the `data` folder. 

Please download the Bert model to the root directory through the [link](https://huggingface.co/google-bert/bert-base-uncased/tree/main)

The project dependencies can be installed by executing the following commands in the root of the repository:
```bash
conda env create --name RCRank python=3.9
conda activate RCRank
pip install -r requirements.txt
```


## Run

After downloading the pre-trained parameters, you can only execute training and inference using the following script.
```bash
python main.py
```

The final results will be saved in the `res` directory.

## Detailed guide

#### Pre-train
If pre-training is required, please download the pre-training data from this [link](https://drive.google.com/file/d/1L26JZDH6TJdleJkGaPbWjNSQm9E3CuO2/view?usp=drive_link), place it in the `data` folder, and execute the following script. The checkpoint will be saved in the `./pretrain/alignment_new` directory. 
```bash
python model/pretrain/pretrain.py
```

We will release the pre-trained parameters as soon as possible.

#### Config
Configuration File: config.py - Training Parameters

- **batch_size**: Number of samples per batch during training.
- **lr**: Learning rate.
- **epoch**: Number of training iterations.
- **device**: Device for computation, either GPU or CPU.
- **opt_threshold**: Threshold for valid root causes.
- **model_name**: Name of the model.
- **use_fuse_model**: Whether to use a fused model.
- **use_threshold_loss**: Whether to use validity and orderliness loss.
- **model_path**: Path to save the model.
- **margin_loss_type**: Margin loss type: "MarginLoss"、"ListnetLoss"、"ListMleLoss".
- **use_margin_loss**: Whether to use margin loss.
- **use_label_loss**: Whether to predict root cause type.
- **use_weight_loss**: Whether to assign sample weights.
- **use_log**: Whether to use log information.
- **use_metrics**: Whether to use metrics information.
- **embed_size**: "emb_size" in QueryFormer.
- **embed_size**: "emb_size" in QueryFormer.
- **pred_hid**: "pred_hid" in QueryFormer.
- **ffn_dim**: "ffn_dim" in QueryFormer.
- **head_size**: "head_size" in QueryFormer.
- **n_layers**: "n_layers" in QueryFormer.
- **dropout**: "dropout" in QueryFormer.
- **input_emb**: "dropout" in QueryFormer.
- **use_sample**: "use_sample" in QueryFormer.
- **ts_weight**: Weight for validity and orderliness.
- **mul_label_weight**: Weight for type of root cause.
- **pred_type**: Type of task.


