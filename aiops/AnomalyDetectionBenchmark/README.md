<a name="t3FFN"></a>
## Benchmarking Multivariate Time Series Anomaly Detection with Large-Scale Real-World Datasets
In this paper, we advance the benchmarking of time series anomaly detection from datasets, evaluation metrics, and algorithm comparison. <br />To the best of our knowledge, we have generated the largest real-world dataset for multivariate time series anomaly detection (MTSAD) from the Hologres AIOps system in the Alibaba Cloud platform. <br />We review and compare popular evaluation metrics including recently proposed.<br />To evaluate classic machine learning and recent deep learning methods fairly, we have performed extensive comparisons of these methods on various datasets. <br />We believe our benchmarking and datasets can promote reproducible results and accelerate the progress of MTSAD research.

<a name="JwX86"></a>
## 1、Datasets
<a name="T1rHA"></a>
### 1.1 Real-world Hologres AIOps Dataset
The Hologres Datasets are on [https://figshare.com/articles/dataset/_b_BigDataAD_Benchmark_Dataset_b_/24040563](https://figshare.com/articles/dataset/_b_BigDataAD_Benchmark_Dataset_b_/24040563) <br />For each instance, a CSV file is given. The train dataset and test dataset in each instance should be split evenly through all the timestamps.

- Illustration of the collection of Hologres AIOps dataset

![](img/system_structure.png)
All the metrics and labels in our dataset are derived from real-world scenarios. All metrics were obtained from the Hologres instance monitoring system and cover a rich variety of metric types, including **CPU usage, queries per second (QPS) and latency**, which are related to many important modules within Hologres. We obtain labels from the ticket system, which integrates three main sources of instance anomalies: user service requests, instance unavailability and fault simulations. User service requests refer to tickets that are submitted directly by users, whereas instance unavailability is typically detected through existing monitoring tools or discovered by Site Reliability Engineers (SREs). Since the system is usually very stable, we augment the anomaly samples by conducting fault simulations. Fault simulation refers to a special type of anomaly, planned beforehand, which is introduced to the system to test its performance under extreme conditions. All records in the ticket system are subject to follow-up processing by engineers, who meticulously mark the start and end times of each ticket. This rigorous approach ensures the accuracy of the labels in our dataset. 

- Statistic Characteristic of Hologres AIOps Datasets

| Instance | Samples | Dims | Anomaly | Anomaly Rate |
| --- | --- | --- | --- | --- |
| instance0 | 167950 | 21 | 2117 | 1.260% |
| instance1 | 167960 | 209 | 66 | 0.039% |
| instance2 | 167950 | 29 | 646 | 0.385% |
| instance3 | 167930 | 40 | 71 | 0.042% |
| instance4 | 167962 | 199 | 238 | 0.142% |
| instance5 | 167950 | 19 | 3 | 0.002% |
| instance6 | 167960 | 77 | 711 | 0.423% |
| instance7 | 167964 | 9 | 24 | 0.014% |
| instance8 | 167946 | 53 | 67 | 0.040% |
| instance9 | 167962 | 19 | 59 | 0.035% |
| instance10 | 167962 | 22 | 17 | 0.010% |
| instance11 | 167964 | 35 | 146 | 0.087% |
| instance12 | 167954 | 299 | 319 | 0.190% |
| instance13 | 167952 | 51 | 1783 | 1.062% |
| instance14 | 167958 | 44 | 1493 | 0.889% |
| instance15 | 167954 | 16 | 314 | 0.187% |
| instance16 | 167952 | 50 | 667 | 0.397% |
| instance17 | 167962 | 102 | 73 | 0.043% |
| instance18 | 167952 | 26 | 9613 | 5.724% |
| instance19 | 167956 | 27 | 27 | 0.016% |
| instance20 | 167948 | 98 | 94 | 0.056% |
| instance21 | 167946 | 51 | 2711 | 1.614% |
| instance22 | 167966 | 19 | 7 | 0.004% |
| instance23 | 167958 | 56 | 32 | 0.019% |
| instance24 | 167954 | 65 | 3452 | 2.055% |
| instance25 | 167952 | 27 | 3 | 0.002% |
| instance26 | 167946 | 332 | 7 | 0.004% |
| instance27 | 167948 | 38 | 125 | 0.074% |
| instance28 | 167958 | 68 | 570 | 0.339% |
| instance29 | 46194 | 45 | 99 | 0.214% |
| instance30 | 167942 | 35 | 1271 | 0.757% |
| instance31 | 167944 | 74 | 161 | 0.096% |
| instance32 | 167932 | 130 | 374 | 0.223% |
| instance33 | 167958 | 45 | 5 | 0.003% |
| instance34 | 167939 | 87 | 150 | 0.089% |
| instance35 | 167948 | 37 | 91 | 0.054% |
| instance36 | 167950 | 200 | 66 | 0.039% |
| instance37 | 167956 | 185 | 9 | 0.005% |
| instance38 | 167962 | 27 | 91 | 0.054% |
| instance39 | 167950 | 121 | 208 | 0.124% |
| instance40 | 167956 | 27 | 259 | 0.154% |
| instance41 | 167948 | 47 | 158 | 0.094% |
| instance42 | 167934 | 92 | 351 | 0.209% |
| instance43 | 167948 | 43 | 554 | 0.330% |
| instance44 | 167934 | 92 | 351 | 0.209% |
| instance45 | 167964 | 134 | 811 | 0.483% |
| instance46 | 167952 | 25 | 2 | 0.001% |
| instance47 | 167958 | 30 | 26 | 0.015% |

Due to Alibaba Internal Data Exposure Policy, we delete all data's timestamps and column names.<br />The last column of every instance file is the anomaly label, other columns are different system metrics.
<a name="aCTNS"></a>
### 1.2 Public Datasets
You can download the Public Datasets through the following URL: [https://drive.google.com/file/d/1MqJ-Qf20wm8MaweyyzGc3SB3JujTIiEd/view?usp=sharing](https://drive.google.com/file/d/1MqJ-Qf20wm8MaweyyzGc3SB3JujTIiEd/view?usp=sharing)
<a name="xA92t"></a>
## 2、Evaluation Metrics
The Evaluation Metrics we considered are:

- Accuracy
- Precision
- Recall
- F1-Score with Point Adjustment
- Composite F1-score
- Affiliation Score
- Volume Under the Surface (VUS) Metric

For more details, see the metrics files in：
```Bash
sreworks-ext/aiops/AnomalyDetectionBenchmark/main/evaluation
```

<a name="Om1HR"></a>
## 3、Models
The methods included are shown below:

- **Classics**
   - **Local Outlier Factor (LOF):** LOF measures the local deviation of the density of a given sample with respect to its neighbors.
   - **K-Nearest Neighbors (KNN):** KNN views the anomaly score of the input instance as the distance to its $k$-th nearest neighbor.
   - **Isolation Forest (IForest):** IForest isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
   - **Long short-term memory (LSTM): **LSTM is among the family of RNNs and LSTM \citet{hochreiter1997long} and can be effectively deployed in the TSAD problem, where the anomalies are detected by the deviation between the predicted and actual ones.
   - **LSTM-based autoencoder (LSTM-AE):** reconstructs input sequence and regards samples with high reconstruction errors as anomalies. 
   - **LSTM-based variational autoencoder(LSTM-VAE):** combines the power of both the LSTM-based model and VAE-based model, which learns to encode the input sequence into a lower-dimensional latent space representation and then decodes it back to reconstruct the original sequence. Similar to LSTM-AE, the reconstruction errors between the input sequence and the reconstructed ones are defined as anomaly scores.
   - **Deep Support Vector Data Description (DeepSVDD):** DeepSVDD trains a neural network while minimizing the volume of a hypersphere that encloses the network representations of the data, forcing the network to extract the common factors of variation.
   - **Copula-Based Outlier Detector (COPOD):** COPOD is a hyperparameter-free, highly interpretable anomaly detection algorithm based on empirical copula models.
   - **Empirical-Cumulative-distribution-based Outlier Detection (ECOD):** ECOD is a hyperparameter-free, highly interpretable anomaly detection algorithm based on empirical CDF functions. Basically, it uses ECDF to estimate the density of each feature independently and assumes that the anomaly locates the tails of the distribution.
- **AnomalyTransformer**
   - **Anomaly Transformer** is a representation of a series of explicit association modelling work which detects anomalies by association discrepancy between a learned Gaussian kernel and attention weight distribution. 
- **BeatGAN**
   - **Adversarially Generated Model (BeatGAN): **BeatGAN outputs explainable results to pinpoint the anomalous time ticks of an input beat, by comparing them to adversarially generated beats. 
- **DAGMM**
   - **Deep Autoencoding Gaussian Mixture Model (DAGMM):** DAGMM utilizes a deep autoencoder to generate a low-dimensional representation and reconstruction error for each input data point, which is further fed into a Gaussian Mixture Model (GMM).
- **DCdetector**
   - **DCdetector** is a dual attention contrastive representation learning framework whose motivation is similar to anomaly transformer but is concise as it does not contain a specially designed Gaussian Kernel or a MinMax learning strategy, nor a reconstruction loss. Contrastive representation learning helps to distinguish anomalies from normal points. 
- **USAD**
   - **UnSupervised Anomaly Detection (USAD): **USAD is based on adversely trained autoencoders to isolate anomalies while providing fast training.

For the methods we used, there are model file folders in the:
```Bash
sreworks-ext/aiops/AnomalyDetectionBenchmark/models
```
All the details are shown inside.
<a name="tVl84"></a>
## 4、Experiments Running（How to reproduce results through code and dataset）
![](img/img.png)
The framework of our benchmark is shown above. As the lack of data usually happens, we provide three different filling NaN methods and focus more on the performance of deep learning models with diverse evaluation metrics.

Following steps will show the process to run the experiments.
- Datasets Downloading
   - Download our datasets from  [https://figshare.com/articles/dataset/_b_BigDataAD_Benchmark_Dataset_b_/24040563/](https://figshare.com/articles/dataset/_b_BigDataAD_Benchmark_Dataset_b_/24040563/) , unzip the file and put it in:
```Bash
sreworks-ext/aiops/AnomalyDetectionBenchmark/datasets/holo
```

   - Download public datasets from [https://drive.google.com/file/d/1MqJ-Qf20wm8MaweyyzGc3SB3JujTIiEd/view?usp=sharing](https://drive.google.com/file/d/1MqJ-Qf20wm8MaweyyzGc3SB3JujTIiEd/view?usp=sharing)
and put them in:
```Bash
sreworks-ext/aiops/AnomalyDetectionBenchmark/datasets/public
```

- Data Preprocessing

We prepare three filling methods for data preprocessing: Mean, Linear Interpolation and Zero.<br />The data preprocessor file is in:
```Bash
sreworks-ext/aiops/AnomalyDetectionBenchmark/main/datafill_methods.ipynb
```
You can also use your own filling method to process the data.

- Model Running

To use the models listed above:

In `sreworks-ext/aiops/AnomalyDetectionBenchmark/main/`, run:
```Bash
python main.py --model <model_name> --dataset <public_dataset> --instance <holo_instance_num>
```
where `<model>` can be either of 'DCDetector', 'AnomalyTransformer', 'KNN', 'LOF', 'IForest', 'COPOD', 'ECOD', 'DeepSVDD', 'LSTM', 'LSTM_AE', 'LSTM_VAE', 'USAD', 'DAGMM', 'BeatGAN.

`<dataset>` denotes the public datasets 'PSM', 'MSL', 'SMD', 'NIPS_TS_Water', 'NIPS_TS_Syn_Mulvar', 'NIPS_TS_Swan', 'NIPS_TS_CCard', 'SMAP', and our proposed 'HOLO'.

`<instance>` indicates the instance number (0~47) of the sub-dataset in the HOLO dataset.

There are more parameters you can tune, see `sreworks-ext/aiops/AnomalyDetectionBenchmark/main/main.py`

We have already put one of the public datasets and a preprocessed instance of our datasets in the datasets folder. 
To see part of the results, run the command:
```Bash
python main.py --model DAGMM --dataset HOLO --instance 15
```

- Result Analysis

The following table  shows the experiment results on part of Hologres instances with filling zero for missing data, where the
abbreviations of the evaluation metrics are accuracy, precision, recall, F1-score, affiliation precision,
affiliation recall, Range_AUC_ROC, Range_AUC_PR, VUS_ROC, VUS_PR, AUC_PR, and AUC_ROC in order.

<table>
    <tr>
        <th>Instance</th>
        <th>Model</th>
        <th>Acc</th>
        <th>F1</th>
        <th>P</th>
        <th>R</th>
        <th>A-P</th>
        <th>A-R</th>
        <th>R_A_P</th>
        <th>R_A_R</th>
        <th>V_P</th>
        <th>V_R</th>
    </tr>
    <tr>
        <th rowspan="13">instance14</th>
        <td>USAD</td>
        <td>99.34</td>
        <td>81.77</td>
        <td>79.20</td>
        <td>84.52</td>
        <td>88.54</td>
        <td>46.40</td>
        <td>4.22</td>
        <td>43.42</td>
        <td>4.12</td>
        <td>39.82</td>
    </tr>
    <tr>
        <td>KNN</td>
        <td>1.75</td>
        <td>3.45</td>
        <td>1.75</td>
        <td>100.00</td>
        <td>50.26</td>
        <td>100.00</td>
        <td>6.46</td>
        <td>80.87</td>
        <td>6.29</td>
        <td>77.59</td>
    </tr>
    <tr>
        <td>LOF</td>
        <td>40.52</td>
        <td>5.57</td>
        <td>2.86</td>
        <td>100.00</td>
        <td>57.08</td>
        <td>100.00</td>
        <td>8.65</td>
        <td>86.39</td>
        <td>8.34</td>
        <td>84.76</td>
    </tr>
    <tr>
        <td>IForest</td>
        <td>98.23</td>
        <td>16.99</td>
        <td>48.10</td>
        <td>10.32</td>
        <td>50.21</td>
        <td>33.80</td>
        <td>14.54</td>
        <td>92.25</td>
        <td>14.41</td>
        <td>92.19</td>
    </tr>
    <tr>
        <td>COPOD</td>
        <td>98.24</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>47.66</td>
        <td>16.08</td>
        <td>20.40</td>
        <td>95.37</td>
        <td>20.24</td>
        <td>95.32</td>
    </tr>
    <tr>
        <td>ECOD</td>
        <td>99.70</td>
        <td>90.94</td>
        <td>98.42</td>
        <td>84.52</td>
        <td>89.27</td>
        <td>45.60</td>
        <td>12.24</td>
        <td>92.52</td>
        <td>12.15</td>
        <td>92.46</td>
    </tr>
    <tr>
        <td>DeepSVDD</td>
        <td>74.24</td>
        <td>11.55</td>
        <td>6.14</td>
        <td>95.86</td>
        <td>64.38</td>
        <td>99.64</td>
        <td>22.50</td>
        <td>91.24</td>
        <td>21.09</td>
        <td>89.41</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>99.34</td>
        <td>81.85</td>
        <td>79.35</td>
        <td>84.52</td>
        <td>88.87</td>
        <td>47.25</td>
        <td>4.25</td>
        <td>45.18</td>
        <td>4.17</td>
        <td>42.19</td>
    </tr>
    <tr>
        <td>LSTM-AE</td>
        <td>99.34</td>
        <td>81.72</td>
        <td>79.10</td>
        <td>84.52</td>
        <td>88.72</td>
        <td>47.22</td>
        <td>4.25</td>
        <td>44.76</td>
        <td>4.17</td>
        <td>42.03</td>
    </tr>
    <tr>
        <td>LSTM-VAE</td>
        <td>99.34</td>
        <td>81.72</td>
        <td>79.10</td>
        <td>84.52</td>
        <td>88.72</td>
        <td>47.22</td>
        <td>4.25</td>
        <td>44.76</td>
        <td>4.17</td>
        <td>42.03</td>
    </tr>
    <tr>
        <td>Anomaly-Transformer</td>
        <td>98.78</td>
        <td>74.13</td>
        <td>59.07</td>
        <td>99.52</td>
        <td>51.09</td>
        <td>99.58</td>
        <td>72.76</td>
        <td>92.42</td>
        <td>72.70</td>
        <td>92.31</td>
    </tr>
    <tr>
        <td>DCdetector</td>
        <td>98.93</td>
        <td>75.96</td>
        <td>63.42</td>
        <td>94.69</td>
        <td>50.58</td>
        <td>99.58</td>
        <td>74.26</td>
        <td>91.79</td>
        <td>71.02</td>
        <td>88.50</td>
    </tr>
    <tr>
        <td>BeatGAN</td>
        <td>99.31</td>
        <td>81.21</td>
        <td>78.15</td>
        <td>84.52</td>
        <td>88.73</td>
        <td>47.34</td>
        <td>4.25</td>
        <td>45.12</td>
        <td>4.16</td>
        <td>42.10</td>
    </tr>
    <tr>
        <th rowspan="13">instance15</th>
        <td>USAD</td>
        <td>4.17</td>
        <td>0.73</td>
        <td>0.37</td>
        <td>100.00</td>
        <td>50.23</td>
        <td>100.00</td>
        <td>5.75</td>
        <td>81.18</td>
        <td>5.65</td>
        <td>80.01</td>
    </tr>
    <tr>
        <td>KNN</td>
        <td>94.31</td>
        <td>5.04</td>
        <td>2.68</td>
        <td>43.05</td>
        <td>60.18</td>
        <td>88.90</td>
        <td>8.06</td>
        <td>74.41</td>
        <td>7.91</td>
        <td>70.03</td>
    </tr>
    <tr>
        <td>LOF</td>
        <td>0.35</td>
        <td>0.70</td>
        <td>0.35</td>
        <td>100.00</td>
        <td>50.33</td>
        <td>100.00</td>
        <td>3.75</td>
        <td>65.82</td>
        <td>3.79</td>
        <td>62.45</td>
    </tr>
    <tr>
        <td>IForest</td>
        <td>99.74</td>
        <td>70.31</td>
        <td>58.94</td>
        <td>87.12</td>
        <td>81.67</td>
        <td>96.03</td>
        <td>19.47</td>
        <td>95.48</td>
        <td>18.82</td>
        <td>95.30</td>
    </tr>
    <tr>
        <td>COPOD</td>
        <td>99.86</td>
        <td>75.47</td>
        <td>98.90</td>
        <td>61.02</td>
        <td>94.43</td>
        <td>52.48</td>
        <td>39.52</td>
        <td>96.77</td>
        <td>38.32</td>
        <td>97.12</td>
    </tr>
    <tr>
        <td>ECOD</td>
        <td>99.67</td>
        <td>13.29</td>
        <td>100.00</td>
        <td>7.12</td>
        <td>100.00</td>
        <td>22.22</td>
        <td>38.36</td>
        <td>98.02</td>
        <td>37.18</td>
        <td>98.07</td>
    </tr>
    <tr>
        <td>DeepSVDD</td>
        <td>99.62</td>
        <td>57.90</td>
        <td>47.60</td>
        <td>73.90</td>
        <td>75.54</td>
        <td>96.72</td>
        <td>25.10</td>
        <td>94.95</td>
        <td>22.57</td>
        <td>92.99</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>94.74</td>
        <td>4.54</td>
        <td>2.42</td>
        <td>35.59</td>
        <td>61.94</td>
        <td>75.81</td>
        <td>9.24</td>
        <td>82.52</td>
        <td>8.64</td>
        <td>81.50</td>
    </tr>
    <tr>
        <td>LSTM-AE</td>
        <td>94.17</td>
        <td>4.11</td>
        <td>2.18</td>
        <td>35.59</td>
        <td>71.06</td>
        <td>73.19</td>
        <td>6.39</td>
        <td>82.36</td>
        <td>6.35</td>
        <td>81.38</td>
    </tr>
    <tr>
        <td>LSTM-VAE</td>
        <td>94.17</td>
        <td>4.11</td>
        <td>2.18</td>
        <td>35.59</td>
        <td>71.06</td>
        <td>73.19</td>
        <td>6.39</td>
        <td>82.36</td>
        <td>6.35</td>
        <td>81.38</td>
    </tr>
    <tr>
        <td>Anomaly-Transformer</td>
        <td>97.76</td>
        <td>8.13</td>
        <td>4.75</td>
        <td>28.14</td>
        <td>48.61</td>
        <td>95.79</td>
        <td>10.43</td>
        <td>56.55</td>
        <td>9.79</td>
        <td>55.87</td>
    </tr>
    <tr>
        <td>DCdetector</td>
        <td>98.91</td>
        <td>34.46</td>
        <td>22.40</td>
        <td>74.58</td>
        <td>49.49</td>
        <td>97.25</td>
        <td>30.59</td>
        <td>68.28</td>
        <td>28.18</td>
        <td>65.85</td>
    </tr>
    <tr>
        <td>BeatGAN</td>
        <td>93.25</td>
        <td>8.54</td>
        <td>4.88</td>
        <td>89.49</td>
        <td>60.21</td>
        <td>88.41</td>
        <td>9.24</td>
        <td>82.45</td>
        <td>8.64</td>
        <td>81.45</td>
    </tr>
    <tr>
        <th rowspan="13">instance23</th>
        <td>USAD</td>
        <td>94.25</td>
        <td>0.78</td>
        <td>0.39</td>
        <td>59.38</td>
        <td>65.39</td>
        <td>59.17</td>
        <td>21.85</td>
        <td>76.42</td>
        <td>19.87</td>
        <td>68.02</td>
    </tr>
    <tr>
        <td>KNN</td>
        <td>50.36</td>
        <td>0.06</td>
        <td>0.03</td>
        <td>40.63</td>
        <td>64.11</td>
        <td>79.61</td>
        <td>2.01</td>
        <td>66.95</td>
        <td>1.66</td>
        <td>64.68</td>
    </tr>
    <tr>
        <td>LOF</td>
        <td>49.82</td>
        <td>0.06</td>
        <td>0.03</td>
        <td>40.63</td>
        <td>64.28</td>
        <td>79.61</td>
        <td>0.36</td>
        <td>59.40</td>
        <td>0.35</td>
        <td>58.93</td>
    </tr>
    <tr>
        <td>IForest</td>
        <td>99.90</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>86.04</td>
        <td>35.45</td>
        <td>0.71</td>
        <td>75.36</td>
        <td>0.69</td>
        <td>74.26</td>
    </tr>
    <tr>
        <td>COPOD</td>
        <td>99.97</td>
        <td>63.33</td>
        <td>67.86</td>
        <td>59.38</td>
        <td>99.85</td>
        <td>40.00</td>
        <td>34.65</td>
        <td>98.68</td>
        <td>29.05</td>
        <td>97.84</td>
    </tr>
    <tr>
        <td>ECOD</td>
        <td>99.97</td>
        <td>63.33</td>
        <td>67.86</td>
        <td>59.38</td>
        <td>99.85</td>
        <td>40.00</td>
        <td>23.71</td>
        <td>96.49</td>
        <td>21.72</td>
        <td>94.05</td>
    </tr>
    <tr>
        <td>DeepSVDD</td>
        <td>38.76</td>
        <td>0.12</td>
        <td>0.06</td>
        <td>100.00</td>
        <td>54.20</td>
        <td>100.00</td>
        <td>1.39</td>
        <td>84.83</td>
        <td>1.21</td>
        <td>82.34</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>97.15</td>
        <td>1.56</td>
        <td>0.79</td>
        <td>59.38</td>
        <td>72.08</td>
        <td>59.33</td>
        <td>22.05</td>
        <td>77.58</td>
        <td>20.50</td>
        <td>70.16</td>
    </tr>
    <tr>
        <td>LSTM-AE</td>
        <td>97.25</td>
        <td>1.62</td>
        <td>0.82</td>
        <td>59.38</td>
        <td>73.27</td>
        <td>59.35</td>
        <td>23.02</td>
        <td>77.88</td>
        <td>20.54</td>
        <td>70.74</td>
    </tr>
    <tr>
        <td>LSTM-VAE</td>
        <td>97.25</td>
        <td>1.62</td>
        <td>0.82</td>
        <td>59.38</td>
        <td>73.27</td>
        <td>59.35</td>
        <td>23.02</td>
        <td>77.88</td>
        <td>20.54</td>
        <td>70.74</td>
    </tr>
    <tr>
        <td>Anomaly-Transformer</td>
        <td>98.86</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>51.31</td>
        <td>97.85</td>
        <td>0.33</td>
        <td>49.62</td>
        <td>0.50</td>
        <td>49.77</td>
    </tr>
    <tr>
        <td>DCdetector</td>
        <td>98.96</td>
        <td>3.62</td>
        <td>1.88</td>
        <td>48.39</td>
        <td>49.71</td>
        <td>98.89</td>
        <td>7.03</td>
        <td>55.38</td>
        <td>7.31</td>
        <td>55.66</td>
    </tr>
    <tr>
        <td>BeatGAN</td>
        <td>95.05</td>
        <td>0.91</td>
        <td>0.46</td>
        <td>59.38</td>
        <td>65.77</td>
        <td>59.33</td>
        <td>22.10</td>
        <td>77.47</td>
        <td>20.54</td>
        <td>70.04</td>
    </tr>
    <tr>
        <th rowspan="13">instance38</th>
        <td>USAD</td>
        <td>52.48</td>
        <td>0.35</td>
        <td>0.18</td>
        <td>88.61</td>
        <td>56.43</td>
        <td>99.97</td>
        <td>10.99</td>
        <td>77.50</td>
        <td>9.28</td>
        <td>76.48</td>
    </tr>
    <tr>
        <td>KNN</td>
        <td>15.30</td>
        <td>0.22</td>
        <td>0.11</td>
        <td>100.00</td>
        <td>54.76</td>
        <td>100.00</td>
        <td>1.51</td>
        <td>72.75</td>
        <td>1.36</td>
        <td>71.29</td>
    </tr>
    <tr>
        <td>LOF</td>
        <td>0.09</td>
        <td>0.19</td>
        <td>0.09</td>
        <td>100.00</td>
        <td>50.89</td>
        <td>100.00</td>
        <td>1.39</td>
        <td>77.19</td>
        <td>1.35</td>
        <td>76.02</td>
    </tr>
    <tr>
        <td>IForest</td>
        <td>99.03</td>
        <td>13.01</td>
        <td>7.10</td>
        <td>77.22</td>
        <td>60.63</td>
        <td>82.78</td>
        <td>1.98</td>
        <td>89.64</td>
        <td>1.81</td>
        <td>88.07</td>
    </tr>
    <tr>
        <td>COPOD</td>
        <td>99.90</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>13.97</td>
        <td>2.00</td>
        <td>5.01</td>
        <td>93.77</td>
        <td>4.40</td>
        <td>93.12</td>
    </tr>
    <tr>
        <td>ECOD</td>
        <td>99.82</td>
        <td>3.82</td>
        <td>3.85</td>
        <td>3.80</td>
        <td>67.63</td>
        <td>87.30</td>
        <td>12.54</td>
        <td>91.75</td>
        <td>12.08</td>
        <td>89.82</td>
    </tr>
    <tr>
        <td>DeepSVDD</td>
        <td>18.51</td>
        <td>0.23</td>
        <td>0.12</td>
        <td>100.00</td>
        <td>51.77</td>
        <td>100.00</td>
        <td>1.94</td>
        <td>76.79</td>
        <td>1.92</td>
        <td>76.89</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>99.78</td>
        <td>5.13</td>
        <td>4.31</td>
        <td>6.33</td>
        <td>67.87</td>
        <td>93.95</td>
        <td>10.52</td>
        <td>80.59</td>
        <td>8.39</td>
        <td>79.76</td>
    </tr>
    <tr>
        <td>LSTM-AE</td>
        <td>99.55</td>
        <td>3.09</td>
        <td>1.94</td>
        <td>7.59</td>
        <td>59.37</td>
        <td>85.87</td>
        <td>13.12</td>
        <td>81.02</td>
        <td>10.77</td>
        <td>80.39</td>
    </tr>
    <tr>
        <td>LSTM-VAE</td>
        <td>99.55</td>
        <td>3.09</td>
        <td>1.94</td>
        <td>7.59</td>
        <td>59.37</td>
        <td>85.87</td>
        <td>13.12</td>
        <td>81.02</td>
        <td>10.77</td>
        <td>80.39</td>
    </tr>
    <tr>
        <td>Anomaly-Transformer</td>
        <td>98.76</td>
        <td>11.38</td>
        <td>6.10</td>
        <td>84.81</td>
        <td>48.90</td>
        <td>84.26</td>
        <td>15.24</td>
        <td>61.26</td>
        <td>13.23</td>
        <td>59.22</td>
    </tr>
    <tr>
        <td>DCdetector</td>
        <td>98.91</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>49.35</td>
        <td>83.86</td>
        <td>1.59</td>
        <td>50.54</td>
        <td>1.31</td>
        <td>50.30</td>
    </tr>
    <tr>
        <td>BeatGAN</td>
        <td>98.65</td>
        <td>10.73</td>
        <td>5.72</td>
        <td>86.08</td>
        <td>64.80</td>
        <td>98.15</td>
        <td>10.53</td>
        <td>80.57</td>
        <td>8.40</td>
        <td>79.73</td>
    </tr>
    <tr>
        <th rowspan="13">instance39</th>
        <td>USAD</td>
        <td>86.22</td>
        <td>0.10</td>
        <td>0.05</td>
        <td>5.36</td>
        <td>60.10</td>
        <td>64.56</td>
        <td>4.34</td>
        <td>68.91</td>
        <td>3.71</td>
        <td>61.92</td>
    </tr>
    <tr>
        <td>KNN</td>
        <td>35.76</td>
        <td>0.39</td>
        <td>0.20</td>
        <td>94.64</td>
        <td>56.66</td>
        <td>96.09</td>
        <td>2.02</td>
        <td>54.33</td>
        <td>2.06</td>
        <td>56.10</td>
    </tr>
    <tr>
        <td>LOF</td>
        <td>29.67</td>
        <td>0.19</td>
        <td>0.10</td>
        <td>50.89</td>
        <td>52.52</td>
        <td>98.73</td>
        <td>0.70</td>
        <td>45.15</td>
        <td>0.74</td>
        <td>44.95</td>
    </tr>
    <tr>
        <td>IForest</td>
        <td>99.87</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>95.53</td>
        <td>10.63</td>
        <td>3.05</td>
        <td>67.03</td>
        <td>2.99</td>
        <td>68.02</td>
    </tr>
    <tr>
        <td>COPOD</td>
        <td>99.87</td>
        <td>35.37</td>
        <td>55.77</td>
        <td>25.89</td>
        <td>78.04</td>
        <td>27.83</td>
        <td>7.13</td>
        <td>86.62</td>
        <td>6.72</td>
        <td>86.59</td>
    </tr>
    <tr>
        <td>ECOD</td>
        <td>99.88</td>
        <td>35.80</td>
        <td>58.00</td>
        <td>25.89</td>
        <td>78.09</td>
        <td>27.83</td>
        <td>5.97</td>
        <td>83.96</td>
        <td>5.62</td>
        <td>83.05</td>
    </tr>
    <tr>
        <td>DeepSVDD</td>
        <td>85.14</td>
        <td>1.13</td>
        <td>0.57</td>
        <td>63.39</td>
        <td>84.94</td>
        <td>77.32</td>
        <td>0.86</td>
        <td>65.64</td>
        <td>0.86</td>
        <td>63.95</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>99.76</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>64.06</td>
        <td>64.81</td>
        <td>5.23</td>
        <td>72.07</td>
        <td>4.60</td>
        <td>65.36</td>
    </tr>
    <tr>
        <td>LSTM-AE</td>
        <td>99.69</td>
        <td>18.41</td>
        <td>14.29</td>
        <td>25.89</td>
        <td>63.21</td>
        <td>59.21</td>
        <td>5.16</td>
        <td>72.35</td>
        <td>4.70</td>
        <td>66.07</td>
    </tr>
    <tr>
        <td>LSTM-VAE</td>
        <td>99.69</td>
        <td>18.41</td>
        <td>14.29</td>
        <td>25.89</td>
        <td>63.21</td>
        <td>59.21</td>
        <td>5.16</td>
        <td>72.35</td>
        <td>4.70</td>
        <td>66.07</td>
    </tr>
    <tr>
        <td>Anomaly-Transformer</td>
        <td>97.97</td>
        <td>5.85</td>
        <td>3.12</td>
        <td>47.32</td>
        <td>48.61</td>
        <td>96.71</td>
        <td>9.40</td>
        <td>56.24</td>
        <td>8.47</td>
        <td>55.33</td>
    </tr>
    <tr>
        <td>DCdetector</td>
        <td>98.93</td>
        <td>11.41</td>
        <td>6.49</td>
        <td>47.32</td>
        <td>51.01</td>
        <td>97.48</td>
        <td>10.89</td>
        <td>56.33</td>
        <td>10.51</td>
        <td>56.00</td>
    </tr>
    <tr>
        <td>BeatGAN</td>
        <td>99.66</td>
        <td>11.25</td>
        <td>8.65</td>
        <td>16.07</td>
        <td>62.48</td>
        <td>70.65</td>
        <td>5.23</td>
        <td>72.05</td>
        <td>4.60</td>
        <td>65.31</td>
    </tr>
    <tr>
        <th rowspan="13">instance44</th>
        <td>USAD</td>
        <td>79.68</td>
        <td>0.15</td>
        <td>0.08</td>
        <td>1.02</td>
        <td>54.56</td>
        <td>83.33</td>
        <td>2.42</td>
        <td>68.04</td>
        <td>2.41</td>
        <td>67.76</td>
    </tr>
    <tr>
        <td>KNN</td>
        <td>98.26</td>
        <td>63.20</td>
        <td>46.42</td>
        <td>98.98</td>
        <td>78.92</td>
        <td>26.68</td>
        <td>62.86</td>
        <td>98.27</td>
        <td>69.13</td>
        <td>98.19</td>
    </tr>
    <tr>
        <td>LOF</td>
        <td>98.25</td>
        <td>63.07</td>
        <td>46.28</td>
        <td>98.98</td>
        <td>78.74</td>
        <td>26.68</td>
        <td>29.08</td>
        <td>97.17</td>
        <td>27.89</td>
        <td>96.55</td>
    </tr>
    <tr>
        <td>IForest</td>
        <td>91.71</td>
        <td>26.55</td>
        <td>15.33</td>
        <td>98.98</td>
        <td>76.90</td>
        <td>32.68</td>
        <td>1.20</td>
        <td>31.97</td>
        <td>1.22</td>
        <td>32.26</td>
    </tr>
    <tr>
        <td>COPOD</td>
        <td>99.90</td>
        <td>96.92</td>
        <td>94.94</td>
        <td>98.98</td>
        <td>93.88</td>
        <td>33.27</td>
        <td>13.64</td>
        <td>94.82</td>
        <td>13.67</td>
        <td>94.71</td>
    </tr>
    <tr>
        <td>ECOD</td>
        <td>99.93</td>
        <td>97.71</td>
        <td>96.47</td>
        <td>98.98</td>
        <td>93.69</td>
        <td>33.27</td>
        <td>11.47</td>
        <td>93.80</td>
        <td>11.42</td>
        <td>93.72</td>
    </tr>
    <tr>
        <td>DeepSVDD</td>
        <td>99.97</td>
        <td>98.98</td>
        <td>98.98</td>
        <td>98.98</td>
        <td>67.92</td>
        <td>33.10</td>
        <td>35.06</td>
        <td>50.71</td>
        <td>43.77</td>
        <td>61.55</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>93.90</td>
        <td>0.27</td>
        <td>0.18</td>
        <td>0.55</td>
        <td>62.75</td>
        <td>67.99</td>
        <td>2.42</td>
        <td>68.16</td>
        <td>2.41</td>
        <td>67.87</td>
    </tr>
    <tr>
        <td>LSTM-AE</td>
        <td>86.68</td>
        <td>0.20</td>
        <td>0.11</td>
        <td>0.87</td>
        <td>65.18</td>
        <td>99.18</td>
        <td>2.44</td>
        <td>68.26</td>
        <td>2.42</td>
        <td>67.96</td>
    </tr>
    <tr>
        <td>LSTM-VAE</td>
        <td>86.68</td>
        <td>0.20</td>
        <td>0.11</td>
        <td>0.87</td>
        <td>65.18</td>
        <td>99.18</td>
        <td>2.44</td>
        <td>68.26</td>
        <td>2.42</td>
        <td>67.96</td>
    </tr>
    <tr>
        <td>Anomaly-Transformer</td>
        <td>98.95</td>
        <td>74.04</td>
        <td>59.14</td>
        <td>98.98</td>
        <td>46.19</td>
        <td>48.06</td>
        <td>74.78</td>
        <td>94.55</td>
        <td>71.01</td>
        <td>90.72</td>
    </tr>
    <tr>
        <td>DCdetector</td>
        <td>99.00</td>
        <td>76.61</td>
        <td>62.49</td>
        <td>98.98</td>
        <td>59.10</td>
        <td>56.36</td>
        <td>76.58</td>
        <td>94.65</td>
        <td>70.38</td>
        <td>88.36</td>
    </tr>
    <tr>
        <td>BeatGAN</td>
        <td>91.60</td>
        <td>0.20</td>
        <td>0.12</td>
        <td>0.55</td>
        <td>52.22</td>
        <td>69.76</td>
        <td>2.41</td>
        <td>68.10</td>
        <td>2.40</td>
        <td>67.81</td>
    </tr>
</table>

*We also evaluate other methods for filling in missing data with mean interpolation and linear interpolation. 
For more results details and analysis, see our paper **"Benchmarking Multivariate Time Series Anomaly Detection with Large-Scale Real-World Datasets"**.

### License
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

See the License file for more details.
