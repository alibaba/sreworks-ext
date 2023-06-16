Introduction of Benchmark

*Methods included are shown below:

	**Local Outlier Factor (LOF)** LOF measures the local deviation of the density of a given sample with respect to its neighbors.

	**K-Nearest Neighbors (KNN)** KNN views the anomaly score of the input instance as the distance to its $k$-th nearest neighbor.

	**Isolation Forest (IForest)** IForest isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

	**Long short-term memory (LSTM)** LSTM is among the family of RNNs and LSTM \citet{hochreiter1997long} and can be effectively deployed in the TSAD problem, where the anomalies are detected by the deviation between the predicted and actual ones.

	**LSTM based autoencoder (LSTM-AE)** reconstructs input sequence and regards samples with high reconstruction errors as anomalies. 

	**Deep Support Vector Data Description (DeepSVDD)** DeepSVDD trains a neural network while minimizing the volume of a hypersphere that encloses the network representations of the data, forcing the network to extract the common factors of variation.

	**Deep Autoencoding Gaussian Mixture Model (DAGMM)** DAGMM utilizes a deep autoencoder to generate a low-dimensional representation and reconstruction error for each input data point, which is further fed into a Gaussian Mixture Model (GMM).

	**LSTM based variational autoencoder(LSTM-VAE)** combines the power of both the LSTM-based model and VAE-based model, which learns to encode the input sequence into a lower-dimensional latent space representation and then decodes it back to reconstruct the original sequence. Similar to LSTM-AE, the reconstruction errors between the input sequence and the reconstructed ones are defined as anomaly scores.

	**Adversarially Generated Model (BeatGAN)**. BeatGAN outputs explainable results to pinpoint the anomalous time ticks of an input beat, by comparing them to adversarially generated beats. 

	**Copula Based Outlier Detector (COPOD)**. COPOD is a hyperparameter-free, highly interpretable anomaly detection algorithm based on empirical copula models.

	**UnSupervised Anomaly Detection (USAD)**. USAD is based on adversely trained autoencoders to isolate anomalies while providing fast training.

	**Anomaly-Transformer** Anomaly Transformer is a representation of a series of explicit association modelling work which detects anomalies by association discrepancy between a learned Gaussian kernel and attention weight distribution. 

	**Empirical-Cumulative-distribution-based Outlier Detection (ECOD)**. ECOD is a hyperparameter-free, highly interpretable anomaly detection algorithm based on empirical CDF functions. Basically, it uses ECDF to estimate the density of each feature independently, and assumes that the anomaly locates the tails of the distribution.

	**DCdetector** DCdetector is a dual attention contrastive representation learning framework whose motivation is similar to anomaly transformer but is concise as it does not contain a specially designed Gaussian Kernel or a MinMax learning strategy, nor a reconstruction loss. Contrastive representation learning help to distinguish anomalies from normal points. 


*The metrics we considered are in metrics files in each method file. For example, Anomaly-Transformer_HOLO/metrics.

*The scripts for each dataset are also included.

*For results details, see our paper "Benchmarking Multivariate Time Series Anomaly Detection with Large-Scale Real-World Datasets".

*The HOLOgres Datasets are on https://sreworks.oss-cn-beijing.aliyuncs.com/aiops/BenchmarkDataFinal.zip . For each instance, a csv file is given. The train dataset and test dataset in each instance should be split evenly through all the timestamps.
