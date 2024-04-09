import numpy as np
import pickle
import argparse
import os
from scipy.signal import butter, lfilter, freqz
import timesynth as ts


def shade_state_state_data(state_subj, t, ax, data='simulation'):
    cmap = plt.get_cmap("tab10")
    # Shade the state on simulation data plots
    for ttt in range(t[0], len(t)):
        if state_subj[ttt] == 0:
            ax.axvspan(ttt + 1, ttt, facecolor='blue', alpha=0.3)
        elif state_subj[ttt] == 1:
            ax.axvspan(ttt + 1, ttt, facecolor='green', alpha=0.3)
        elif state_subj[ttt] == 2:
            ax.axvspan(ttt + 1, ttt, facecolor='orange', alpha=0.3)

np.random.seed(42)

SIG_NUM = 3
STATE_NUM = 3
P_S0 = [1 / 3]

imp_feature = [[0], [1], [2]]
correlated_feature = {0: {0: [1]}, 1: {1: [2]}, 2: {0: [1, 2]}}
scale = {0: [0.8, -0.5, -0.2],
         1: [0, -1.0, 0],
         2: [-0.2, -0.2, 0.8]}

transition_matrix = [[0.95, 0.02, 0.03],
                     [0.02, 0.95, 0.03],
                     [0.03, 0.02, 0.95]]


def init_distribution_params():
    # Covariance matrix is constant across states but distribution means change based on the state value
    state_count = STATE_NUM
    cov = np.eye(SIG_NUM) * 0.1
    covariance = []
    for i in range(state_count):
        c = cov.copy()
        # for j in correlated_feature[i].keys():
        #     c[j,correlated_feature[i][j]] = 0.01
        #     c[correlated_feature[i][j], j] = 0.01
        # c = c + np.eye(SIG_NUM)*1e-3
        # print(c)
        covariance.append(c)
    covariance = np.array(covariance)
    mean = []
    for i in range(state_count):
        m = scale[i]
        mean.append(m)
    mean = np.array(mean)
    return mean, covariance


def next_state(previous_state, t):
    p_vec = transition_matrix[previous_state]
    # if previous_state == 0:
    #     params = 0.9
    # elif previous_state==1:
    #     params = 0.9
    # else:
    #     params = 0.9
    #
    # params = params - float(t / 500) if params > 0.7 else params
    # p_vec = np.zeros(STATE_NUM)
    # p_vec[previous_state] = params
    # p_vec[np.setdiff1d([0,1,2], previous_state)] = (1-params)/2.
    next_st = np.random.choice([0, 1, 2], p=p_vec)
    return next_st


def state_decoder(previous, next_st):
    return int(next_st * (1 - previous) + (1 - next_st) * previous)


def generate_linear_labels(X):
    logit = np.exp(-3 * np.sum(X, axis=1))

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_XOR_labels(X):
    y = np.exp(X[:, 0] * X[:, 1])

    prob_1 = np.expand_dims(1 / (1 + y), 1)
    prob_0 = np.expand_dims(y / (1 + y), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:, :4] ** 2, axis=1) - 1.5)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_additive_labels(X):
    logit = np.exp(-10 * np.sin(-0.2 * X[:, 0]) + 0.5 * X[:, 1] + X[:, 2] + np.exp(X[:, 3]) - 0.8)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_linear_labels_v2(X):
    logit = np.exp(-0.2 * X[:, 0] + 0.5 * X[:, 1] + X[:, 2] + X[:, 3] - 0.8)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def create_signal(sig_len, gp_params, mean, cov):
    signal = None
    state_local = []
    y = []
    importance = []
    y_logits = []

    previous = np.random.binomial(1, P_S0)[0]
    previous_label = None
    delta_state = 1

    # Sample for "previous" state (this is current state now)
    imp_sig = np.zeros(SIG_NUM)
    imp_sig[imp_feature[previous]] = 1
    importance.append(imp_sig)
    state_local.append(previous)

    for ii in range(1, sig_len):
        next_st = next_state(previous, delta_state)
        state_n = next_st

        imp_sig = np.zeros(SIG_NUM)
        if previous != state_n:
            # this samples labels+samples until  current point - before state change at next time point
            gp_vec = [ts.signals.GaussianProcess(lengthscale=g, mean=m, variance=0.1) for g, m in
                      zip(gp_params, mean[previous])]
            sample_ii = np.array([gp.sample_vectorized(time_vector=np.array(range(delta_state))) for gp in gp_vec])
            # print(sample_ii.shape)
            # sample_ii = np.random.multivariate_normal(mean[state_n], cov[state_n])
            if signal is not None:
                signal = np.hstack((signal, sample_ii))
            else:
                signal = sample_ii

            # signal.extend(sample_ii)
            # sample_ii = (sample_ii).reshape((1, -1))

            # y_probs = state_n * generate_linear_labels(sample_ii[:, imp_feature[state_n]]) + \
            #          (1 - state_n) * generate_linear_labels(sample_ii[:, imp_feature[state_n]])

            y_probs = generate_linear_labels(sample_ii.T[:, imp_feature[previous]])

            y_logit = [yy[1] for yy in y_probs]
            y_label = [np.random.binomial(1, yy) for yy in y_logit]

            # y_logit_past = y_logit
            y.extend(y_label)
            y_logits.extend(y_logit)
            delta_state = 1
            imp_sig[imp_feature[state_n]] = 1
            imp_sig[-1] = 1
        else:
            delta_state += 1
        importance.append(imp_sig)

        # previous_label = y_label
        state_local.append(state_n)
        previous = state_n

    # sample points in the last state-change
    gp_vec = [ts.signals.GaussianProcess(lengthscale=g, mean=m, variance=0.1) for g, m in
              zip(gp_params, mean[previous])]
    sample_ii = np.array([gp.sample_vectorized(time_vector=np.array(range(delta_state))) for gp in gp_vec])

    # sometimes only one state is ever sampled
    if signal is not None:
        signal = np.hstack((signal, sample_ii))
    else:
        signal = sample_ii

    y_probs = generate_linear_labels(sample_ii.T[:, imp_feature[previous]])

    y_logit = [yy[1] for yy in y_probs]
    y_label = [np.random.binomial(1, yy) for yy in y_logit]

    y.extend(y_label)
    y_logits.extend(y_logit)

    # signal = signal
    y = np.array(y)
    importance = np.array(importance)

    # print(signal.shape, y.shape, len(state_local), importance.shape, len(y_logits))
    return signal, y, state_local, importance, y_logits


def decay(x):
    return [0.9 * (1 - 0.1) ** x, 0.9 * (1 - 0.1) ** x]


def logit(x):
    return 1. / (1 + np.exp(-2 * (x)))


def normalize(train_data, test_data, config='mean_normalized'):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_size = train_data.shape[1]
    len_of_stay = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)
    if config == 'mean_normalized':
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide='ignore', invalid='ignore')
        train_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in train_data])
        test_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in test_data])
    elif config == 'zero_to_one':
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])
    return train_data_n, test_data_n


def create_dataset(count, signal_len):
    dataset = []
    labels = []
    importance_score = []
    states = []
    label_logits = []
    mean, cov = init_distribution_params()
    gp_lengthscale = np.random.uniform(0.2, 0.2, SIG_NUM)
    for num in range(count):
        sig, y, state, importance, y_logits = create_signal(signal_len, gp_params=gp_lengthscale, mean=mean, cov=cov)
        dataset.append(sig)
        labels.append(y)
        importance_score.append(importance.T)
        states.append(state)
        label_logits.append(y_logits)

        if num % 50 == 0:
            print(num, count)
    dataset = np.array(dataset)
    labels = np.array(labels)
    importance_score = np.array(importance_score).transpose(0,2,1)
    states = np.array(states)
    label_logits = np.array(label_logits)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    # train_data_n, test_data_n = normalize(train_data, test_data)
    train_data_n = train_data.transpose(0,2,1)
    test_data_n = test_data.transpose(0,2,1)
    if not os.path.exists('simulated_data_l2x'):
        os.mkdir('simulated_data_l2x')
    with open('simulated_data_l2x/state_dataset_x_train.pkl', 'wb') as f:
        pickle.dump(train_data_n, f)
    with open('simulated_data_l2x/state_dataset_x_test.pkl', 'wb') as f:
        pickle.dump(test_data_n, f)
    with open('simulated_data_l2x/state_dataset_y_train.pkl', 'wb') as f:
        pickle.dump(labels[:n_train], f)
    with open('simulated_data_l2x/state_dataset_y_test.pkl', 'wb') as f:
        pickle.dump(labels[n_train:], f)
    with open('simulated_data_l2x/state_dataset_importance_train.pkl', 'wb') as f:
        pickle.dump(importance_score[:n_train], f)
    with open('simulated_data_l2x/state_dataset_importance_test.pkl', 'wb') as f:
        pickle.dump(importance_score[n_train:], f)
    with open('simulated_data_l2x/state_dataset_logits_train.pkl', 'wb') as f:
        pickle.dump(label_logits[:n_train], f)
    with open('simulated_data_l2x/state_dataset_logits_test.pkl', 'wb') as f:
        pickle.dump(label_logits[n_train:], f)
    with open('simulated_data_l2x/state_dataset_states_train.pkl', 'wb') as f:
        pickle.dump(states[:n_train], f)
    with open('simulated_data_l2x/state_dataset_states_test.pkl', 'wb') as f:
        pickle.dump(states[n_train:], f)

    print(train_data_n.shape)
    print(labels.shape)
    print(importance_score.shape)
    print(label_logits.shape)
    print(states.shape)

    return dataset, labels, states, label_logits


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_len', type=int, default=100, help='Length of the signal to generate')
    parser.add_argument('--signal_num', type=int, default=1000, help='Number of the signals to generate')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    np.random.seed(234)
    dataset, labels, states, label_logits = create_dataset(args.signal_num, args.signal_len)

    if args.plot:
        import matplotlib.pyplot as plt

        f, (x1, x2, x3) = plt.subplots(3, 1)
        state_1_1 = []
        state_1_0 = []
        state_0_1 = []
        state_0_0 = []
        state_2_0 = []
        state_2_1 = []

        idx0 = np.where(states == 0)
        idx1 = np.where(states == 1)
        idx2 = np.where(states == 2)

        for c in range(len(idx0[0])):
            if labels[idx0[0][c], idx0[1][c]] == 0:
                state_0_0.append(labels[idx0[0][c], idx0[1][c]])
            else:
                state_0_1.append(labels[idx0[0][c], idx0[1][c]])

        for c in range(len(idx1[0])):
            if labels[idx1[0][c], idx1[1][c]] == 0:
                state_1_0.append(labels[idx1[0][c], idx1[1][c]])
            else:
                state_1_1.append(labels[idx1[0][c], idx1[1][c]])

        for c in range(len(idx2[0])):
            if labels[idx2[0][c], idx2[1][c]] == 0:
                state_2_0.append(labels[idx2[0][c], idx2[1][c]])
            else:
                state_2_1.append(labels[idx2[0][c], idx2[1][c]])

        x1.hist(state_0_0, label='label 0')
        x1.hist(state_0_1, label='label 1')
        x1.set_title('state 0')
        x1.legend()

        x2.hist(state_1_0, label='label 0')
        x2.hist(state_1_1, label='label 1')
        x2.set_title('state 1')
        x2.legend()

        x3.hist(state_2_0, label='label 0')
        x3.hist(state_2_1, label='label 1')
        x3.set_title('state 2')
        x3.legend()

        plt.savefig('plot.pdf')

        f, (x1, x2) = plt.subplots(2, 1)
        for id in range(len(labels)):
            for i, sample in enumerate(dataset[id]):
                if labels[id, i]:
                    x1.scatter(sample[0], sample[1], c='r')
                else:
                    x1.scatter(sample[0], sample[1], c='b')
                if states[id, i]:
                    x2.scatter(sample[0], sample[1], c='b')
                else:
                    x2.scatter(sample[0], sample[1], c='r')
            x1.set_title('Distribution based on label')
            x2.set_title('Distribution based on state')
        plt.savefig('plot2.pdf')

        plot_id = 5

        f = plt.figure(figsize=(18, 9))
        x1 = f.subplots()
        shade_state_state_data(states[plot_id], range(dataset.shape[2]), x1)
        for i in range(SIG_NUM):
            x1.plot(range(dataset.shape[2]), dataset[plot_id, i, :], linewidth=1, label='feature %d' % (i))
        # x1.plot(range(dataset.shape[2]), dataset[plot_id, 1, :], linewidth=3, label='feature %d' % (1))
        # x1.plot(range(dataset.shape[2]), dataset[plot_id, 2, :], linewidth=3, label='feature %d' % (2))
        x1.plot(range(dataset.shape[2]), label_logits[plot_id, :], linewidth=3, label='label')
        plt.legend()
        plt.savefig('plotsample_l2x.pdf')