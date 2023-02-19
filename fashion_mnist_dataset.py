import os
import gzip
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset
import torch

def load_data(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def processFashionMnist(path, ones_class, zeros_class):

    """
    Processes fashion mnist dataset to be used for binary classification
    """

    label_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}


    train_data, train_labels = load_data(path, kind='train')
    test_data, test_labels = load_data(path, kind='t10k')


    #Standarize the feature of the data
    scaler = preprocessing.StandardScaler().fit(train_data)
    standard_train_data = scaler.transform(train_data)
    standard_test_data = scaler.transform(test_data)

    # Process training data to binary
    b_train_labels = train_labels[(train_labels==ones_class) | (train_labels==zeros_class)]
    indices = np.where(np.in1d(train_labels, b_train_labels))[0]
    b_train_data = train_data[indices]
    standard_train_data = standard_train_data[indices]

    assert b_train_data.shape[0] == b_train_labels.size and standard_train_data.shape[0] == b_train_labels.size

    # Process test data to binary
    b_test_labels = test_labels[(test_labels==ones_class) | (test_labels==zeros_class)]
    indices = np.where(np.in1d(test_labels, b_test_labels))[0]
    b_test_data = test_data[indices]
    standard_test_data = test_data[indices]

    assert b_test_data.shape[0] == b_test_labels.size and standard_test_data.shape[0] == b_test_labels.size

    # Change training labels to 0 or 1
    train_labels = np.zeros(b_train_labels.size)
    train_labels[b_train_labels == ones_class] = 1

    # Chnage test labels to 0 or 1
    test_labels = np.zeros(b_test_labels.size)
    test_labels[b_test_labels == ones_class] = 1

    data = {}

    train_size = int(train_labels.size / 2)

    data['simulation_data'] = (b_train_data[:train_size, :], train_labels[:train_size])
    data['training_data'] = (b_train_data[train_size:, :], train_labels[train_size:])
    data['test_data'] = (b_test_data, test_labels)
    data['standard_simulation_data'] = (standard_train_data[:train_size, :], train_labels[:train_size])
    data['standard_training_data'] = (standard_train_data[train_size:, :], train_labels[train_size:])
    data['standard_test_data'] = (standard_test_data, test_labels)

    return data


def create_weak_signal_view(path, ones_class, zeros_class):

    data = processFashionMnist(path, ones_class, zeros_class)

    train_data, train_labels = data['standard_training_data']
    sim_data, sim_labels = data['standard_simulation_data']
    test_data, test_labels = data['standard_test_data']

    weak_signal_train_data = []
    weak_signal_sim_data = []
    weak_signal_test_data = []

    #for fashion mnist dataset, select the 1/4 feature, middle feature and the 3/4 feature as weak signals
    views = {0:195, 1:391, 2:587}

    for i in range(3):
        #pick a random feature for the individual weak signals
        f = views[i]

        weak_signal_train_data.append(train_data[:, f:f+1])
        weak_signal_sim_data.append(sim_data[:, f:f+1])
        weak_signal_test_data.append(test_data[:, f:f+1])


    weak_signal_data = [weak_signal_train_data, weak_signal_sim_data, weak_signal_test_data]

    return data, weak_signal_data


def train_weak_signals(data, weak_signal_data, num_weak_signal):
    """
    Trains different views of weak signals
    """

    train_data, train_labels = data['standard_training_data']
    sim_data, sim_labels = data['standard_simulation_data']
    test_data, test_labels = data['standard_test_data']


    weak_signal_train_data = weak_signal_data[0]
    weak_signal_sim_data = weak_signal_data[1]
    weak_signal_test_data = weak_signal_data[2]

    models = []
    stats = np.zeros((num_weak_signal, 2))
    w_sig_prob_train = []
    w_sig_test_accuracies = []
    weak_train_accuracies = []

    w_sig_prob_test = []

    w_sig_prob_sim = []


    for i in range(num_weak_signal):
        # fit model
        model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        model.fit(weak_signal_sim_data[i], sim_labels)
        models.append(model)

        # evaluate probability of P(X=1)
        probability = model.predict_proba(weak_signal_train_data[i])
        score_0 = train_labels * probability[:,0] + (1 - train_labels) * (1 - probability[:,0])
        score_1 = train_labels * (1 - probability[:,1]) + (1 - train_labels) * probability[:,1]
        stats[i,0] = np.sum(score_0) / score_0.size
        stats[i,1] = np.sum(score_1) / score_1.size
        w_sig_prob_train.append(probability)

        # evaluate accuracy for train data
        weak_train_accuracies.append(accuracy_score(train_labels, np.round(probability[:,1])))

        # evaluate accuracy for test data
        test_predictions = model.predict(weak_signal_test_data[i])
        w_sig_test_accuracies.append(accuracy_score(test_labels, test_predictions))

        w_sig_prob_test.append(model.predict_proba(weak_signal_test_data[i]))

        w_sig_prob_sim.append(model.predict_proba(weak_signal_sim_data[i]))


    model = {}
    model['models'] = models
    model['train_prob'] = np.array(w_sig_prob_train)
    model['error_bounds'] = stats
    model['train_accuracy'] = weak_train_accuracies
    model['test_accuracy'] = w_sig_test_accuracies

    model['test_prob'] = np.array(w_sig_prob_test)

    model['sim_prob'] = np.array(w_sig_prob_sim)


    print(weak_train_accuracies)
    print(w_sig_test_accuracies)

    return model


def load_dataset(path, num_weak_signals, classes):

    classes = classes.split(",")

    zero_class = int(classes[0])
    one_class = int(classes[1])

    data, weak_signal_data = create_weak_signal_view(path, one_class, zero_class)

    w_model = train_weak_signals(data, weak_signal_data, num_weak_signals)


    training_data, training_labels = data['standard_training_data']
    sim_data, sim_labels = data['standard_simulation_data']
    test_data, test_labels = data['standard_test_data']

    num_features, num_data_points = training_data.shape


    models = w_model['models']

    train_weak_signal_ub = w_model['error_bounds']
    train_weak_signal_probabilities = w_model['train_prob']

    test_weak_signal_ub = w_model['error_bounds']
    test_weak_signal_probabilities = w_model['test_prob']


    sim_weak_signal_ub = w_model['error_bounds']
    sim_weak_signal_probabilities = w_model['sim_prob']


    return {"training_data":training_data, "training_labels":training_labels,
            "training_ub": train_weak_signal_ub, "training_weak_signal_probs":np.swapaxes(train_weak_signal_probabilities,0,1),

            "sim_data": sim_data, "sim_labels": sim_labels,
            "sim_ub": sim_weak_signal_ub, "sim_weak_signal_probs": np.swapaxes(sim_weak_signal_probabilities,0,1),


            "test_data":test_data, "test_labels":test_labels,
            "test_ub":test_weak_signal_ub, "test_weak_signal_probs":np.swapaxes(test_weak_signal_probabilities,0,1),

            "weak_models":models, "num_features":num_features
            }



class FashionMNISTDataset(Dataset):

    def __init__(self, dataset, labels, num_classes, weak_signal_ub, weak_signal_probs):


        '''

        :param dataset:  K x N matrix, each column is a K-dimensional data point
        :param labels: 1 x N matrix, each entry is 0 or 1
        :param weak_signal_ub: constant
        :param weak_signal_probs: num_signal x N matrix, each entry is [0,1]
        '''


        super().__init__()

        self.num_signals = len(weak_signal_probs)
        self.num_datapoint = dataset.shape[0]
        self.feature_dim = dataset.shape[1]

        self.dataset = dataset / 255.0
        self.labels = labels
        self.num_classes = num_classes
        self.weak_signal_ub = weak_signal_ub
        self.weak_signal_probs = weak_signal_probs



    def __len__(self):
        return self.num_datapoint


    def __getitem__(self, index):

        x = self.dataset[index,:]
        l = self.labels[index]
        ws = self.weak_signal_probs[index,:]
        ub = self.weak_signal_ub


        x = torch.from_numpy(x)
        y = torch.zeros(self.num_classes)
        y[int(l)] = 1.0
        ws = torch.from_numpy(ws)
        ub = torch.from_numpy(ub)

        x = x.float()
        y = y.float()
        ws = ws.float()
        ub = ub.float()


        '''
        x: B x n_dim
        y: B x num_classes one-hot vector
        ws: B x num_weak_signal
        ub: B x num_weak_signal
        
        '''

        return {"x":x, "y":y, "ws":ws, "ub":ub}

