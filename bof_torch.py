import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class BoF_Pooling(nn.Module):
    def __init__(self, n_codewords, features, spatial_level=0, **kwargs):
        super(BoF_Pooling, self).__init__()
        """
        Initializes a BoF Pooling layer
        :param n_codewords: the number of the codewords to be used
        :param spatial_level: 0 -> no spatial pooling, 1 -> spatial pooling at level 1 (4 regions). Note that the
         codebook is shared between the different spatial regions
        :param kwargs:
        """

        self.N_k = n_codewords
        self.spatial_level = spatial_level
        self.V, self.sigmas = None, None
        self.relu = nn.ReLU()
        self.init(features)

        self.softmax = nn.Softmax(dim=1)

    def init(self, features):
        self.V = nn.Parameter(nn.init.uniform_(torch.empty((self.N_k, features, 1, 1), requires_grad=True)))
        # self.V.shape = (output channels, input channels, kernel width, kernel height)
        self.sigmas = nn.Parameter(nn.init.constant_(torch.empty((1, self.N_k, 1, 1), requires_grad=True), 0.1))

    def forward(self, input):
        # Calculate the pairwise distances between the codewords and the feature vectors
        x_square = torch.sum(input=input, dim=1, keepdim=True)
        y_square = torch.sum(self.V ** 2, dim=1, keepdim=True).permute([3, 0, 1, 2]) # permute axis to

        dists = x_square + y_square - 2 * F.conv2d(input, self.V)
        #dists = torch.maximum(dists, torch.zeros(size=dists.shape))
        dists = self.relu(dists) # replace maximum to keep grads

        quantized_features = self.softmax(- dists / (self.sigmas ** 2))

        # Compile the histogram
        if self.spatial_level == 0:
            histogram = torch.mean(quantized_features, dim=[2, 3])
        elif self.spatial_level == 1:
            shape = quantized_features.shape
            mid_1 = shape[2] / 2
            mid_1 = int(mid_1)
            mid_2 = shape[3] / 2
            mid_2 = int(mid_2)
            histogram1 = torch.mean(quantized_features[:, :, :mid_1, :mid_2], [2, 3])
            histogram2 = torch.mean(quantized_features[:, :, mid_1:, :mid_2], [2, 3])
            histogram3 = torch.mean(quantized_features[:, :, :mid_1, mid_2:], [2, 3])
            histogram4 = torch.mean(quantized_features[:, :, mid_1:, mid_2:], [2, 3])
            histogram = torch.stack([histogram1, histogram2, histogram3, histogram4], 1)
            histogram = torch.reshape(histogram, (-1, 4 * self.N_k))
        else:
            # No other spatial level is currently supported (it is trivial to extend the code)
            assert False

        # Simple trick to avoid rescaling issues
        return histogram * self.N_k

    def compute_output_shape(self, input_shape): # 当spatial_level=0时，输出的特征数=n_codewords，为1时输出的特征数为n_codewords * 4
        if self.spatial_level == 0:
            return (input_shape[0], self.N_k)
        elif self.spatial_level == 1:
            return (input_shape[0], 4 * self.N_k)


def initialize_bof_layers(model, data_loader, n_samples=100, n_feature_samples=5000, batch_size=32, k_means_max_iters=300,
                          k_means_n_init=4):
    """
    Initializes the BoF layers of a model
    :param model: the model
    :param data: data to be used for initializing the model
    :param n_samples: number of data samples used for the initializes
    :param n_feature_samples: number of feature vectors to be used for the clustering process
    :param batch_size:
    :param k_means_max_iters: the maximum number of iterations for the clustering algorithm (k-means)
    :param k_means_n_init: defines how many times to run the k-means algorithm
    :return:
    """
    features = {}
    def get_features(name):
        def hook(module, input):
            if len(input) == 1:
                data = input[0].cpu().detach().permute([0, 2, 3, 1]).numpy()
                features[name].append(data.reshape(-1, data.shape[-1]))

        return hook

    iternum = int(n_samples / batch_size + 0.5)
    for name, layer in model.named_modules():
        if isinstance(layer, BoF_Pooling):
            print("Found BoF layer (layer %s), initializing..." % name)

            # Compile a function for getting the feature vectors
            # get_features = K.function([model.input] + [model.training], [model.layers[i - 1].output])
            features[name] = []
            handler = layer.register_forward_pre_hook(get_features(name))

            # iterate dataset to trigger hook to get features
            for i in range(iternum):
                data, labels = data_loader.__iter__().next()

                if len(list(data.shape)) == 5:
                    data = data[:, 0]
                if torch.cuda.is_available():
                    data = data.cuda()
                output = model(data)

            handler.remove()

            layer_features = np.concatenate(features[name])
            np.random.shuffle(layer_features)
            layer_features = layer_features[:n_feature_samples]

            # Cluster the features
            kmeans = KMeans(n_clusters=layer.N_k, n_init=k_means_n_init, max_iter=k_means_max_iters)
            kmeans.fit(layer_features)
            # V of BoF pooling layer
            V = kmeans.cluster_centers_
            V = V.reshape((V.shape[0], V.shape[1], 1, 1))

            # Set the value for the codebook
            layer.V.data = torch.tensor(np.float32(V)).cuda() if torch.cuda.is_available() else \
                torch.tensor(np.float32(V))
            # Get the mean distance for initializing the sigmas
            mean_dist = np.mean(pairwise_distances(layer_features[:100]))

            # Set the value for sigmas
            sigmas = np.ones((1, layer.N_k, 1, 1)) * (mean_dist ** 2)
            layer.sigmas.data = torch.tensor(np.float32(sigmas)).cuda() if torch.cuda.is_available() else \
                torch.tensor(np.float32(sigmas))


if __name__ == '__main__':
    x = torch.ones(size=(32, 32, 11, 11)) * 0.5
    model = BoF_Pooling(64, features=32, spatial_level=1)
    y = model(x)
    print(y.mean())
