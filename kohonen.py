import numpy as np
import matplotlib.pyplot as plt

from normalize import normalize_data


class KohonenNetwork:
    def __init__(self, input_dim, map_dim):
        self.input_dim = input_dim
        self.map_dim = map_dim
        self.weights = np.random.rand(map_dim[0], map_dim[1], input_dim)

    def train(self, data, epochs=10, lr=0.1):
        for epoch in range(epochs):
            for i in range(len(data)):
                input_vec = data[i]
                bmu = self.get_bmu(input_vec)
                self.update_weights(bmu, input_vec, lr)

    def get_bmu(self, input_vec):
        input_vec = input_vec.reshape((1, 1, self.input_dim))
        distances = np.sum((self.weights - input_vec) ** 2, axis=2)
        bmu_pos = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_pos

    def update_weights(self, bmu, input_vec, lr):
        bmu_weights = self.weights[bmu[0], bmu[1], :]
        for i in range(self.map_dim[0]):
            for j in range(self.map_dim[1]):
                dist = np.sqrt(np.sum((bmu - np.array([i, j])) ** 2))
                if dist <= 1:
                    neighbor_factor = np.exp(-(dist ** 2)) / (2 * (0.5 ** 2))
                    self.weights[i, j, :] += lr * neighbor_factor * (input_vec - self.weights[i, j, :])


def main():
    # normalize the data
    my_data = normalize_data()
    my_data = np.array(my_data).transpose()

    # train the data using a Kohonen Network
    net = KohonenNetwork(6, (2, 2))
    net.train(my_data, epochs=100, lr=0.1)

    # plot the network
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(my_data[:, 0], my_data[:, 1], my_data[:, 2], c='b')
    for i in range(net.map_dim[0]):
        for j in range(net.map_dim[1]):
            ax.scatter(net.weights[i, j, 0], net.weights[i, j, 1], net.weights[i, j, 2], c='r', s=100, marker='x')

    # plt.show()
    plt.savefig('plot.png')


if __name__ == '__main__':
    main()
