import numpy as np 
from keras.datasets import mnist

class NN:
    weights = []
    bias = []
    activation_functions = []
    hidden_layer={}
    output_layer={}
    def __init__(self, input_size):
        self.input_size = input_size

    def add_layer(self, layer_size, activation_function):
        if not bool(self.weights):
            self.weights.append(np.random.randn(layer_size, self.input_size) / np.sqrt(self.input_size))
            self.bias.append(np.random.randn(layer_size, 1) / np.sqrt(layer_size))
        else:
            self.weights.append(np.random.randn(layer_size, self.bias[-1].shape[0]) / np.sqrt(self.bias[-1].shape[0]))
            self.bias.append(np.random.randn(layer_size, 1) / np.sqrt(layer_size))
        self.activation_functions.append(activation_function)

    def get_activation_function(self, z, activation, derivative):
        if activation.upper() == 'SIGMOID':
            if derivative == False:
                return self.sigmoid(z)
            else:
                return self.d_sigmoid(z)
        elif activation.upper() == 'SOFTMAX':
            return self.softmax(z)

        '''elif activation.upper() == 'RELU'
            if derivative == False:
                return self.softmax(z)
            else:
                return self.d_softmax(z)'''

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, z):
        z_u = np.exp(z)
        return z_u / np.sum(z_u)

    def categorical_cross_entropy_error(self, y, y_hat):
        dec = -y_hat[self.decode_y(y)][0]
        return -np.log(dec)

    def forward_prop(self, x, y):
        z = []
        a = []

        a.append(x)

        for k in range(len(self.bias)):
            z.append((self.weights[k] @ a[-1]) + self.bias[k])
            a.append(self.get_activation_function(z[-1], self.activation_functions[k], False))
        
        return a, z

    def backward_prop(self, a, z, y, m):
        d_weights = []
        d_bias = []

        y_hat = a[-1] - y  #d_z
        d_z = y_hat
    
        d_w = (1/m) * (y_hat @ a[-2].T)
        d_weights.append(d_w)
        d_b = (1/m) * np.sum(y_hat, axis=1, keepdims=True)
        d_bias.append(d_b)

        for k in range(len(self.weights)-1, 0, -1):
            d_a = self.weights[k].T @ d_z
            d_z = d_a * self.get_activation_function(z[k-1], self.activation_functions[k-1], True)
            
            d_w = (1/m) * (d_z @ a[k-1].T)
            d_weights.append(d_w)
            d_b = (1/m) * np.sum(d_z, axis=1, keepdims=True)
            d_bias.append(d_b)

        loss = self.categorical_cross_entropy_error(y, y_hat)

        return d_weights, d_bias, loss
    
    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size, lambda_=0.01, alpha=0.1):
        self.lambda_ = lambda_
        print(f"initial accuracy: {self.calc_accuracy(x_train, y_train)}")
        batches = int(x_train.shape[0] // batch_size)
        for k in range(epochs):
            epoch_loss = 0
            ## randomize samples from training set
            random_indices = np.random.permutation(x_train.shape[0])
            
            for j in range(batches):
                start = j * batch_size
                end = min(start+batch_size, x_train.shape[0]-1)
                
                x = x_train[random_indices[start:end], :]
                y = y_train[:, random_indices[start:end]]
                m_batch = end - start

                epoch_loss += self.mini_batch(x, y, m_batch, alpha)

            # regularize loss
            #epoch_loss += 
            print(f"loss after {k+1} epochs: {epoch_loss}")
            print(f"training accuracy: {self.calc_accuracy(x_train, y_train)}")
            print(f"test accuracy: {self.calc_accuracy(x_test, y_test)}")

    def mini_batch(self, x, y, m_batch, alpha):
        batch_loss = 0
        for i in range(m_batch):
            a, z = self.forward_prop(x[i, :].reshape([x.shape[1],1]), y[:, i].reshape([y.shape[0],1]))
            d_weights, d_bias, loss = self.backward_prop(a, z, y[:, i].reshape([y.shape[0],1]), 1)
            batch_loss += loss
            self.update_params(d_weights, d_bias, alpha, m_batch)
        return batch_loss.copy()

    def update_params(self, d_weights, d_bias, alpha, m_batch):
        d_weights = np.flip(d_weights)
        d_bias = np.flip(d_bias)
        for k in range(len(d_weights)):
            self.weights[k] -= alpha * (d_weights[k] + (self.lambda_/m_batch) * self.weights[k])
            self.bias[k] -= alpha * d_bias[k]

    def calc_accuracy(self, x_train, y_train):
        acc = 0
        for i in range(x_train.shape[0]):
            predict, _ = self.forward_prop(x_train[i, :].reshape([x_train.shape[1],1]), y_train[:, i].reshape([y_train.shape[0],1]))
            if np.argmax(predict[-1]) == self.decode_y(y_train[:, i].reshape([y_train.shape[0],1])):
                acc += 1
        return acc/x_train.shape[0]

    def decode_y(self, y):
        return np.squeeze(np.where(np.ndarray.flatten(y) == 1))


def split_y(y):
    temp = np.zeros([10, y.shape[0]])
    for i in range(y.shape[0]):
        temp[y[i], i] = 1
    return temp

############################################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
x_test = x_test.reshape([x_test.shape[0], x_test.shape[1] * x_test.shape[2]])

#only train on n samples
n=10000
rand_indices = np.random.choice(x_train.shape[0], n, replace=True)[:n]

x_train = x_train[rand_indices, :]
y_train = y_train[rand_indices]
rand_indices = np.random.choice(x_test.shape[0], n, replace=True)[:n]
x_test = x_test[rand_indices, :]
y_test = y_test[rand_indices]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = split_y(y_train)
y_test = split_y(y_test)

# classify input size
Net = NN(784)
Net.add_layer(500, 'sigmoid')
Net.add_layer(200, 'sigmoid')
Net.add_layer(10, 'softmax')
Net.train(x_train, y_train, x_test, y_test, 10, 100, 0.01, 0.03)
