import numpy as np 
from keras.datasets import mnist

class NN:
    hidden_layer={}
    output_layer={}
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer['bias'] = np.random.randn(hidden_size, 1) / np.sqrt(hidden_size)
        self.hidden_layer['Weight'] = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.output_layer['bias'] = np.random.randn(output_size, 1) / np.sqrt(hidden_size)
        self.output_layer['Weight'] = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def dSigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def categorical_cross_entropy_error(self, y, y_hat):
        dec = -y_hat[self.decode_y(y)][0]
        return -np.log(dec)

    def forward_prop(self, x, y):
        z = []
        a = []

        a.append(x)

        z.append((self.hidden_layer['Weight'] @ a[-1]) + self.hidden_layer['bias'])
        a.append(self.sigmoid(z[-1]))

        z.append((self.output_layer['Weight'] @ a[-1]) + self.output_layer['bias'])
        a.append(self.sigmoid(z[-1]))
        
        return a, z

    def backward_prop(self, a, z, y, m):
        y_hat = a[-1] - y
    
        dW2 = (1/self.output_size) * (y_hat @ a[1].T)
        db2 = (1/m) * np.sum(y_hat, axis=1, keepdims=True)
        
        dA1 = self.output_layer['Weight'].T @ y_hat
        dZ1 = dA1 * self.dSigmoid(z[0])
        
        dW1 = (1/m) * (dZ1 @ a[0].T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        loss = self.categorical_cross_entropy_error(y, y_hat)
        
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return grads, loss
    
    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size, alpha=0.1):
        batches = int(x_train.shape[0] // batch_size)
        for k in range(epochs):
            epoch_loss = 0
            ## might implement drawing a random sample from the training set
            x_train_shuffled = x_train
            y_train_shuffled = y_train
            
            for j in range(batches):
                start = j * batch_size
                end = min(start+batch_size, x_train.shape[0]-1)
                
                x = x_train_shuffled[start:end, :]
                y = y_train_shuffled[:, start:end]
                m_batch = end - start

                epoch_loss += self.mini_batch(x, y, m_batch, alpha)

  
            print(f"loss after {k+1} epochs: {epoch_loss}")
            print(f"training accuracy: {self.calcAccuracy(x_train_shuffled, y_train_shuffled)}")
            print(f"test accuracy: {self.calcAccuracy(x_test, y_test)}")

    def mini_batch(self, x, y, m_batch, alpha):
        batch_loss = 0
        for i in range(m_batch):
            a, z = self.forward_prop(x[i, :].reshape([x.shape[1],1]), y[:, i].reshape([y.shape[0],1]))
            grads, loss = self.backward_prop(a, z, y[:, i].reshape([y.shape[0],1]), 1)
            batch_loss += loss
            self.update_params(grads, alpha)
        return batch_loss

    def update_params(self, grads, alpha):
        self.output_layer['Weight'] -= alpha * grads['dW2']
        self.output_layer['bias'] -= alpha * grads['db2']
        
        self.hidden_layer['Weight'] -= alpha * grads['dW1']
        self.hidden_layer['bias'] -= alpha * grads['db1']

    def calcAccuracy(self, x_train, y_train):
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
n=5000
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

Net = NN(784, 300, 10)
Net.train(x_train, y_train, x_test, y_test, 10, 100, 0.1)
