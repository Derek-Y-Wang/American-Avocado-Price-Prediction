import tensorflow as tf
import numpy as np
from artifical_neural_net_datapreprocessor import NeuralNetPreprocessor


class ArtificialBrain:
    def __init__(self, train_X, train_y):
        self.X_train = train_X
        self.y_train = train_y
        self.ann = tf.keras.models.Sequential()

    def create_net(self, hidden):
        for i in range(hidden):
            self.ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

        self.ann.add(tf.keras.layers.Dense(units=1))

    def train_brain(self):
        self.ann.compile(optimizer='adam', loss='mean_squared_error')
        self.ann.fit(self.X_train, self.y_train, batch_size=40, epochs=150)

    def save(self):
        self.ann.save("ANN")

    def load(self, path):
        self.ann = tf.keras.models.load_model(path)


if __name__ == "__main__":
    data = NeuralNetPreprocessor()
    data.process()

    # print(data.get_X_test())
    art_nn = ArtificialBrain(data.get_X_train(), data.get_y_train())
    # art_nn.create_net(6)
    # art_nn.train_brain()
    # art_nn.save()

    # test
    art_nn.load("ANN")
    y_pred = art_nn.ann.predict(data.get_X_test())
    np.set_printoptions(precision=2)
    viewable = np.concatenate((y_pred.reshape(len(y_pred), 1),
                               data.get_y_test().reshape(len(data.get_y_test()),
                                                         1)), 1)
    print(viewable)






