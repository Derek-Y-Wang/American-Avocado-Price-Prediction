import tensorflow as tf
import numpy as np
import artificial_neural_net_datapreprocessor as processed_data


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
        self.ann.fit(self.X_train, self.y_train, batch_size=32, epochs=100)

    def test(self, x_test, expected):
        y_pred = self.ann.predict(x_test)
        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                              expected.reshape(len(expected), 1)), 1))


if __name__ == "__main__":
    data = processed_data.process()

    art_nn = ArtificialBrain(data.get_X_train(), data.get_y_train())
    art_nn.test(data.get_X_test, data.get_y_test)
