import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNetPreprocessor:

    def __init__(self):
        source = './avocado.csv'
        self.data = pd.read_csv(source)
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    def _remove_unrelated_features(self):
        for i in self.data:
            if i == 'Date' or i == '4046' or i == '4225' or i == '4770':
                del self.data[i]

    def process(self):
        X = self.data.iloc[:, 2:].values
        y = self.data.iloc[:, 1].values

        # label encode the regions as well as the year
        le = LabelEncoder()
        X[:, -1] = le.fit_transform(X[:, -1])
        X[:, -2] = le.fit_transform(X[:, -2])

        # We want to one hot encode the 'type'
        # remember one hot encoding moves the binaries to the front of the array
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-3])], remainder="passthrough")
        X = np.array(ct.fit_transform(X))

        # let us split that dataset into test and results
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1,
                                                            random_state=0)
        # feature scale
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def get_X_test(self):
        return self.X_test

    def get_X_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

a = NeuralNetPreprocessor()
a._remove_unrelated_features()
a.process()
