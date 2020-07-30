import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np


class AvocadoCleaner:
    def __init__(self):
        source = './avocado.csv'
        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.2f}".format

        self.data = pd.read_csv(source)

        self.data["Total Volume"] /= 10000.0

    def _remove_unwanted_features(self):
        for i in self.data:
            if i == "region" or i == 'Total Volume' or i == 'AveragePrice' or i == 'year':
                continue
            else:
                del self.data[i]

    def _remove_repeated_dates(self):
        day = set()
        for index, col in self.data.iterrows():
            if col['Date'] not in day:
                day.add(col['Date'])
            else:
                # print("removed")
                self.data.drop(self.data.index[index])
        # print(len(day))

    def _remove_extremes(self):
        for index, row in self.data.iterrows():
            if row['Total Volume'] > 50.0 or row['AveragePrice'] > 2:
                self.data = self.data.drop(index=index)
                self.data.reset_index(drop=True)

    def _fill_missing_data(self):
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.data[:, :-1])
        imputer.transform(self.data[:, :-1])

    def encode_region(self):
        # reoptimize numpy array has too many items for onehot encoding
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        # print(self.data['region'])
        X = self.data.iloc[:, :].values
        print(X)
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        return X.size

    def process_data(self):
        self._remove_unwanted_features()
        # self._remove_repeated_dates()
        self._remove_extremes()
        self._fill_missing_data()
        return self.data

    def plot_graph(self):
        # Label the axes.
        plt.xlabel("Total Volume")
        plt.ylabel("AveragePrice")
        random_examples = self.data.sample(n=100)
        plt.scatter(random_examples['Total Volume'],
                    random_examples['AveragePrice'])
        plt.show()


# a = AvocadoCleaner()
# a._remove_unwanted_features()
# a._remove_extremes()
# print(a.data)
# print(a.encode_region())

