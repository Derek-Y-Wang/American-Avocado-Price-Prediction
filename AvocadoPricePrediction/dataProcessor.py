import pandas as pd
import matplotlib.pyplot as plt


class AvocadoCleaner:
    def __init__(self):
        source = './avocado.csv'
        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.2f}".format

        self.data = pd.read_csv(source)
        self.data["Total Volume"] /= 1000.0

    def _remove_unwanted_features(self):
        for i in self.data:
            if i == 'Total Volume' or i == 'AveragePrice' or i == 'Date':
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

    def process_data(self):
        self._remove_unwanted_features()
        self._remove_repeated_dates()
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
# a.process_data()
# a.plot_graph()
# with pd.option_context('display.max_rows', 10, 'display.max_columns',
#                        None):  # more options can be specified also
#     print(a.data)
# a._remove_repeated_dates()
