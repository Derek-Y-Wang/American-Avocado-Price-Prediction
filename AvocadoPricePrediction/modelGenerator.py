import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import dataProcessor as data
import numpy as np

training_df = data.AvocadoCleaner().process_data()
test_df = data.AvocadoCleaner().process_data()


# @title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Defined the create_model and traing_model functions.")


# @title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against 200 random training examples."""

    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Create a scatter plot from 200 random points of the dataset.
    random_examples = training_df.sample(n=300)
    plt.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 5
    y0 = trained_bias
    x1 = 50
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()


print("Defined the plot_the_model and plot_the_loss_curve functions.")

my_label = 'AveragePrice'
my_feature = 'Total Volume'
# The following variables are the hyperparameters.
learning_rate = 0.07
epochs = 250
batch_size = 30

# Split the original training set into a reduced training set and a
# validation set.
validation_split = 0.2

# Shuffle the examples.
shuffled_train_df = training_df.reindex(
    np.random.permutation(training_df.index))

# Invoke the functions to build and train the model. Train on the shuffled
# training set.
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, shuffled_train_df,
                                         my_feature,
                                         my_label, epochs, batch_size)

plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)


def predict_price(n, feature, label):
    """Predict admission chance based on a feature."""

    batch = training_df[feature][50:50 + n]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature              label               predicted")
    print("  value              value               value")
    print("Avocados(10thousands)  Avocado $            Avocado $ ")
    print("--------------------------------------")
    for i in range(n):
        print("%2f     %2f            %2f" % (training_df[feature][50 + i],
                                              training_df[label][50 + i],
                                              predicted_values[i][0]))


predict_price(10, my_feature, my_label)
