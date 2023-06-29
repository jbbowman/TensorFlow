# TensorFlow
## Background
The following repository describes how to build and train a machine learning algorithm with TensorFlow, NumPy, and pandas. I will mainly use TensorFlow’s Keras API to create and train my model. I will also rely on some work from my previous lab working with neural networks to create a hypothetical situation and generate input data. To begin, I will first introduce TensorFlow and how it works.

## Introduction
TensorFlow is an open-source library developed by Google primarily for tasks that require heavy numerical computations. It is particularly well-suited for implementing large-scale machine learning models, as its name suggests - the "tensor" refers to the data represented as arrays, and the "flow" refers to the computations applied to these structures as they pass through a graph-like setup.
The core feature of TensorFlow is its use of data flow graphs. In these graphs, nodes represent mathematical operations, while the edges represent the data (tensors) that pass between them. This structure allows TensorFlow to use a single API for deploying computations to one or more CPUs or GPUs in a desktop, server, or mobile device without needing to rewrite code.
Below is an example of a linear regression model using TensorFlow. First, we'll load a dataset, prepare it for our model, and finally, we'll create, train, and evaluate the model. For this task, we'll be using the Titanic dataset, hosted by Google.

Code:
```Python
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from IPython.display import clear_output
import tensorflow as tf

# load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# converting categorical data into numeric data
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# creating an input function to convert our dataframe into a tensorflow dataset object
def mk_input_fctn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
      ds = tf.data.Dataset.from_tensor_slices(
          (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
      if shuffle:
          ds = ds.shuffle(1000)  # randomize order of data
      ds = ds.batch(batch_size).repeat(
          num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
      return ds  # return a batch of the dataset
  return input_function

train_input_fn = mk_input_fctn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = mk_input_fctn(dfeval, y_eval, num_epochs=1, shuffle=False)

# creating the model
linear_est = tf.estimator.LinearClassifier(
    feature_columns=feature_columns)  # we create a linear estimtor by passing the feature columns we created earlier

# training model
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears console output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model

# predicted survival rate
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
```

This python module starts by loading the Titanic dataset. We, then convert categorical data into numeric data, which is a common requirement for machine learning algorithms. The results are printed. Then, we create an input function to feed data into our model, and a function to generate our model - in this case, a Linear Classifier. We train the model using our training data and evaluate it using our evaluation data. Finally, we print the accuracy of our model.

Output:
```Output
[VocabularyListCategoricalColumn(key='sex', vocabulary_list=('male', 'female'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='n_siblings_spouses', vocabulary_list=(1, 0, 3, 4, 2, 5, 8), dtype=tf.int64, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='parch', vocabulary_list=(0, 1, 2, 5, 3, 4), dtype=tf.int64, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='class', vocabulary_list=('Third', 'First', 'Second'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='deck', vocabulary_list=('unknown', 'C', 'G', 'A', 'B', 'D', 'F', 'E'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='embark_town', vocabulary_list=('Southampton', 'Cherbourg', 'Queenstown', 'unknown'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='alone', vocabulary_list=('n', 'y'), dtype=tf.string, default_value=-1, num_oov_buckets=0), NumericColumn(key='age', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='fare', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
0.7765151
```

In the next section, we will apply these concepts to a real-world problem. Specifically, we will prepare data and build and train a model to evaluate incoming data so as to accurately hypothesize an outcome. 

## Problem
Suppose a bank wants to predict whether a customer is likely to default on their loan based on their age, credit score, debt-to-income ratio, and loan-to-asset value ratio. The bank has this information for previous borrowers as well as the outcome of their loan. Using this dataset, the bank can use a logistic regression model to predict whether a new customer is likely to default on a loan or not. Before developing our logistic regression model, the following code generates customers, a data framework, and data in a prepared format to test and develop our model:

Code:
```Python
customers.py:
import numpy as np
import pandas as pd

class Customer:
    instance_counter = 0

    def __init__(self):
        self.id = Customer.instance_counter
        self.credit_score = (np.random.randint(300, 851) - 300) / 550
        self.debt_income_ratio = np.random.random()
        self.loan_value_ratio = np.random.random()
        self.age = np.random.randint(18, 75) / 75
        self.default = np.random.rand() < 0.5 * self.debt_income_ratio + 0.5 * self.loan_value_ratio - 0.5 * self.credit_score
        Customer.instance_counter += 1

    def __str__(self):
        return self.id

def generate_customers(num_customers):
    customer_data = []
    for i in range(num_customers):
        customer = Customer()
        customer_data.append([customer.credit_score, customer.debt_income_ratio, customer.loan_value_ratio, customer.age, customer.default])
    return customer_data

def generate_dataframe(customer_data):
    return pd.DataFrame(customer_data, columns=["Credit Score", "Debt/Income", "Loan/Value", "Age", "Default"])

def prep_data(data):
    rg = np.random.default_rng()

    features = data.drop(columns='Default').to_numpy()
    targets = data['Default'].to_numpy()
    weights = rg.random(features.shape[1])
    return features, targets, weights

if __name__ == '__main__':
    customer_data = generate_customers(5)
    data = generate_dataframe(customer_data)
    prepped_data = prep_data(data)

    print(customer_data, data, prepped_data, sep="\n\n")
```

Output:
```Output
[[0.6436363636363637, 0.633090468982135, 0.10385785585637752, 0.9333333333333333, 0], [0.23454545454545456, 0.6169783228052368, 0.11853714815446115, 0.30666666666666664, 0], [0.11454545454545455, 0.46611204163509357, 0.9958532078406778, 0.6533333333333333, 1], [0.10909090909090909, 0.6516349798529134, 0.49722891199630226, 0.92, 1], [0.8127272727272727, 0.9122495953119779, 0.7318774525822924, 0.8933333333333333, 1]]

   Credit Score  Debt/Income  Loan/Value       Age  Default
0      0.643636     0.633090    0.103858  0.933333        0
1      0.234545     0.616978    0.118537  0.306667        0
2      0.114545     0.466112    0.995853  0.653333        1
3      0.109091     0.651635    0.497229  0.920000        1
4      0.812727     0.912250    0.731877  0.893333        1

(array([[0.64363636, 0.63309047, 0.10385786, 0.93333333],
       [0.23454545, 0.61697832, 0.11853715, 0.30666667],
       [0.11454545, 0.46611204, 0.99585321, 0.65333333],
       [0.10909091, 0.65163498, 0.49722891, 0.92      ],
       [0.81272727, 0.9122496 , 0.73187745, 0.89333333]]), array([0, 0, 1, 1, 1], dtype=int64), array([0.00624203, 0.39826641, 0.94854967, 0.37765307]))
```

## Building/Training a Neural Network
After generating customer data, we can begin developing our TensorFlow model. The overarching aim of the code is to create a machine learning model to predict loan defaults, given customer data.

Code:
```Python
import tensorflow as tf

class LoanPredictionModel:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=10, validation_split=0.2):
        self.history = self.model.fit(X, y, epochs=epochs, validation_split=validation_split, verbose=1)
```

In tf_model.py, the model is constructed within a class named LoanPredictionModel. The model is a sequential Keras model with three dense layers. The first layer has 32 neurons, the second has 16, and the last layer has a single neuron. Activation functions for these layers are respectively relu, relu, and sigmoid. The model uses adam as the optimizer, binary_crossentropy as the loss function, and accuracy as the metric for the training process.
Below is the main.py module. It serves as the entry point of the program, initializing the model and plotting the training history.

Code:
```Python
from customers import generate_customers, generate_dataframe, prep_data
from tf_model import LoanPredictionModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    loan_predictor = init_model()
    plot_history(loan_predictor.history)

def init_model():
    customer_data = generate_customers(20)
    data = generate_dataframe(customer_data)
    features, targets, _ = prep_data(data)

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    loan_predictor = LoanPredictionModel()
    loan_predictor.train(x_train_scaled, y_train)

    return loan_predictor
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
```

The init_model function prepares the data for the model. It uses functions imported from the customers module to generate a set of synthetic customer data and prepare it for use in the model. The features are scaled using a StandardScaler to ensure that they are on the same scale, which can help the model train more effectively. The data is then split into training and testing sets using a 80/20 split.
The model is trained using the prepared and scaled training data. The trained model, loan_predictor, is then returned from the init_model function.
In plot_history, the training history of the model is plotted, specifically, the training and validation accuracy and loss over epochs, using matplotlib's pyplot.

Output:
```Output
Train on 64 samples, validate on 16 samples

Epoch 1/10

32/64 [==============>...............] - ETA: 0s - loss: 0.7318 - acc: 0.4375
64/64 [==============================] - 0s 3ms/sample - loss: 0.7047 - acc: 0.5469 - val_loss: 0.7131 - val_acc: 0.3125
Epoch 2/10

32/64 [==============>...............] - ETA: 0s - loss: 0.6902 - acc: 0.5938
64/64 [==============================] - 0s 62us/sample - loss: 0.6904 - acc: 0.6406 - val_loss: 0.6986 - val_acc: 0.5000
Epoch 3/10

32/64 [==============>...............] - ETA: 0s - loss: 0.6958 - acc: 0.5938
64/64 [==============================] - 0s 62us/sample - loss: 0.6762 - acc: 0.6562 - val_loss: 0.6844 - val_acc: 0.5000
Epoch 4/10

32/64 [==============>...............] - ETA: 0s - loss: 0.6648 - acc: 0.6562
64/64 [==============================] - 0s 62us/sample - loss: 0.6624 - acc: 0.6875 - val_loss: 0.6708 - val_acc: 0.5625
Epoch 5/10

32/64 [==============>...............] - ETA: 0s - loss: 0.6777 - acc: 0.5938
64/64 [==============================] - 0s 62us/sample - loss: 0.6493 - acc: 0.7188 - val_loss: 0.6582 - val_acc: 0.6875
Epoch 6/10

32/64 [==============>...............] - ETA: 0s - loss: 0.6467 - acc: 0.7188
64/64 [==============================] - 0s 47us/sample - loss: 0.6365 - acc: 0.7500 - val_loss: 0.6461 - val_acc: 0.7500
Epoch 7/10

32/64 [==============>...............] - ETA: 0s - loss: 0.6336 - acc: 0.7500
64/64 [==============================] - 0s 62us/sample - loss: 0.6233 - acc: 0.7656 - val_loss: 0.6345 - val_acc: 0.7500
Epoch 8/10

32/64 [==============>...............] - ETA: 0s - loss: 0.6252 - acc: 0.7812
64/64 [==============================] - 0s 47us/sample - loss: 0.6113 - acc: 0.7969 - val_loss: 0.6236 - val_acc: 0.8125
Epoch 9/10

32/64 [==============>...............] - ETA: 0s - loss: 0.5905 - acc: 0.8125
64/64 [==============================] - 0s 62us/sample - loss: 0.6009 - acc: 0.7969 - val_loss: 0.6134 - val_acc: 0.8125
Epoch 10/10

32/64 [==============>...............] - ETA: 0s - loss: 0.5751 - acc: 0.8438
64/64 [==============================] - 0s 47us/sample - loss: 0.5889 - acc: 0.8125 - val_loss: 0.6039 - val_acc: 0.8125
```

The printed output from a sample run indicates that the model is trained for 10 epochs. In each epoch, the model's loss decreases, and accuracy increases, both on the training data and the validation data. This indicates that the model is learning to predict whether a loan will default based on the input features. The final accuracy on the validation data is 81.25%, which suggests that the model is reasonably effective at making accurate predictions.

Conclusion
This paper has demonstrated how to implement a TensorFlow model for binary classification using an advanced neural network with multiple neurons (perceptrons). The program combines customer data generation with model training and evaluation, providing a complete implementation of a neural network.
