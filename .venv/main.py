import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Import data and name the columns as followed
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv('./Data/magic04.data', names=cols);
# print(df.head())

# Check unique occurences of class column
# print(df["class"].unique())

# Since computers cant understand letters that much
# convert g and h to 1 and 0
# astype(int) converts the selected column to integer
df["class"] = (df["class"] == "g").astype(int)
#print(df.head());

#=========================================================================
# Plot all the features into a histogram
# histogram represents the distribution of a value in the dataset

# for label in cols[: -1]:
#     plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.7, density=True)
#     plt.hist(df[df['class']==0][label], color='red', label='hydron', alpha=0.7 , density=True)
#     plt.title(label)
#     plt.ylabel('Probability')
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

# =======================================================================
# Make Train, valid and test datasets
# split 60% to training, 60-80% to valid, 80-100% to test
train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

# some labels have a higher value than other
# like one is in 100s and other is in 0.0s
# so this can show imbalance and innacuracy in data because
# model will give more importance to the higher value
# thats why we are scaling the values
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # There are more values of gamma than hydron so we want to increase the values of hydron
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# ========================================================================
# K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=3 )
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
# print(classification_report(y_test, y_pred))

# ========================================================================
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
# print(classification_report(y_test, y_pred))


# ========================================================================
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)
y_pred = lg_model.predict(X_test)
# print(classification_report(y_test, y_pred))

# =======================================================================
# Support Vector Mechanism
from sklearn.svm import SVC
svc_model = SVC()
svc_model = svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
#print(classification_report(y_test, y_pred))

# =======================================================================
# Neural Network for Classification
import tensorflow as tf

def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  plt.show()

# make layers of neural network
# keep last layer as output layer and sigmoid for classification 0 or 1
# nn_model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(32, activation='relu', input_shape=(10, )),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, epochs, batch_size):
    nn_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10, )),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

    history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    return nn_model, history

least_loss_value = float('inf')
least_loss_model = None

epochs = 100
for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, epochs, batch_size)
                plot_history(history)
                valid_loss = model.evaluate(X_valid, y_valid)[0]
                if valid_loss < least_loss_value:
                    least_loss_value = valid_loss
                    least_loss_model = model