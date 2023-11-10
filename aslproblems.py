import numpy as np
import tensorflow as tf
from datasets import sign_language
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

# Step 1: Load the training and test data
np.random.seed(5)
tf.set_random_seed(2)
(x_train, y_train), (x_test, y_test) = sign_language.load_data()

# Step 2: Visualize the training data
labels = ['A', 'B', 'C']
fig = plt.figure(figsize=(20, 5))
for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
    ax.set_title("{}".format(labels[y_train[i]]))
plt.show()

# Step 3: Examine the Dataset
num_A_train = np.sum(y_train == 0)
num_B_train = np.sum(y_train == 1)
num_C_train = np.sum(y_train == 2)
num_A_test = np.sum(y_test == 0)
num_B_test = np.sum(y_test == 1)
num_C_test = np.sum(y_test == 2)

print("Training set:")
print("\tA: {}, B: {}, C: {}".format(num_A_train, num_B_train, num_C_train))
print("Test set:")
print("\tA: {}, B: {}, C: {}".format(num_A_test, num_B_test, num_C_test))

# Step 4: Encode the data
y_train_OH = np_utils.to_categorical(y_train)
y_test_OH = np_utils.to_categorical(y_test)

# Step 5: Defining the Model
model = Sequential()
model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

# Step 6: Compiling the Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
hist = model.fit(x_train, y_train_OH, validation_split=0.20, epochs=2, batch_size=32)

# Step 8: Test the model
score = model.evaluate(x=x_test, y=y_test_OH, verbose=0)
print('Test accuracy:', score[1])

# Step 9: Visualize the Mistakes
y_probs = model.predict(x_test)
y_preds = np.argmax(y_probs, axis=1)
bad_test_idxs = np.where(y_preds != y_test)[0]

fig = plt.figure(figsize=(25, 4))
for i, idx in enumerate(bad_test_idxs):
    ax = fig.add_subplot(2, np.ceil(len(bad_test_idxs) / 2), i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    ax.set_title("{} (pred: {})".format(labels[y_test[idx]], labels[y_preds[idx]]))
