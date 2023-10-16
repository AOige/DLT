import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

#Index 1 is activity_id
#Index [4:16)/[21,33)/[38,50) are 3D-acc1, 3D-acc2, 3D-gyro, 3D-magn of 3 IMUs respectively (36 features in total)
loc = [1] + [*range(4,16)] + [*range(21,33)] + [*range(38,50)]

def window(data, label, size, stride):
  ##Cut the array data and label according to the size and stride of the sliding window'
  x, y = [], []
  for i in range(0, len(label), stride):
    if i+size < len(label): #Discard data less than a sliding window size
        l = set(label[i:i+size])
        if len(l) > 1 or label[i] == 0: #When a sliding window contains multiple activities or activity_id is 0, discard:
          continue
        elif len(l) == 1:
          x.append(data[i: i + size, :])
          y.append(label[i])

  return x, y

def main():
    X, Y = [], []
    # Traverse 9 subject files
    for i in range(1, 10):
        total = pd.read_csv(r'C:\Users\AYO IGE\Documents\Datasets\HAR\PAMAP2\Protocol\subject10' + str(i) + '.dat', header=None, sep=' ', usecols=loc).values
        total = total[~np.isnan(total).any(axis=1), :]  # Remove NAs
        data = total[:, 1:]
        label = total[:, 0].reshape(-1)



        # Call the window function for sliding window processing
        x, y = window(data, label, 171, 85)
        X += x
        Y += y
    # Number the indices sequentially from 0
    cate_idx = list(Counter(Y).keys())
    cate_idx.sort()
    for i in range(len(Y)):
        Y[i] = cate_idx.index(Y[i])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    np.save('./x_train', x_train)
    np.save('./x_test', x_test)
    np.save('./y_train', y_train)
    np.save('./y_test', y_test)


if __name__ == '__main__':
    main()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, MaxPool1D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard


train_x = np.load(r"C:\Users\AYO IGE\x_train.npy")
train_y = np.load(r"C:\Users\AYO IGE\y_train.npy")
test_x = np.load(r"C:\Users\AYO IGE\x_test.npy")
test_y = np.load(r"C:\Users\AYO IGE\y_test.npy")

train_x.shape

from tensorflow.keras.layers import Multiply

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


from tensorflow.keras.utils import to_categorical
train_y = to_categorical(train_y, 12)
test_y = to_categorical(test_y, 12)
train_y.shape, test_y.shape


from keras.layers import GlobalAveragePooling1D, Reshape, Bidirectional, LSTM, concatenate, GlobalMaxPool1D
from keras.activations import tanh, relu


def SqueezeAndExcitation(inputs, ratio=8):
    b, c = (21,64)
    x = GlobalAveragePooling1D()
    x = Dense(c//ratio, activation="relu", use_bias=False)(inputs)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = Multiply()([inputs, x])
    return x


def WSenseNet(x, C = 128):
    conv1d_ = Conv1D(C, kernel_size = 5, padding="same", activation="elu")
    x = conv1d_(x)
    gmax_pool = GlobalMaxPool1D(keepdims = True)
    x = gmax_pool(x)
    xc = x
    x = Conv1D(C, kernel_size = 1, padding="same", activation='sigmoid')(xc)
    x = Multiply()([x, xc])
    return x

inputs = Input(shape=(171,36))

#Input 1
input1   = Input(shape=(171,36))
conv1_1  = Conv1D(filters=16, kernel_size=3, padding = 'same', data_format='channels_last', activation= 'relu')(input1)
batch1_1 = BatchNormalization()(conv1_1)
pool1_1  = MaxPool1D((2))(batch1_1)
conv1_2  = Conv1D(filters=32, kernel_size=5, padding = 'same', data_format='channels_last', activation= 'relu')(pool1_1)
batch1_2 = BatchNormalization()(conv1_2)
pool1_2  = MaxPool1D((2))(batch1_2)
conv1_3  = Conv1D(filters=64, kernel_size=7, padding = 'same', data_format='channels_last', activation= 'relu')(pool1_2)
batch1_3 = BatchNormalization()(conv1_3)
pool1_3  = MaxPool1D((2))(batch1_3)


##Input 1_2
input2   = Input(shape=(171,36))
Bilstm1_1  = Bidirectional(LSTM(40, return_sequences= True, activation= 'tanh'))(input2)
pool2_1  = MaxPool1D((2))(Bilstm1_1)
Bilstm1_2  = Bidirectional(LSTM(20, return_sequences= True, activation= 'tanh'))(pool2_1)
pool2_2  = MaxPool1D((2))(Bilstm1_2)
lstm1_1  = LSTM(64, return_sequences= True, activation= 'tanh')(pool2_2)
pool2_3  = MaxPool1D((2))(lstm1_1)

merge1 = concatenate([pool1_3, pool2_3], axis =1)

SEBlock1  = SqueezeAndExcitation(merge1)



#Input 2
input3   = Input(shape=(171,36))
conv2_1  = Conv1D(filters=16, kernel_size=3, padding = 'same', data_format='channels_last', activation= 'relu')(input3)
batch2_1 = BatchNormalization()(conv2_1)
pool3_1  = MaxPool1D((2))(batch2_1)
conv2_2  = Conv1D(filters=32, kernel_size=5, padding = 'same', data_format='channels_last', activation= 'relu')(pool3_1)
batch2_2 = BatchNormalization()(conv2_2)
pool3_2  = MaxPool1D((2))(batch2_2)
conv2_3  = Conv1D(filters=64, kernel_size=7, padding = 'same', data_format='channels_last', activation= 'relu')(pool3_2)
batch2_3 = BatchNormalization()(conv2_3)
pool3_3  = MaxPool1D((2))(batch2_3)


# ##Input 2_2
input4   = Input(shape=(171,36))
Bilstm2_1  = Bidirectional(LSTM(40, return_sequences= True, activation= 'tanh'))(input4)
pool4_1  = MaxPool1D((2))(Bilstm2_1)
Bilstm2_2  = Bidirectional(LSTM(20, return_sequences= True, activation= 'tanh'))(pool4_1)
pool4_2  = MaxPool1D((2))(Bilstm2_2)
lstm2_1  = LSTM(64, return_sequences= True, activation= 'tanh')(pool4_2)
pool4_3  = MaxPool1D((2))(lstm2_1)

merge2 = concatenate([pool3_3 , pool4_3], axis =1)
SEBlock2  = SqueezeAndExcitation(merge2)
#merged4  = concatenate([SEBlock1, SEBlock2], axis = 1)


#Input 3
input5   = Input(shape=(171,3))
conv3_1  = Conv1D(filters=16, kernel_size=3, padding = 'same', data_format='channels_last', activation= 'relu')(input5)
batch3_1 = BatchNormalization()(conv3_1)
pool5_1  = MaxPool1D((2))(batch3_1)
conv3_2  = Conv1D(filters=32, kernel_size=5, padding = 'same', data_format='channels_last', activation= 'relu')(pool5_1)
batch3_2 = BatchNormalization()(conv3_2)
pool5_2  = MaxPool1D((2))(batch3_2)
conv3_3  = Conv1D(filters=64, kernel_size=7, padding = 'same', data_format='channels_last', activation= 'relu')(pool5_2)
batch3_3 = BatchNormalization()(conv3_3)
pool5_3  = MaxPool1D((2))(batch3_3)


#Input 3_2
input6   = Input(shape=(171,3))
Bilstm3_1  = Bidirectional(LSTM(40, return_sequences= True, activation= 'tanh'))(input6)
pool6_1  = MaxPool1D((2))(Bilstm3_1)
Bilstm3_2  = Bidirectional(LSTM(20, return_sequences= True, activation= 'tanh'))(pool6_1)
pool6_2  = MaxPool1D((2))(Bilstm3_2)
lstm3_1  = LSTM(64, return_sequences= True, activation= 'tanh')(pool6_2)
pool6_3  = MaxPool1D((2))(lstm3_1)

merge3 = concatenate([pool5_3, pool6_3], axis =1)
SEBlock3  = SqueezeAndExcitation(merge3)



merged_alpha  = concatenate([SEBlock1, SEBlock2, SEBlock3], axis = 1)
lstm5    = Bidirectional(LSTM(64, return_sequences = True, activation ='tanh'))(merged_alpha)
lstm6    = LSTM(128, return_sequences = True, activation ='tanh')(lstm5)
WNet     = WSenseNet(lstm6)
flat     = Flatten()(WNet)
#dropout  = Dropout(0.2)(flat)
FC1     = Dense(512, activation = 'relu')(flat)
FC2     = Dense(128, activation = 'relu')(FC1)
output  = Dense(12, activation ='softmax')(FC2)

model = Model(inputs = [input1, input2, input3, input4, input5, input6], outputs = output)
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer= 'Adam')
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]


history = model.fit([train_x, train_x, train_x, train_x, train_x, train_x], train_y,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks
    )


pip install mlxtend


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score


y_pred = np.argmax(model.predict([test_x, test_x, test_x, test_x, test_x, test_x]), axis=-1)
y_true = np.argmax(test_y, axis=1)
    
acc = accuracy_score(y_true, y_pred,normalize=True)
f1 = f1_score(y_true, y_pred, average='weighted')
mat = confusion_matrix(y_true, y_pred)
report = (classification_report(y_true,y_pred))
plot_confusion_matrix(conf_mat = mat, show_normed=True, figsize=(10,10) )
#print("Experiment:",r)
print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")
print(mat)
print(report)
print("__________________________________________________")


import matplotlib.pyplot as plt


def plot_learningCurve(history, epochs):
  #accuracy
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc ='upper left')
  plt.show()
#validaion loss
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc ='upper left')
  plt.show()

plot_learningCurve(history, *)


import seaborn as sns

cf=np.empty((12,12))
for i in range(12):
    cf[i]=(mat[i]/np.sum(mat[i]))

# group_counts = ["{0:.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()]
labels = [f"{v1}" for v1 in group_percentages]
labels = np.asarray(labels).reshape(12,12)

fig = plt.figure(figsize=(12,10))
ax=fig.add_subplot(111)
sns.heatmap(mat, annot=labels,fmt="",cmap='RdYlBu_r',ax=ax)
ax.set_xlabel("Predicted Labels\n\n Overall_accuracy=""{0:.2%}".format(acc(y_true,y_pred)))
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(['rope_jumping','lying','sitting','standing','walking','running','cycling','Nordic_walking','ascending_stairs','descending_stairs','vacuum_cleaning','ironing'],rotation=90)
ax.yaxis.set_ticklabels(['rope_jumping','lying','sitting','standing','walking','running','cycling','Nordic_walking','ascending_stairs','descending_stairs','vacuum_cleaning','ironing'],rotation=360)
plt.savefig("Metrics//CM_PAMAP2",dpi=1200, bbox_inches='tight')

