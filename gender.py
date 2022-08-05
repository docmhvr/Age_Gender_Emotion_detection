from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

def compile_gender_model(Input):
    input = Input(shape=(224, 224, 3))
    cnn1 = Conv2D(36, kernel_size=3, activation='relu')(input)
    cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)
    cnn2 = Conv2D(64, kernel_size=3, activation='relu')(cnn1)
    cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)
    cnn3 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
    cnn3 = MaxPool2D(pool_size=3, strides=2)(cnn3)
    cnn4 = Conv2D(256, kernel_size=3, activation='relu')(cnn3)
    cnn4 = MaxPool2D(pool_size=3, strides=2)(cnn4)
    cnn5 = Conv2D(512, kernel_size=3, activation='relu')(cnn4)
    cnn5 = MaxPool2D(pool_size=3, strides=2)(cnn5)
    dense = Flatten()(cnn5)
    dense = Dropout(0.2)(dense)
    dense = Dense(512, activation='relu')(dense)
    dense = Dense(512, activation='relu')(dense)
    output = Dense(1, activation='sigmoid', name='gender')(dense)
    sex_model = Model(input, output)
    sex_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return sex_model