from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

def compile_model(Input):
    input = Input(shape=(48, 48, 1))
    cnn1 = Conv2D(36, kernel_size=3, activation='relu')(input)
    cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)
    cnn2 = Conv2D(64, kernel_size=3, activation='relu')(cnn1)
    cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)
    cnn3 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
    cnn3 = MaxPool2D(pool_size=3, strides=2)(cnn3)
    dense = Flatten()(cnn3)
    dense = Dropout(0.3)(dense)
    dense = Dense(256, activation='relu')(dense)
    output = Dense(7, activation='softmax', name='race', kernel_regularizer=l1(1))(dense)
    emotion_model = Model(input, output)
    emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    emotion_model.fit(input, y, epochs=50, batch_size=32, verbose=0)
    return emotion_model