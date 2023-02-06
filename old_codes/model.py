import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.applications import VGG16


class CNNIntel(object):
    def __init__(self, batch_size, num_classes, learning_rate, epochs, models_path, input_shape):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.models_path = models_path
        self.input_shape = input_shape

    def model(self, train_dataset, train_labels, validation_dataset, model_name):
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=25)

        model = Sequential()

        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D((4, 4)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (4, 4), activation="relu"))
        model.add(MaxPooling2D((3, 3)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        history = model.fit(train_dataset, train_labels, epochs=self.epochs, validation_data=validation_dataset,
                            callbacks=[early_stopping], verbose=2)
        path = os.path.join(self.models_path, model_name + '.h5')
        model.save(path)

        return history


class TransferModel(object):
    def __init__(self, epochs, input_shape, models_path, defrosted=True, weights=None, trainable=True):
        self.epochs = epochs
        self.input_shape = input_shape
        self.models_path = models_path
        self.defrosted = defrosted
        self.weights = weights  # if it is none then model is going to be non-pretrained
        self.trainable = trainable  # if trainable is true, non-pretrained model will start to train

    def model(self, train_data, train_labels, validation_dataset, model_name):
        model = VGG16(input_shape=self.input_shape, include_top=False, weights=self.weights)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        if self.defrosted:
            for layer in model.layers[:10]:
                layer.trainable = False
        else:
            model.trainable = self.trainable

        model.summary()

        global_average_layer = GlobalAveragePooling2D()

        prediction_layer = Dense(6, activation='sigmoid')

        model_transfer = Sequential([model, global_average_layer, prediction_layer])

        base_learning_rate = 0.0001
        model_transfer.compile(
            optimizer=optimizers.RMSprop(lr=base_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Fitting the model with train batches
        history = model_transfer.fit(
            train_data, train_labels,
            epochs=self.epochs, verbose=2,
            validation_data=validation_dataset,
            callbacks=[early_stopping]
        )
        path = os.path.join(self.models_path, model_name + '.h5')
        model.save(path)

        return history
