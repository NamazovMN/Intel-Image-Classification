import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16


class BuildModel:
    """
    Class is used as dynamic model builder. It also checks compatibility among provided parameters
    """

    def __init__(self, config_parameters: dict):
        """
        Method is an initializer for the class
        :param config_parameters: required parameters for the project
        """
        self.configuration = config_parameters
        self.check_compatibility_cnn()
        self.check_compatibility_fcn()
        self.model = self.build_model_layers()
        self.save_model_structure()

    def check_compatibility_fcn(self) -> None:
        """
        Method is used as compatibility checker among provided information for Fully Connected Network (FCN)
        :return: None
        """
        fcn_layers = ['dropout', 'bn_lin']
        for each_layer in fcn_layers:
            if len(self.configuration[each_layer]) != len(self.configuration['dense']):
                raise IndexError(f"Number of {each_layer} is not compatible with number of dense layers")

    def check_compatibility_cnn(self) -> None:
        """
        Method is used as compatibility checker among provided information for Convolutional Neural Network (CNN)
        :return: None
        """
        cnn_layers = ['dropout_cnn', 'bn', 'mp_kernels', 'mp_strides', 'kernels', 'strides']
        for each_layer in cnn_layers:
            if len(self.configuration[each_layer]) != len(self.configuration['conv']):
                raise IndexError(f"Number of {each_layer} layers is not compatible with conv layers")

    def build_conv_layers(self) -> dict:
        """
        Method is utilized to collect layers will be used for CNN model
        :return: dictionary where keys are name of layers and values are specific layers
        """
        conv_list = dict()
        image_dim = self.configuration['image_dim']
        for idx, each in enumerate(self.configuration['conv']):
            if idx == 0:
                conv_layer = layers.Conv2D(
                    each,
                    kernel_size=self.configuration['kernels'][idx],
                    strides=self.configuration['strides'][idx],
                    activation='relu', input_shape=(image_dim, image_dim, 3))

            else:
                conv_layer = layers.Conv2D(
                    each,
                    kernel_size=self.configuration['kernels'][idx],
                    strides=self.configuration['strides'][idx],
                    activation='relu')
            conv_list[f'conv_{idx}'] = conv_layer
            conv_list[f'pooling_{idx}'] = layers.MaxPooling2D(
                pool_size=(self.configuration['mp_kernels'][idx], self.configuration['mp_strides'][idx])
            )
            if self.configuration['bn'][idx]:
                conv_list[f'bn_{idx}'] = layers.BatchNormalization()
            if self.configuration['dropout_cnn'][idx]:
                conv_list[f'dropout_{idx}'] = layers.Dropout(self.configuration['dropout_cnn'][idx])
        conv_list['flatten'] = layers.Flatten()
        return conv_list

    def build_linear_layers(self) -> dict:
        """
        Method is used for collecting layers of FCN
        :return: dictionary where keys are names of layers and values are specific layers
        """
        linear_layers = dict()

        for idx, layer in enumerate(self.configuration['dense']):
            linear_layers[f'dense_{idx}'] = layers.Dense(self.configuration['dense'][idx])
            if self.configuration['dropout']:
                linear_layers[f'dropout_lin_{idx}'] = layers.Dropout(self.configuration['dropout'][idx])
            if self.configuration['bn_lin']:
                linear_layers[f'bn_lin_{idx}'] = layers.BatchNormalization()
        linear_layers['out'] = layers.Dense(6, activation='softmax')
        return linear_layers

    def build_model_layers(self) -> dict:
        """
        Method is utilized for setting the model layers up
        :return: dictionary where keys are names of layers and values are specific layers
        """
        model_layers = self.build_conv_layers()
        for name, layer in self.build_linear_layers().items():
            model_layers[name] = layer

        return model_layers

    def save_model_structure(self) -> None:
        """
        Method is utilized for saving the model structure
        :return: None
        """
        model_struct_file = os.path.join('train_results',
                                         f'experiment_{self.configuration["exp_num"]}/model_structure.pickle')
        with open(model_struct_file, 'wb') as model_file:
            pickle.dump(self.model, model_file)


class TransferModel(tf.Module):
    """
    Class is utilized for setting the Transfer Learning Model
    """

    def __init__(self, parameters: dict, defrosted: bool = True):
        """
        Method is utilized as initializer of the class
        :param parameters: dictionary object which includes all required information for this very task
        :param defrosted: boolean variable specifies whether CNN layers will be frozen or not
        """
        super().__init__()
        input_shape = (parameters['image_dim'], parameters['image_dim'], 3)
        model = VGG16(input_shape=input_shape, include_top=False, weights=parameters['transfer_weights'])
        if defrosted:
            for layer in model.layers[:10]:
                layer.trainable = False
        else:
            model.trainable = True

        model.summary()
        global_average_layer = layers.GlobalAveragePooling2D()

        prediction_layer = layers.Dense(len(parameters['labels_dict']), activation='sigmoid')
        self.model = models.Sequential([model, global_average_layer, prediction_layer])

    def __call__(self, input_data: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Method is a call function to perform feed-forward phase of training
        :param input_data: batch of image tensors or single image tensors
        :param training: boolean variable specifies whether training (True) or evaluation (False) is performed
        :return: output in shape of batch_size x number of classes
        """
        out = self.model(input_data, training=training)
        return out


class MySequentialModule(tf.Module):
    def __init__(self, model_object: BuildModel, name: str = 'IntelModel'):
        """
        Method is utilized as initializer of the class

        :param model_object: model object which conveys model structure for the specific experiment
        :param name: string object for indicating the model name
        """
        super().__init__(name=name)
        self.model = models.Sequential()
        for name, layer in model_object.model.items():
            self.model.add(layer)

    def __call__(self, input_data: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Method is a call function to perform feed-forward phase of training
        :param input_data: batch of image tensors or single image tensors
        :param training: boolean variable specifies whether training (True) or evaluation (False) is performed
        :return: output in shape of batch_size x number of classes
        """
        out = self.model(input_data, training=training)

        return out
# model = models.Sequential()
# model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(10))
