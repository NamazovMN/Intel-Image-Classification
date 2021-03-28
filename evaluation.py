import os

import matplotlib.pyplot as plt

from model import CNNIntel
from model import TransferModel
from read_data import ReadData
import tensorflow
from evaluate_choose import EvaluateChoose
from tensorflow.keras import optimizers

def plot_images_results(path_image, history_model, metrics):
    plt.plot(history_model.history[metrics])
    plt.plot(history_model.history['val_' + metrics])
    plt.title('model ' + metrics)
    plt.ylabel(metrics)
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path_image)
    plt.close()


# required folder paths
dataset_path_train = 'data/seg_train/seg_train'
dataset_path_validation = 'data/seg_test/seg_test'
generated_datasets_path = 'generated_datasets'
augmented_datasets_path = 'augmented_datasets'
configuration_path = 'configuration_documents'
dataset_examples_path = 'dataset_examples'
models_path = 'models'
results_path = 'results'

# model and image parameters
width = 75
height = 75
learning_rate = 0.0001
batch_size = 256
number_of_classes = 6
input_shape = (width, height, 3)

# Declaration of classes of data readers
rd_train = ReadData(dataset_path_train, generated_datasets_path, augmented_datasets_path, configuration_path,
                    dataset_examples_path, width, height, 'train')
rd_validation = ReadData(dataset_path_validation, generated_datasets_path, augmented_datasets_path, configuration_path,
                         dataset_examples_path, width, height, 'validation')

# Validation data is taken from folders
data_validation, labels_validation = rd_validation.load_augmented_dataset([])
validation_dataset = (data_validation, labels_validation)

# Evaluate and Choose class is declared here

ev_choose = EvaluateChoose(models_path, configuration_path)
maximum_result = ev_choose.choose_max(data_validation, labels_validation)
augmentation_of_max = ev_choose.split_return_aug_types(maximum_result[0])

# Transfer Learning for non-pretrained model 100 epochs

#     unfreezed   weights     trainable
# NP  False       None        True          non pretrained
# PF  False       'imagenet'  False         frozen pretrained
# PD  True        'imagenet'  True          defrosted pretrained

models = {'pre_trained_frozen': [False, 'imagenet', False],
          'unfreezed_pretrained': [True, 'imagenet', True],
          'non_pretrained': [False, None, True]}

epochs = 100
print('VGG16 will be trained in {} combinations on the following dataset: {}.'.format(len(models), augmentation_of_max))
train_data_raw, train_labels_raw = rd_train.load_augmented_dataset(augmentation_of_max)

for each in models.keys():
    model_params = models[each]
    print('Transfer Learning model has chosen as {} version of ResNet'.format(each))
    print('Model parameters will be: Unfreezed: {}, Weights: {}, Trainable: {}'.format(model_params[0], model_params[1],
                                                                                       model_params[2]))

    transfer_learning_model = TransferModel(epochs, input_shape, models_path, defrosted=model_params[0],
                                            weights=model_params[1], trainable=model_params[2])
    history = transfer_learning_model.model(train_data_raw, train_labels_raw, validation_dataset, model_name=each)

    path_image_accuracy = os.path.join(results_path, each + '_accuracy.png')
    path_image_loss = os.path.join(results_path, each + '_loss.png')

    plot_images_results(path_image_accuracy, history, 'accuracy')
    plot_images_results(path_image_loss, history, 'loss')


