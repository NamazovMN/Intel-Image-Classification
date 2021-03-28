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


dataset_path_train = 'data/seg_train/seg_train'
dataset_path_validation = 'data/seg_test/seg_test'
generated_datasets_path = 'generated_datasets'
augmented_datasets_path = 'augmented_datasets'
configuration_path = 'configuration_documents'
dataset_examples_path = 'dataset_examples'
models_path = 'models'
results_path = 'results'

if not os.path.exists(configuration_path):
    os.makedirs(configuration_path)

if not os.path.exists(dataset_examples_path):
    os.makedirs(dataset_examples_path)

if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(models_path):
    os.makedirs(models_path)

width = 75
height = 75
augmentation_types = ['rotation',
                      'flipud',
                      'fliplr',
                      'shear',
                      'rotation_flipud_together',
                      'rotation_fliplr_together',
                      'rotation_shear_together',
                      'fliplr_flipud_together',
                      'shear_flipud_together',
                      'shear_fliplr_together']

rd_train = ReadData(dataset_path_train, generated_datasets_path, augmented_datasets_path, configuration_path,
                    dataset_examples_path, width, height, 'train')
rd_validation = ReadData(dataset_path_validation, generated_datasets_path, augmented_datasets_path, configuration_path,
                         dataset_examples_path, width, height, 'validation')

if not os.path.exists(generated_datasets_path):
    os.makedirs(generated_datasets_path)
    rd_train.save_dataset()
    rd_validation.save_dataset()

if not os.path.exists(augmented_datasets_path):
    os.makedirs(augmented_datasets_path)
    rd_train.augment_dataset(augmentation_types)

# 1. Train normal model
# 2. Choose and evaluate
# 3. choose max data and get this for transfer learning
# 4. train and evaluate


epochs = 150
learning_rate = 0.0001
batch_size = 256
number_of_classes = 6
input_shape = (width, height, 3)
history_models = {}
# # [],
augmentation_choices = [[],
                        ['rotation'],
                        ['flipud'],
                        ['fliplr'],
                        ['shear'],
                        ['rotation_flipud_together'],
                        ['rotation_fliplr_together'],
                        ['rotation_shear_together'],
                        ['fliplr_flipud_together'],
                        ['shear_flipud_together'],
                        ['shear_fliplr_together'],
                        ['shear', 'flipud'],
                        ['shear', 'fliplr'],
                        ['shear', 'rotation'],
                        ['rotation', 'fliplr'],
                        ['rotation', 'flipud'],
                        ['flipud', 'fliplr']]

print('There will be {} models running!'.format(3 + len(augmentation_choices)))
print('Customized Convolutional Neural Network will be trained on {} combinations of dataset augmentations.'.format(
    len(augmentation_choices)))

# # customized CNN model for Intel dataset 150 epochs for each dataset augmentation method
data_validation, labels_validation = rd_validation.load_augmented_dataset([])
validation_dataset = (data_validation, labels_validation)
count_models = 0
for each in augmentation_choices:
    count_models += 1
    data_train, labels_train = rd_train.load_augmented_dataset(each)
    model_name_2 = ''
    print('Model no: {}/{}'.format(count_models, len(augmentation_choices)))
    if len(each) > 1:
        for i in each:
            model_name_2 += '_'+i
    elif len(each) == 1:
        model_name_2 += '_' + each[0]
    else:
        model_name_2 += '_without_augmentation'
    model_name = 'model' + model_name_2

    print('{} started to train:'.format(model_name))
    print(100 * '*-')
    conv_net = CNNIntel(batch_size, number_of_classes, learning_rate, epochs, models_path, input_shape)
    history = conv_net.model(data_train, labels_train, validation_dataset, model_name)
    data_train, labels_train = [], []
    data_validation, labels_validation = [], []
    path_image_accuracy = os.path.join(results_path, model_name + '_accuracy.png')
    path_image_loss = os.path.join(results_path, model_name + '_loss.png')
    history_models[model_name] = history
    plot_images_results(path_image_accuracy, history, 'accuracy')
    plot_images_results(path_image_loss, history, 'loss')

# Transfer Learning for non-pretrained model 150 epochs

#     defrosted   weights     trainable
# NP  False       None        True          non pretrained
# PF  False       'imagenet'  False         frozen pretrained
# PD  True        'imagenet'  True          defrosted pretrained

# validation_data_raw = rd_validation.load_augmented_dataset([])
