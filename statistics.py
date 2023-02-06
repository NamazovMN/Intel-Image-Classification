import collections
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage
from skimage import io
from sklearn.metrics import confusion_matrix

from collect_data import CollectData
from dataset import IntelSet
from train import Train
from utilities import *


class Statistics:
    """
    Class is utilized for providing statistics information before and after training of the model
    """

    def __init__(self, config_parameters: dict, info_object: CollectData, resize: bool, train_set: IntelSet,
                 dev_set: IntelSet):
        """
        Method is an initializer for the class
        :param config_parameters: required parameters for the project
        :param info_object: information object which includes all required information per dataset
        :param resize: boolean variable specifies whether resizing will be applied or not
        :param train_set: training dataset object
        :param dev_set: test dataset object
        """
        self.configuration = self.set_configuration(config_parameters, info_object, resize)
        self.runner = Train(config_parameters, train_set, dev_set)
        self.id2label = self.set_reverse_labels()

    @staticmethod
    def set_configuration(parameters: dict, info_object: CollectData, resize: bool) -> dict:
        """
           Method is utilized for extracting task-specific parameters from all required parameters and add additional
           parameters to the parameters object
           :param parameters: dictionary that contains all task-specific parameters
           :param info_object: information object which includes all required information per dataset
           :param resize: boolean variable specifies whether resizing will be applied or not
        """
        image_dim = f"_{parameters['image_dim']}" if resize else ''
        processed_dir = os.path.join(parameters['dataset_dir'], f'processed{image_dim}')
        info_dir = os.path.join(processed_dir, 'statistics')
        environment = os.path.join('train_results', f'experiment_{parameters["exp_num"]}')
        results = os.path.join(environment, 'results.pickle')
        check_dir(info_dir)
        return {
            'processed_dir': processed_dir,
            'info_object': info_object,
            'info_dir': info_dir,
            'environment': environment,
            'results': results
        }

    def set_reverse_labels(self) -> dict:
        """
        Method is used for reversing label to idx dictionary to idx to label
        :return: dictionary, where keys are label idx and values are labels
        """
        return {idx: label for label, idx in self.configuration['info_object'].label2id.items()}

    def show_distribution(self, is_train: bool) -> None:
        """
        Method is utilized to plot and print data distribution according to the image labels
        :param is_train: boolean variable to specify whether train dataset is analyzed or test
        :return: None
        """
        info_parameter = 'train' if is_train else 'test'
        info_dict_train = self.configuration['info_object'][info_parameter]
        labels = [data['label'] for data in info_dict_train]
        distribution = Counter(labels)
        dist_data = {label: distribution[idx] for label, idx in self.configuration['info_object'].label2id.items()}
        for each_label, num_data in dist_data.items():
            print(f"{num_data} images belong to {each_label} in {info_parameter} dataset")
        self.plot_bar(dist_data, is_train=is_train)

    def plot_bar(self, info_data: dict, is_train: bool = True) -> None:
        """
        Method is utilized for plotting the data distribution bar graph for specific dataset
        :param info_data: dictionary which contains information for plotting
        :param is_train: specifies whether train dataset or test dataset is analyzed
        :return: None
        """
        train_info = 'train' if is_train else 'test'
        plt.figure()
        title = f"Data distribution for labels in {train_info} dataset"
        plt.title(title)
        plt.xticks(np.arange(len(info_data.keys())))
        plt.bar(info_data.keys(), info_data.values())
        plt.plot()
        figure_path = os.path.join(self.configuration['info_dir'],
                                   f'{train_info}_dist.png')
        plt.savefig(figure_path)

    def provide_examples(self) -> None:
        """
        Method is used to visualize examples for each label
        :return: None
        """
        images = dict()
        for label, idx in self.configuration['info_object'].label2id.items():
            for each in self.configuration['info_object']['train']:
                if each['label'] == idx:
                    images[label] = each['image_path']
                    break
        half = int(len(images) / 2)
        figure, axis = plt.subplots(2, half, figsize=(20, 10))
        figure.suptitle(f'Example images for labels in dataset')
        for label_idx, (label, image_path) in enumerate(images.items()):
            current_idx = label_idx if label_idx < half else label_idx - half
            idx = 1 if label_idx >= half else 0
            axis[idx, current_idx].set_title(label)
            image = io.imread(image_path)
            image = skimage.transform.resize(image, (800, 800))
            axis[idx, current_idx].imshow(image)
            plt.plot()
            plt.grid()
        figure_path = os.path.join(self.configuration['info_dir'], f'examples.png')
        figure.savefig(figure_path)

    def plot_results(self, is_accuracy: bool = True) -> None:
        """
        Method is used to plot accuracy/loss graphs after training session is over, according to provided variable
        :param is_accuracy: boolean variable specifies the type of data will be plotted
        :return: None
        """

        metric_key = 'acc' if is_accuracy else 'loss'
        dev_data = list()
        train_data = list()
        with open(self.configuration['results'], 'rb') as result_data:
            result_dict = pickle.load(result_data)
        ordered = collections.OrderedDict(sorted(result_dict.items()))

        for epoch, results in ordered.items():
            dev_data.append(results[f'dev_{metric_key}'])
            train_data.append(results[f'train_{metric_key}'])
        plt.figure()
        plt.title(f'{metric_key.title()} results over {len(result_dict.keys())} epochs')
        plt.plot(list(result_dict.keys()), train_data, 'g', label='Train')
        plt.plot(list(result_dict.keys()), dev_data, 'r', label='Validation')
        plt.grid()
        plt.xlabel('Number of epochs')
        plt.ylabel(f'{metric_key.title()} results')
        plt.legend(loc=4)
        figure_path = os.path.join(self.configuration['environment'], f'{metric_key}_plot.png')
        plt.savefig(figure_path)
        plt.show()

    def get_confusion_details(self, metric: str = 'f1_dev') -> None:
        """
        Method is utilized for inference over test set and creating confusion matrix
        :param metric: string object which specifies metric that the best model will be chosen accordingly
        :return: None
        """
        epoch = self.runner.load_model(metric=metric, is_best=True)
        _, _, _, predictions, targets = self.runner.evaluate(epoch)
        confusion = confusion_matrix(targets, predictions)
        self.plot_confusion_matrix(confusion)

    def plot_confusion_matrix(self, confusion: np.array) -> None:
        """
        Method is utilized to plot confusion matrix according to the given matrix
        :param confusion: numpy array for confusion matrix
        :return: None
        """
        labels = [self.id2label[idx] for idx in range(len(self.configuration['info_object'].label2id))]
        plt.figure(figsize=(8, 6), dpi=100)
        sns.set(font_scale=1.1)

        ax = sns.heatmap(confusion, annot=True, fmt='d', )

        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(labels)

        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(labels)
        ax.set_title(f"Confusion Matrix", fontsize=14, pad=20)
        image_name = os.path.join(self.configuration['environment'], f'confusion_matrix.png')
        plt.savefig(image_name)
        plt.show()

    def get_statistics(self, before: bool = True) -> None:
        """
        Method is utilized for providing statistics
        :param before: boolean variable which specifies types of statistics that will be provided
        :return: None
        """
        if before:
            self.show_distribution(is_train=True)
            self.show_distribution(is_train=False)
            self.provide_examples()
        else:

            self.plot_results(is_accuracy=True)
            self.plot_results(is_accuracy=False)
            self.get_confusion_details()
