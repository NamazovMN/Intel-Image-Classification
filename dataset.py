import pickle
import random

import numpy as np
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
from collect_data import CollectData
from utilities import *


class IntelSet:
    def __init__(self, config_parameters: dict, info_obj: CollectData, ds_type: str,
                 resize_image: bool, zero_center: bool = False):
        """
        Method is an initializer for the class
        :param config_parameters: required parameters for the project
        :param info_obj: information object which includes all required information per dataset
        :param ds_type: type of the dataset is going to be prepared
        :param resize_image: boolean variable specifies whether resizing will be applied or not
        :param zero_center: boolean variable specifies whether zero centering will be applied or not
        """
        self.configuration = self.set_configuration(config_parameters, ds_type, info_obj, resize_image)
        self.ds_type = ds_type
        self.labels_dict = self.set_one_hot()
        self.dataset = self.process_image()

    @staticmethod
    def set_configuration(parameters, ds_type, info_object, resize_image):
        """
        Method is utilized for extracting task-specific parameters from all required parameters and add additional
        parameters to the parameters object
        :param parameters: dictionary that contains all task-specific parameters
        :param ds_type: type of the dataset is going to be prepared
        :param info_object: information object which includes all required information per dataset
        :param resize_image: boolean variable specifies whether resizing will be applied or not
        :return: dictionary that contains required parameters for dataset preparation
        """
        img_dim = f'_{parameters["image_dim"]}' if resize_image else ''
        process_dir = os.path.join(parameters['dataset_dir'], f'processed{img_dim}')
        check_dir(process_dir)
        out_dir = os.path.join(process_dir, f'{ds_type}.pickle')
        return {
            'image_dim': parameters['image_dim'],
            'resize': resize_image,
            'labels_dict': info_object.label2id,
            'info_obj': info_object[ds_type],
            'process_dir': process_dir,
            'out_dir': out_dir
        }

    def get_image(self, image_path: str) -> np.array:
        """
        Method is utilized for reading and resizing image, it is set True
        :param image_path: path to the specific data
        :return: image in form of numpy array
        """
        image = io.imread(image_path)
        if self.configuration['resize']:
            image = resize(image, (self.configuration['image_dim'], self.configuration['image_dim'], image.shape[2]))
        return image

    def set_one_hot(self) -> dict:
        """
        Method is used for setting labels to one-hot vector form, since loss computation is done based on logits
        :return: dictionary contains label index and their one-hot representations
        """
        one_hot = np.zeros(shape=(len(self.configuration['labels_dict']), len(self.configuration['labels_dict'])))
        labels_dict = dict()
        for idx in self.configuration['labels_dict'].values():
            one_hot[idx][idx] = 1
            labels_dict[idx] = one_hot[idx]
        return labels_dict

    def process_image(self) -> dict:
        """
        Method is used as main function of the class, so that all steps are combined inside of this method
        :return: dictionary which contains data and label information for dataset
        """
        if not os.path.exists(self.configuration['out_dir']):
            images_dict = {
                'data': list(),
                'label': list()
            }
            ti = tqdm(iterable=self.configuration['info_obj'], total=len(self.configuration['info_obj']),
                      desc=f'Data is collected for {self.ds_type}')

            random.shuffle(self.configuration['info_obj'])
            for data in ti:
                images_dict['data'].append(self.get_image(data['image_path']))
                images_dict['label'].append(self.labels_dict[data['label']])
            with open(self.configuration['out_dir'], 'wb') as raw:
                pickle.dump(images_dict, raw)

        with open(self.configuration['out_dir'], 'rb') as raw:
            images_dict = pickle.load(raw)

        return images_dict

    def __getitem__(self, item: str) -> list:
        """
        Method is a getter which gets data or label information from dataset
        :param item: string that specifies whether data or labels are required from the dataset
        :return: list object for given request
        """
        return self.dataset[item]
