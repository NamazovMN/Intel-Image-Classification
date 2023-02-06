import os.path
import pickle
from tqdm import tqdm


class CollectData:
    """
    Class is used to collect required information for image dataset, in order to accelerate data processing phase
    """
    def __init__(self, config_parameters: dict):
        """
        Method is an initializer for the class
        :param config_parameters: required parameters for the project
        """
        self.configuration = self.set_configuration(config_parameters)
        self.label2id = self.get_labels()
        self.info_dict = self.process_info()

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method is utilized for extracting task-specific parameters from all required parameters and add additional
        parameters to the parameters object
        :param parameters: dictionary that contains all task-specific parameters
        :return: task specific parameters dictionary
        """
        return {
            'input_dir': os.path.join(parameters['dataset_dir'], 'dataset'),
            'labels_path': os.path.join(parameters['dataset_dir'], 'labels_dict.pickle'),
            'info_dir': os.path.join(parameters['dataset_dir'], 'info_dict.pickle')
        }

    def get_labels(self) -> dict:
        """
        Method is used to collect labels of the images and save it, in order to avoid from randomly collected labels
        for each running phase
        :return: label to idx dictionary
        """
        if not os.path.exists(self.configuration['labels_path']):
            train_folder = os.path.join(self.configuration['input_dir'], 'seg_train')
            labels_dict = {label: idx for idx, label in enumerate(os.listdir(train_folder))}
            with open(self.configuration['labels_path'], 'wb') as labels_data:
                pickle.dump(labels_dict, labels_data)

        with open(self.configuration['labels_path'], 'rb') as labels_data:
            labels_dict = pickle.load(labels_data)
        return labels_dict

    def collect_info(self, ds_type: str) -> list:
        """
        Method is utilized for collecting required info in a dictionary per image and combining all this information in
        a list
        :param ds_type: string that specifies dataset whether it is training set or test set
        :return: list object includes all required information for dataset
        """
        path_data = os.path.join(self.configuration['input_dir'], f'{ds_type}')
        dataset = list()
        if 'pred' not in ds_type:
            for labels, label_id in self.label2id.items():
                current_folder = os.path.join(path_data, labels)
                current_ti = tqdm(
                    iterable=os.listdir(current_folder),
                    desc=f'Info is collected for {labels} for {ds_type}'
                )
                current_data = [{'image_path': os.path.join(current_folder, image_path), 'label': label_id}
                                for image_path in current_ti]
                dataset.extend(current_data)
        else:
            dataset = [{'image_path': os.path.join(path_data, image_path), 'label': -1}
                       for image_path in os.listdir(path_data)]
        return dataset

    def process_info(self) -> dict:
        """
        Method is used for collecting all information for each set in a dictionary
        :return: dict object includes all required information per dataset type
        """
        info_dict = dict()
        if not os.path.exists(self.configuration['info_dir']):
            for ds_type in os.listdir(self.configuration['input_dir']):
                info_dict[ds_type] = self.collect_info(ds_type)
            with open(self.configuration['info_dir'], 'wb') as info_data:
                pickle.dump(info_dict, info_data)
        with open(self.configuration['info_dir'], 'rb') as info_data:
            info_dict = pickle.load(info_data)
        return info_dict

    def __getitem__(self, item: str) -> list:
        """
        Method is used for getting information for specific dataset type
        :param item: set type, which will be used to collect information accordingly
        :return: list object for corresponding type
        """
        return self.info_dict[f'seg_{item}']
