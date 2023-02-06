import os
import numpy as np
import random
import json
import PIL
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import rotate, AffineTransform, warp


class ReadData(object):
    def __init__(self, dataset_path, generated_dataset_path, augmented_dataset_path, configuration_path, examples_path,
                 width, height, dataset_type):
        self.dataset_path = dataset_path
        self.generated_dataset_path = generated_dataset_path
        self.augmented_dataset_path = augmented_dataset_path
        self.configuration_path = configuration_path
        self.examples_path = examples_path
        self.width = width
        self.height = height
        self.dataset_type = dataset_type
        self.data_path = os.path.join(self.generated_dataset_path, self.dataset_type + '_data.npy')
        self.labels_path = os.path.join(self.generated_dataset_path, self.dataset_type + '_labels.npy')
        self.shuffled_data_path = os.path.join(self.generated_dataset_path, self.dataset_type + '_shuffled_data.npy')
        self.shuffled_labels_path = os.path.join(self.generated_dataset_path,
                                                 self.dataset_type + '_shuffled_labels.npy')

    def aspect_ratio_resize_smart(self, img, base=150):
        if img.size[0] <= img.size[1]:
            basewidth = base
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), PIL.Image.LANCZOS)
        else:
            baseheight = base
            wpercent = (baseheight / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(wpercent)))
            img = img.resize((wsize, baseheight), PIL.Image.LANCZOS)

        img = img.resize((self.width, self.height), PIL.Image.LANCZOS)
        return np.array(img) / 255.0

    def get_directories(self):
        # this function is used to get directories and generate the classes according to labels of folders
        # if config.json exists, it loads data from there
        # else (this happens only the first run), it generates dictionary and save it newly generated config.json document

        directories = {}

        file_to_save = os.path.join(self.configuration_path, 'directories.json')

        if os.path.exists(file_to_save):
            with open(file_to_save) as dir_file:
                directories = json.load(dir_file)
        else:
            for each in enumerate(os.listdir(self.dataset_path)):
                directories[each[1]] = each[0]
            with open(file_to_save, 'w') as dir_file:
                json.dump(directories, dir_file)

        return directories

    def generate_dict_images(self):
        # This function is generated a dictionary of images. Keys of dictionary are class names, and values
        # are list of images.
        directories = self.get_directories()
        dict_data = {}
        for each in directories.keys():
            path_for_class = os.path.join(self.dataset_path, each)
            images = []
            print('Images belong to {} label are gathered...'.format(each))
            for each_image in tqdm(os.listdir(path_for_class)):
                path_for_image = os.path.join(path_for_class, each_image)
                image = Image.open(path_for_image)
                images.append(self.aspect_ratio_resize_smart(image))
            dict_data[directories[each]] = images
        return dict_data

    def save_dataset(self):
        # This function is used to save datasets as npy files to the related folders. It is used to prevent more time
        # to generate dataset at each run.
        dictionary_of_data = self.generate_dict_images()
        data = []
        labels = []
        for each in dictionary_of_data:
            data += dictionary_of_data[each]
            for i in range(len(dictionary_of_data[each])):
                labels.append(each)

        dataset = []
        for each_data, each_label in zip(data, labels):
            dataset.append((each_data, each_label))
        random.shuffle(dataset)
        shuffled_data = []
        shuffled_labels = []
        for (each_data, each_label) in dataset:
            shuffled_data.append(each_data)
            shuffled_labels.append(each_label)
        np.save(self.data_path, np.array(data))
        np.save(self.labels_path, np.array(labels))
        np.save(self.shuffled_data_path, np.array(shuffled_data))
        np.save(self.shuffled_labels_path, np.array(shuffled_labels))
        self.plot_six_images(data[0:6], labels[0:6], self.dataset_type)
        self.plot_six_images(shuffled_data[0:6], shuffled_labels[0:6], 'shuffled_' + self.dataset_type)
        print('Data and Labels of train dataset were saved to the {} as npy files, separately'.format(
            self.generated_dataset_path))

    def load_dataset(self):
        # This function loads datasets from the path that they were saved.
        data = np.load(self.shuffled_data_path)
        labels = np.load(self.shuffled_labels_path)
        return data, labels

    def augment_dataset(self, augmentation_types):
        # This function is used to augment dataset according to the given augmentation types. Notice that, we augment
        # dataset in 10 scenarios.
        # Additionally, after augmentation it saves augmented datasets to the given path (augmented_datasets).
        data, labels = self.load_dataset()

        for each in augmentation_types:
            image_data = []
            image_labels = []
            path_augmentation_data = os.path.join(self.augmented_dataset_path,
                                                  each + '_' + self.dataset_type + '_data.npy')
            path_augmentation_labels = os.path.join(self.augmented_dataset_path,
                                                    each + '_' + self.dataset_type + '_labels.npy')
            print("Augmentation type {} is applied to whole data.".format(each))
            for (each_data, each_label) in tqdm(zip(data, labels)):
                if each == 'rotation':
                    image_data.append(rotate(each_data, angle=65))
                    image_labels.append(each_label)
                elif each == 'flipud':
                    image_data.append(np.flipud(each_data))
                    image_labels.append(each_label)
                elif each == 'fliplr':
                    image_data.append(np.fliplr(each_data))
                    image_labels.append(each_label)
                elif each == 'shear':
                    tf = AffineTransform(shear=0.5)
                    image_data.append(warp(each_data, tf, order=1, preserve_range=True, mode='wrap'))
                    image_labels.append(each_label)
                elif each == 'rotation_flipud_together':
                    image_data.append(rotate(np.flipud(each_data), angle=65))
                    image_labels.append(each_label)
                elif each == 'rotation_fliplr_together':
                    image_data.append(rotate(np.fliplr(each_data), angle=65))
                    image_labels.append(each_label)
                elif each == 'rotation_shear_together':
                    tf = AffineTransform(shear=0.5)
                    image_data.append(rotate(warp(each_data, tf, order=1, preserve_range=True, mode='wrap'), angle=65))
                    image_labels.append(each_label)
                elif each == 'fliplr_flipud_together':
                    image_data.append(np.flipud(np.fliplr(each_data)))
                    image_labels.append(each_label)
                elif each == 'shear_flipud_together':
                    tf = AffineTransform(shear=0.5)
                    image_data.append(np.flipud(warp(each_data, tf, order=1, preserve_range=True, mode='wrap')))
                    image_labels.append(each_label)
                elif each == 'shear_fliplr_together':
                    tf = AffineTransform(shear=0.5)
                    image_data.append(np.fliplr(warp(each_data, tf, order=1, preserve_range=True, mode='wrap')))
                    image_labels.append(each_label)
                else:
                    print('This kind of augmentation was not mentioned!')
            np_data = np.array(image_data)
            np_labels = np.array(image_labels)
            np.save(path_augmentation_data, np_data)
            np.save(path_augmentation_labels, np_labels)
            self.plot_six_images(np_data[0:6], np_labels[0:6], self.dataset_type, True, each)

    def load_augmented_dataset(self, augmentation_types):
        # This function is used to load augmented datasets. If input is empty list it returns original dataset.
        # Otherwise, it returns chosen augmentation types together with original data.
        data, labels = self.load_dataset()
        data_augmented = list(data)
        labels_augmented = list(labels)

        if len(augmentation_types) != 0:
            count = 0
            info_sentence = 'Chosen augmentation type(s): '
            for each in augmentation_types:
                info_sentence += each + '   '
            print(info_sentence)
        else:
            print('Augmentation type has not been chosen. Thus, raw data will be loaded')
        print(25*'*-')
        for each in augmentation_types:
            print('{} type augmented data and labels are loaded from {}.'.format(each, self.augmented_dataset_path))
            path_augmentation_data = os.path.join(self.augmented_dataset_path,
                                                  each + '_' + self.dataset_type + '_data.npy')
            path_augmentation_labels = os.path.join(self.augmented_dataset_path,
                                                    each + '_' + self.dataset_type + '_labels.npy')
            data_augmented += list(np.load(path_augmentation_data))
            labels_augmented += list(np.load(path_augmentation_labels))
        dataset = []
        for data_each in zip(data_augmented, labels_augmented):
            dataset.append(data_each)
        random.shuffle(dataset)
        data_augmented = []
        labels_augmented = []
        for (each_data, each_label) in dataset:
            data_augmented.append(each_data)
            labels_augmented.append(each_label)

        data_augmented = np.array(data_augmented)
        labels_augmented = np.array(labels_augmented)
        print('{} augmentation type(s) has/have been chosen.'.format(len(augmentation_types)))

        print('Model will be trained on the dataset which has shape of: {}'.format(data_augmented.shape))

        return data_augmented, labels_augmented

    def plot_six_images(self, image_array, labels_array, dataset_version, augmented=False,
                        augmentation_type='rotation'):
        # This function is used in order to generate dataset examples. During generation and augmentation of dataset
        # this function is called and results are saved to the given path (dataset_examples).
        rows = 2
        columns = 3
        figure, axes = plt.subplots(rows, columns)
        count = 0
        for each_row in range(rows):
            for each_column in range(columns):
                axes[each_row, each_column].imshow(image_array[count])
                axes[each_row, each_column].set_title(labels_array[count])
                count += 1
        if not augmented:
            file_name = dataset_version + '.png'
        else:
            file_name = dataset_version + '_' + augmentation_type + '_augmented.png'
        file_name = os.path.join(self.examples_path, file_name)
        plt.savefig(file_name)
        plt.close()
