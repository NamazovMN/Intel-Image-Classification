import os
import numpy as np
import tensorflow
import json
from tensorflow.keras import optimizers


class EvaluateChoose(object):
    def __init__(self, models_path, configuration_path):
        self.models_path = models_path
        self.configuration_path = configuration_path

    def evaluation(self, validation_data, validation_labels):
        # This function takes validation data and labels and evaluate them
        # Evaluation results are gathered in list which is saved to txt file. This saving is done because of preventing
        # evaluate it every running.
        # Returns results of evaluation which is list of tuples. Each tuple has such shape (model_name, acc, loss).
        results = []

        file = 'results_of_models.json'
        file_path = os.path.join(self.configuration_path, file)
        if os.path.exists(file_path):
            with open(file_path) as results_file:
                results_dict = json.load(results_file)
            results_str = results_dict['results']
            for each in results_str:
                results.append((each[0], float(each[1]), float(each[2])))
            print('Results were loaded from {}!'.format(self.configuration_path))
        else:
            models = os.listdir(self.models_path)
            for each in models[:-3]:
                print('{} model is loaded from the {} folder.'.format(each, self.models_path))
                model_path = os.path.join(self.models_path, each)
                model = tensorflow.keras.models.load_model(model_path)
                loss, acc = model.evaluate(validation_data, validation_labels, verbose=2)
                results.append((each, str(acc), str(loss)))
            results_dict = {'results': results}
            with open(file_path, 'w') as results_file:
                json.dump(results_dict, results_file)
            print('Results were saved to {}!'.format(self.configuration_path))
        return results

    def choose_max(self, validation_data, validation_labels):
        # validation data and validation labels are given as input in order to call evaluation function
        # We get results list and we choose the model which has maximum accuracy
        results = self.evaluation(validation_data, validation_labels)
        max_result = max(results, key=lambda item: item[1])
        print('The best accuracy was obtained from the {}!'.format(max_result[0]))

        return max_result

    def split_return_aug_types(self, model_name):
        # This function is for splitting augmentation types from the model name. It is used in order to make the code
        # run automatically
        temp_name = model_name[:-3]
        delete_index = len('model_')
        result = temp_name[delete_index:]
        aug_types = []

        if 'together' in result:
            aug_types.append(result)
        elif 'without_augmentation' in result:
            aug_types = []
        else:
            if '_' in result:
                aug_types = result.split('_')
            else:
                aug_types.append(result)
        return aug_types
