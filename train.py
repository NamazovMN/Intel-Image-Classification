import math
import pickle

import tensorflow as tf
from utilities import *
from tqdm import tqdm
from model import BuildModel, MySequentialModule, TransferModel
from sklearn.metrics import f1_score
from dataset import IntelSet
from typing import Any


class Train:
    def __init__(self, config_parameters: dict, train_set: IntelSet, dev_set: IntelSet):
        """
        Method is used as initializer for class
        :param config_parameters: required parameters for the project
        :param train_set: train dataset object
        :param dev_set: dev dataset object (which is actually test set, since development set was not provided)
        """
        self.configuration = self.set_configuration(config_parameters, train_set, dev_set)
        self.model = self.set_model()
        self.optimizer = self.set_optimizer()
        self.loss = self.set_loss()

    def set_configuration(self, parameters: dict, train_data_object: IntelSet, dev_data_object: IntelSet) -> dict:
        """
        Method is utilized for extracting task-specific parameters from all required parameters and add additional
        parameters to the parameters object
        :param parameters: dictionary that contains all task-specific parameters
        :param train_data_object: train data object
        :param dev_data_object: dev data object
        :return: dictionary includes all required parameters for this specific task
        """
        train_results = os.path.join('train_results', f'experiment_{parameters["exp_num"]}')

        check_dir(train_results)

        self.modify_config(parameters, train_results)
        checkpoints_dir = os.path.join(train_results, 'checkpoints')

        check_dir(checkpoints_dir)
        model_build = BuildModel(parameters)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data_object['data'], train_data_object['label']))
        dev_dataset = tf.data.Dataset.from_tensor_slices((dev_data_object['data'], dev_data_object['label']))
        num_batches = math.ceil(len(train_data_object['label']) / parameters['batch_size'])
        dev_steps = math.ceil(len(dev_data_object['label']) / parameters['batch_size'])
        return {
            'lr': parameters['lr'],
            'dev_steps': dev_steps,
            'num_batches': num_batches,
            'model_obj': model_build,
            'train_results': train_results,
            'epochs': parameters['num_epochs'],
            'image_dim': parameters['image_dim'],
            'train_dataset': train_dataset.batch(parameters['batch_size']),
            'dev_dataset': dev_dataset.batch(parameters['batch_size']),
            'ckpt_dir': checkpoints_dir,
            'resume_training': parameters['resume_training'],
            'labels_dict': train_data_object.configuration['labels_dict'],
            'es_apply': parameters['es_apply'],
            'es_monitor': parameters['es_monitor'],
            'es_limit': parameters['es_limit'],
            'transfer_weights': 'imagenet' if parameters['transfer_weights'] else None,
            'transfer_learning': parameters['transfer_learning']

        }

    @staticmethod
    def modify_config(parameters: dict, train_results_path: str) -> None:
        """
        Method is used to save and update project configuration parameters
        :param parameters: dictionary contains all required parameters for the project
        :param train_results_path: path to the folder for train results (experiment folder)
        :return: None
        """
        config_data = os.path.join(train_results_path, 'config_parameters.pickle')

        if parameters['resume_training']:
            with open(config_data, 'rb') as config_file:
                configuration = pickle.load(config_file)
            train_data = os.path.join(train_results_path, 'results.pickle')
            with open(train_data, 'rb') as tr_res:
                tr_data = pickle.load(tr_res)
            epoch = max(tr_data.keys())
            configuration[epoch + 1] = parameters
        else:
            configuration = {0: parameters}

        with open(config_data, 'wb') as config_file:
            pickle.dump(configuration, config_file)

    def set_optimizer(self) -> tf.keras.optimizers.Adam:
        """
        Method is utilized for setting the optimizer for the model
        :return: optimizer object
        """
        return tf.keras.optimizers.Adam(learning_rate=self.configuration['lr'])

    @staticmethod
    def set_loss() -> tf.keras.losses.CategoricalCrossentropy:
        """
        Method is utilized for setting the loss function for the model
        :return: loss object
        """
        return tf.keras.losses.CategoricalCrossentropy()

    def set_model(self) -> Any:
        """
        Method is utilized for setting the model for the project, which can either be sequential model or transfer
        learning model
        :return: classifier model
        """
        if self.configuration['transfer_learning']:
            model = TransferModel(self.configuration)
        else:
            model = MySequentialModule(self.configuration['model_obj'])

        return model

    @staticmethod
    def early_stopping(best: float, current: float, wait: int) -> tuple:
        """
        Method is used as custom early stopping callback for training phase
        :param best: the recent best value for specific metric
        :param current: the current value for specific metric
        :param wait: waiting step which increases when best value is not updated
        :return: tuple which contains new best value and new wait step
        """
        new_best = current if current < best else best
        new_wait = 0 if current < best else wait + 1
        return new_best, new_wait

    def train_step(self, batch: tuple) -> tuple:
        """
        Method is utilized for training the model for one batch
        :param batch: where the first element is input data and the other is tensor of labels
        :return: tuple object contains loss value, accuracy value and number of data for one batch
        """
        with tf.GradientTape() as tape:
            output = self.model(batch[0], training=True)
            loss_value = self.loss(batch[1], output)

        grads = tape.gradient(loss_value, self.model.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.model.trainable_weights))
        accuracy, num_data, _, _ = self.compute_accuracy(output, batch[1])
        return loss_value, accuracy, num_data

    @staticmethod
    def compute_accuracy(predictions: tf.Tensor, labels: tf.Tensor) -> tuple:
        """
        Method is utilized for accuracy computation
        :param predictions: tensor which is output batch
        :param labels: tensor which is target batch
        :return: tuple that contains number of correct predictions, number of data per batch, predictions and targets in
                form of lists
        """

        preds = tf.argmax(predictions, axis=-1).numpy().tolist()
        targets = tf.argmax(labels, axis=-1).numpy().tolist()
        correct = sum([1 for t, p in zip(targets, preds) if t == p])
        return correct, len(preds), preds, targets

    def train_epoch(self) -> None:
        """
        Training the model for provided number of epochs
        :return: None
        """
        epoch_choice = 0
        best = 1000
        wait = 0
        if self.configuration['resume_training']:
            epoch_choice = self.load_model() + 1

        for epoch in range(epoch_choice, self.configuration['epochs']):
            epoch_loss = 0
            num_data = 0
            accuracy = 0
            ti = tqdm(self.configuration['train_dataset'], total=self.configuration['num_batches'],
                      desc=f'epoch: {epoch}', leave=True)
            for batch in ti:
                loss, acc, num_step = self.train_step(batch)
                epoch_loss += loss
                accuracy += acc
                num_data += num_step

                ti.set_description(f'Training => Epoch: {epoch}, '
                                   f'Train Loss: {epoch_loss / self.configuration["num_batches"]: .4f},'
                                   f' Train Accuracy: {accuracy / num_data: .4f}')
            dev_loss, dev_acc, dev_f1, _, _ = self.evaluate(epoch)
            train_results_dict = {
                'train_loss': epoch_loss / self.configuration['num_batches'],
                'dev_loss': dev_loss,
                'train_acc': accuracy / num_data,
                'dev_acc': dev_acc,
                'f1_dev': dev_f1
            }

            self.save_parameters(train_results_dict, epoch)
            print(f'Train and Evaluation Results, along with model parameters were saved successfully for {epoch}!')
            print(f'{20 * "<"} {20 * ">"}')

            if self.configuration['es_apply']:
                metric = train_results_dict[self.configuration['es_monitor']]
                best, wait = self.early_stopping(best, metric, wait)
                if self.configuration['es_limit'] == wait:
                    break

    def evaluate(self, epoch: int) -> tuple:
        """
        Method is utilized for evaluating the model for specific epoch
        :param epoch: current epoch for the model
        :return: tuple that contains average development loss, accuracy (in percents) and f1 score for the epoch
        """
        eval_ti = tqdm(self.configuration['dev_dataset'], total=self.configuration['dev_steps'],
                       desc=f'epoch: {epoch}', leave=True)
        dev_loss = 0
        dev_acc = 0
        outputs = list()
        targets = list()
        num_data = 0
        for batch in eval_ti:
            with tf.GradientTape():
                output = self.model(batch[0], training=False)
                loss_value = self.loss(batch[1], output)
            acc, num_steps, predictions, labels = self.compute_accuracy(output, batch[1])
            num_data += num_steps
            outputs.extend(predictions)
            targets.extend(labels)
            dev_loss += loss_value
            dev_acc += acc
            eval_ti.set_description(
                f"Evaluation => Epoch: {epoch}, Dev Loss: {dev_loss / self.configuration['dev_steps']: .4f}, "
                f"Dev Accuracy: {dev_acc / num_data :.4f}")
        f1 = f1_score(targets, outputs, average='macro')
        print(f'F1 score for {epoch}: {f1}')
        return dev_loss / self.configuration['dev_steps'], dev_acc / num_data, f1, outputs, targets

    def save_parameters(self, result_dict: dict, epoch: int) -> None:
        """
        Method is utilized for saving training results and model parameters after training the specific epoch
        :param result_dict: dictionary contains training results for epoch
        :param epoch: integer specifies current epoch
        :return: None
        """
        file_name = os.path.join(self.configuration['train_results'], 'results.pickle')
        result = dict()
        if not os.path.exists(file_name):
            result[epoch] = result_dict
        else:
            with open(file_name, 'rb') as data:
                result = pickle.load(data)
            result[epoch] = result_dict
        with open(file_name, 'wb') as data:
            pickle.dump(result, data)

        checkpoint_path = os.path.join(self.configuration['ckpt_dir'],
                                       f"model_{epoch}_f1_{result_dict['f1_dev']: .3f}_"
                                       f"dl_{result_dict['dev_loss']: .3f}_"
                                       f"tl_{result_dict['train_loss']: .3f}_"
                                       f"da_{result_dict['dev_acc']: .3f}")
        self.model.model.save(checkpoint_path)

    def get_epoch(self, metric: str = 'f1_dev', is_best: bool = False) -> int:
        """
        Method is utilized for getting the specific epoch number
        :param metric: specifies according to which metric the best epoch will be defined
        :param is_best: specifies whether the best epoch will be collected (True) or the last epoch (False)
        :return: integer specifies the chosen epoch
        """
        with open(os.path.join(self.configuration['train_results'], 'results.pickle'), 'rb') as res_file:
            train_results = pickle.load(res_file)
        results = {epoch: data[metric] for epoch, data in train_results.items()}
        if is_best:
            epoch = max(results, key=results.get)
        else:
            epoch = max(results.keys())
        return epoch

    def load_model(self, metric: str = 'f1_dev', is_best: bool = False) -> int:
        """
        Method is utilized for loading the model according to choice of the scenario
        :param metric: specifies according to which metric the best epoch will be defined
        :param is_best: specifies whether the best epoch will be collected (True) or the last epoch (False)
        :return: integer value to specify epoch number
        """
        epoch = self.get_epoch(metric, is_best)
        path = str()
        for each in os.listdir(self.configuration['ckpt_dir']):
            if f'model_{epoch}_' in each:
                path = os.path.join(self.configuration['ckpt_dir'], each)
                break
        if not path:
            raise FileNotFoundError('No path was found, you need to train the model first!')

        self.model.model = tf.keras.models.load_model(
            path, custom_objects={"IntelSetModel": self.model.model}
        )
        return epoch
