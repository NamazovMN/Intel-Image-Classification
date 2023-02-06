import argparse
import os


def get_parameters() -> argparse.Namespace:
    """
    Function is utilized for collecting user-defined parameters
    :return: Namespace object for user-defined parameters
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, required=False, default='data',
                        help='Specifies the path to the dataset directory')
    parser.add_argument('--image_dim', type=int, required=False, default=75,
                        help='Specifies the width and height of the image data')
    parser.add_argument('--batch_size', type=int, required=False, default=32,
                        help='Specifies batch size for the training and evaluation')
    parser.add_argument('--dropout', type=float, required=False, default=[0.5, 0.4], nargs='+',
                        help='Dropout rates for FCN model')
    parser.add_argument('--lr', default=0.0001, required=False, type=float,
                        help='Specifies learning rate for the model')
    parser.add_argument('--bn_lin', type=int, required=False, default=[0, 0], nargs='+',
                        help='Dropout rates for FCN model')
    parser.add_argument('--dense', type=int, required=False, default=[64, 24], nargs='+',
                        help='Dense Layer input dimensions for FCN model')
    parser.add_argument('--bn', type=int, required=False, default=[0, 0, 0], nargs='+',
                        help='Batch Normalization layers for CNN')
    parser.add_argument('--conv', type=int, required=False, default=[64, 128, 256], nargs='+',
                        help='Specifies number convolutional kernels as output of each convolutional layer')

    parser.add_argument('--kernels', type=int, required=False, default=[5, 4, 3], nargs='+',
                        help='Specifies dimensions of kernels at each layer of CNN')
    parser.add_argument('--strides', type=int, required=False, default=[1, 1, 1], nargs='+',
                        help='Specifies stride dimension of kernels at each layer of CNN')
    parser.add_argument('--mp_kernels', type=int, required=False, default=[4, 3, 2], nargs='+',
                        help='Specifies kernel dimensions at each MaxPooling layer')
    parser.add_argument('--mp_strides', type=int, required=False, default=[1, 1, 1], nargs='+',
                        help='Specifies stride dimensions at each MaxPooling layer')
    parser.add_argument('--dropout_cnn', type=float, required=False, default=[0.3, 0.3, 0.3], nargs='+',
                        help='Dropout rates for FCN model')
    parser.add_argument('--exp_num', type=int, required=False, default=13,
                        help='Specifies experiment number')
    parser.add_argument('--num_epochs', type=int, required=False, default=3,
                        help='Specifies number of epochs that model will be trained')

    parser.add_argument('--resume_training', action='store_true', default=False, required=False,
                        help='Specifies training will be continued from the last epoch (True) or not (False)')
    parser.add_argument('--transfer_learning', action='store_true', default=False, required=False,
                        help='Specifies whether transfer learning will be activated (True) or not (False)')
    parser.add_argument('--transfer_weights', action='store_true', default=False, required=False,
                        help='Specifies whether imagenet weights will be used (True) or not')
    parser.add_argument('--transfer_model', type=str, default='VGG', required=False,
                        help='Specifies the transfer learning model')

    parser.add_argument('--es_apply', action='store_true', default=False, required=False,
                        help='Specifies whether early stopping will be activated or not')
    parser.add_argument('--es_monitor', type=str, default='dev_loss', required=False,
                        help='Specifies metric for early stopping check')
    parser.add_argument('--es_limit', type=int, default=5, required=False,
                        help='Specifies early stopping limit')
    parser.add_argument('--train', action='store_true', default=False, required=False,
                        help='Specifies whether training session will be activated or not')
    return parser.parse_args()


def collect_parameters() -> dict:
    """
    Function is utilized for transforming user-defined parameters into dictionary object
    :return: dictionary that includes all required parameters for the project
    """
    parameters = get_parameters()
    configuration = dict()
    for argument in vars(parameters):
        configuration[argument] = getattr(parameters, argument)

    return configuration


def check_dir(directory: str) -> None:
    """
    Function is utilized for checking the existence of the requested path. If it does not exist, it will be created
    :param directory: path to check its existence
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
