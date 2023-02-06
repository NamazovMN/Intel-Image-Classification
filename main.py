from utilities import *
from src.old.dataset import IntelSet
from train import Train
from collect_data import CollectData
from dataset import IntelSet
from statistics import Statistics

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def __main__():
    parameters = collect_parameters()
    cd = CollectData(parameters)
    train_data = IntelSet(parameters, cd, 'train', True)
    dev_data = IntelSet(parameters, cd, 'test', True)
    st = Statistics(parameters, cd, True, train_data, dev_data)
    st.get_statistics()
    if parameters['train']:
        trainer = Train(parameters, train_data, dev_data)
        trainer.train_epoch()
    st.get_statistics(False)


if __name__ == '__main__':
    __main__()
