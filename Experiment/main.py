import tensorflow as tf
import pdb

from data_generators.data_generator import DataGenerator
from models.model import Model
from trainers.trainer import Trainer
from utils.utils import *

def main():
    # capture the config path from the run arguments
    # then process the json configration file

    # try:
    #     args = get_args()
    #     config = process_config(args.config)

    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    config, _ = get_config_from_json('configs/config.json')
    # create the experiments dirs
    # create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create instance of the model you want
    model = Model(config)
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    # logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = Trainer(sess, config, model, data)
    
    for epoch in range(150):
        losses, samples, likelihood, diversity = trainer.train_epoch()
        print(likelihood, diversity)
    # here you train your model
    # trainer.train()


if __name__ == '__main__':
    main()