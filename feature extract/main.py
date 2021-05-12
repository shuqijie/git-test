
import torch
import sys

from RAM import utils
# import data_loader
from RAM import data_loader1 as data_loader
from RAM import trainer as tr
from RAM import config
kwargs = {'num_workers': 1, 'pin_memory': True}

def main(config):
    # 得到data,ckpt，log的路径
    utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    # instantiate data loaders
    if config.is_train:
        # 得到验证集
        dloader = data_loader.get_train_valid_loader(
            config.data_dir,
            config.batch_size,
            kwargs,
        )
    else:
        # 得到测试集
        dloader = data_loader.get_test_loader(
            config.data_dir, config.batch_size, kwargs,
        )

    trainer = tr.Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        feauture_source, feature_target = trainer.train()
    # or load a pretrained model and test
    else:
        pass
        # trainer.test()
    return feauture_source, feature_target
#
# if __name__ == "__main__":
#     config, unparsed = config.get_config()
#     main(config)
