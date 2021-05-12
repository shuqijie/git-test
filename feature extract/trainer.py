import os
import time
import shutil
import pickle

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from RAM.model import RecurrentAttention
from RAM.utils import AverageMeter


class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config
        # if config.use_gpu and torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else:
        self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader)
            self.num_valid = len(self.valid_loader)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model，建立论文的模型
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        #将模型加载到cuda里面运行
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.init_lr
        )
        # 学习率动态调整的函数
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience
        )

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):
            #最后输出的信息相关代码
            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch，训练一个epoch,修改代码在这里可以返回一个源域特征b_t给RAM的输入
            baseline_train,baseline_validate = self.train_one_epoch(epoch)
            return baseline_train,baseline_validate
            # evaluate on validation set,修改代码在这里可以返回一个目标域特征b_t给RAM的输入
            # baseline_validate = self.validate(epoch)
            # return baseline_train baseline_train 
            # # reduce lr if validation loss plateaus
            self.scheduler.step(-valid_acc)

            #返回true或者false,判断当前验证集的准确率是不是最好的
            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f}"
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += "[*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc))

            # check for improvement
            if not is_best:
                #没有改进的次数
                self.counter += 1
            if self.counter > self.train_patience:
                #没有改进有一段时间了，停止训练
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        #RAM模型
        self.model.train()

        # 初始化val = 0 self.avg = 0 self.sum = 0 self.count = 0
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        tic = time.time()
        train_bt = []
        valid_bt=[]
        with tqdm(total=self.num_train) as pbar:
            #train_loader训练集数据
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                # 让输出变成1x1x28x28
                # x=x.reshape(1,1,28,-1)
                # b = torch.rand(5376, 28)
                # x = torch.matmul(x, b)
                y = torch.ones(1).long().to(self.device)
                x, y = x.to(self.device), y.to(self.device)
                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state，初始化位置向量和隐状态
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    # forward pass through model，RAM的前行传播
                    print("l_t",l_t.shape)
                    print("h_t", h_t.shape)
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                    # store
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                # 在这里进行b_t调用将它给DAAN文件下的source参数
                
                log_pi.append(p)
                baselines.append(b_t)
                locs.append(l_t[0:9])

                # convert list to tensors and reshape
                # print(baselines)
                # baselines = torch.stack(baselines).transpose(1, 0)
                # log_pi = torch.stack(log_pi).transpose(1, 0)
                #将每一个训练集的bt特征装在一起
                train_bt.append((baselines,y))
                # calculate reward
                #每行的最大值
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                baselines = torch.Tensor(baselines)
                

            for i, (x, y) in enumerate(self.valid_loader):
                y = torch.ones(1).long().to(self.device)
                x, y = x.to(self.device), y.to(self.device)

                # duplicate M times
                x = x.repeat(self.M, 1, 1, 1)

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # extract the glimpses
                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                    # store
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                # 在这里进行b_t调用将它给DAAN文件下的target参数

                log_pi.append(p)
                baselines.append(b_t)

                # convert list to tensors and reshape
                # baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                #将每一个验证集的bt特征装在一起
                valid_bt.append((baselines,y))
                # # average
                # log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
                # log_probas = torch.mean(log_probas, dim=0)
                #
                # baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
                # baselines = torch.mean(baselines, dim=0)
                
            return train_bt,valid_bt


                

    # @torch.no_grad()
    # def validate(self, epoch):
    #     """Evaluate the RAM model on the validation set.
    #     """
    #     losses = AverageMeter()
    #     accs = AverageMeter()

       

    # @torch.no_grad()
    # def test(self):
    #     """Test the RAM model.

    #     This function should only be called at the very
    #     end once the model has finished training.
    #     """
    #     correct = 0

    #     # load the best checkpoint
    #     self.load_checkpoint(best=self.best)

    #     for i, (x, y) in enumerate(self.test_loader):
    #         x, y = x.to(self.device), y.to(self.device)

    #         # duplicate M times
    #         x = x.repeat(self.M, 1, 1, 1)

    #         # initialize location vector and hidden state
    #         self.batch_size = x.shape[0]
    #         h_t, l_t = self.reset()

    #         # extract the glimpses
    #         for t in range(self.num_glimpses - 1):
    #             # forward pass through model
    #             h_t, l_t, b_t, p = self.model(x, l_t, h_t)

    #         # last iteration
    #         h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)   
    #         return b_t

    # def save_checkpoint(self, state, is_best):
    #     """Saves a checkpoint of the model.

    #     If this model has reached the best validation accuracy thus
    #     far, a seperate file with the suffix `best` is created.
    #     """
    #     filename = self.model_name + "_ckpt.pth.tar"
    #     ckpt_path = os.path.join(self.ckpt_dir, filename)
    #     torch.save(state, ckpt_path)
    #     if is_best:
    #         filename = self.model_name + "_model_best.pth.tar"
    #         shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    # def load_checkpoint(self, best=False):
    #     """Load the best copy of a model.

    #     This is useful for 2 cases:
    #     - Resuming training with the most recent model checkpoint.
    #     - Loading the best validation model to evaluate on the test data.

    #     Args:
    #         best: if set to True, loads the best model.
    #             Use this if you want to evaluate your model
    #             on the test data. Else, set to False in which
    #             case the most recent version of the checkpoint
    #             is used.
    #     """
    #     print("[*] Loading model from {}".format(self.ckpt_dir))

    #     filename = self.model_name + "_ckpt.pth.tar"
    #     if best:
    #         filename = self.model_name + "_model_best.pth.tar"
    #     ckpt_path = os.path.join(self.ckpt_dir, filename)
    #     ckpt = torch.load(ckpt_path)

    #     # load variables from checkpoint从检查点加载变量
    #     self.start_epoch = ckpt["epoch"]
    #     self.best_valid_acc = ckpt["best_valid_acc"]
    #     self.model.load_state_dict(ckpt["model_state"])
    #     self.optimizer.load_state_dict(ckpt["optim_state"])

    #     if best:
    #         print(
    #             "[*] Loaded {} checkpoint @ epoch {} "
    #             "with best valid acc of {:.3f}".format(
    #                 filename, ckpt["epoch"], ckpt["best_valid_acc"]
    #             )
    #         )
    #     else:
    #         print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
