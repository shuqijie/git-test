from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from model import DAAN
from torch.utils import model_zoo
import numpy as np
from IPython import embed
import tqdm
import sys
sys.path.append("../")
from RAM import config
from RAM import main
# Training settings

parser = argparse.ArgumentParser(description='PyTorch DAAN')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default="E:\数据集\cd\OfficeHomeDataset_10072016/",
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="Clipart",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="Product",
                    help='the name of the test dir')
parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--num_class', default=65, type=int,
                    help='the number of classes')
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# def load_data():
#     source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
#     target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
#     target_test_loader  = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)

#     print(source_train_loader,target_train_loader,target_test_loader)
#     return source_train_loader, target_train_loader, target_test_loader

def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k is 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)


def train(epoch, model,feauture_source,feature_target):
    #total_progress_bar = tqdm.tqdm(desc='Train iter', total=args.epochs)
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)
    if args.diff_lr:
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.bottleneck.parameters()},
            {'params': model.domain_classifier.parameters()},
            {'params': model.dcis.parameters()},
            {'params': model.source_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)

    print_learning_rate(optimizer)

    global D_M, D_C, MU

    model.train()
    len_dataloader = len(feauture_source)
    DEV = DEVICE

    d_m = 0
    d_c = 0
    ''' update mu per epoch '''
    if D_M==0 and D_C==0 and MU==0:
        MU = 0.5
    else:
        D_M = D_M/len_dataloader
        D_C = D_C/len_dataloader
        MU = 1 - D_M/(D_M + D_C)
    #返回的形式是（b_t,y）前面是特征，后面是标签




    for batch_idx, (source_data, source_label) in tqdm.tqdm(enumerate(feauture_source),
                                    total=len_dataloader,
                                    desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
        p = float(batch_idx+1 + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        optimizer.zero_grad()
        source_data, source_label = source_data, source_label
        for target_data, target_label in feature_target:
            target_data, target_label = target_data, target_label
            break
        #调用DAANNet中的forward的代码，source_data, target_data分别为自己加载的源域，目标域的数据
        source_label = torch.tensor(source_label).to(DEV)
        source_data = torch.tensor(source_data).to(DEV)
        target_data = torch.tensor(target_data).to(DEV)
        # source_data = torch.from_numpy(source_data)
        # target_data = torch.from_numpy(target_data)
        out = model(source_data,target_data, source_label, DEV, alpha)
        s_output, s_domain_output, t_domain_output = out[0],out[1],out[2]
        s_out = out[3]
        t_out = out[4]

        #global loss
        sdomain_label = torch.zeros(args.batch_size).long().to(DEV)
        err_s_domain = F.nll_loss(F.log_softmax(s_domain_output.reshape(1,-1), dim=0), sdomain_label)
        tdomain_label = torch.ones(args.batch_size).long().to(DEV)
        err_t_domain = F.nll_loss(F.log_softmax(t_domain_output.reshape(1,-1), dim=0), tdomain_label)

        #local loss
        loss_s = 0.0
        loss_t = 0.0
        tmpd_c = 0
        for i in range(args.num_class):
            loss_si = F.nll_loss(F.log_softmax(s_out[i].reshape(1,-1), dim=0), sdomain_label)
            loss_ti = F.nll_loss(F.log_softmax(t_out[i].reshape(1,-1), dim=0), tdomain_label)
            loss_s += loss_si
            loss_t += loss_ti
            tmpd_c += 2 * (1 - 2 * (loss_si + loss_ti))
        tmpd_c /= args.num_class

        d_c = d_c + tmpd_c.cpu().item()

        global_loss = 0.05*(err_s_domain + err_t_domain)
        local_loss = 0.01*(loss_s + loss_t)

        d_m = d_m + 2 * (1 - 2 * global_loss.cpu().item())

        join_loss = (1 - MU) * global_loss + MU * local_loss
        soft_loss = F.nll_loss(F.log_softmax(s_output.reshape(1,-1), dim=0), source_label)
        if args.gamma == 1:
            gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        if args.gamma == 2:
            gamma = epoch /args.epochs
        loss = soft_loss + join_loss
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('\nLoss: {:.6f},  label_Loss: {:.6f},  join_Loss: {:.6f}, global_Loss:{:.4f}, local_Loss:{:.4f}'.format(
                loss.item(), soft_loss.item(), join_loss.item(), global_loss.item(), local_loss.item()))
        #total_progress_bar.update(1)
    D_M = np.copy(d_m).item()
    D_C = np.copy(d_c).item()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data, data, target, DEVICE)
            s_output = out[0]
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


if __name__ == '__main__':
    model = DAAN.DAANNet(num_classes=args.num_class, base_net='ResNet50').to(DEVICE)
    correct = 0
    D_M = 0
    D_C = 0
    MU = 0
    for epoch in range(1, args.epochs + 1):
        #加载数据集，分别是：source_dir，test_dir,test_dir目录下
        config, unparsed = config.get_config()
        feauture_source, feature_target = main.main(config)
        train(epoch, model, feauture_source,feature_target)
        t_correct = test(model, feature_target)
        if t_correct > correct:
            correct = t_correct
        print("%s max correct:" % args.test_dir, correct)
        print(args.source_dir, "to", args.test_dir)
