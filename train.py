from __future__ import print_function

import os
from glob import glob
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable

from model import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0.0, 0.02)

class Trainer(object):
    def __init__(self, config, data_loader):
        # basic configuration
        self.config = config
        self.data_loader = data_loader

        self.num_gpu = config.num_gpu
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay

        self.hidden_size1 = config.hidden_size1
        self.hidden_size2 = config.hidden_size2

        self.load_path = config.load_path
        self.model_dir = config.model_dir

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.build_model()
        if self.load_path:
            self.load_model()

        # GPU control
        if self.num_gpu == 1:
            self.rnn_1.cuda()
            self.rnn_2.cuda()
            # self.mlp.cuda()
        elif self.num_gpu > 1:
            self.rnn_1 = nn.DataParallel(self.rnn_1.cuda(), device_ids=range(self.num_gpu))
            self.rnn_2 = nn.DataParallel(self.rnn_2.cuda(), device_ids=range(self.num_gpu))

    def build_model(self):
        self.rnn_1 = RNN_1(12, self.hidden_size1)
        # self.rnn_1 = RNN_1(12, self.hidden_size1, 1, True)
        self.rnn_2 = RNN_2(self.hidden_size1, self.hidden_size2)
        # self.rnn_1 = RNN_2(self.hidden_size1, self.hidden_size2, 1, True)

        self.rnn_1.apply(weights_init)
        self.rnn_2.apply(weights_init)

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'rnn1_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [int(os.path.basename(path.split('.')[1].split('_')[-1])) for path in paths]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        rnn1_filename = '{}/rnn1_{}.pth'.format(self.load_path, self.start_step)
        self.rnn_1.load_state_dict(
            torch.load(rnn1_filename, map_location=map_location))
        print("[*] RNN_1 network loaded: {}".format(rnn1_filename))

        rnn2_filename = '{}/rnn2_{}.pth'.format(self.load_path, self.start_step)
        self.rnn_2.load_state_dict(
            torch.load(rnn2_filename, map_location=map_location))
        print("[*] RNN_2 network loaded: {}".format(rnn2_filename))

    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out

    def save_model(self, step):
        print("[*] Save models to {}...".format(self.model_dir))
        torch.save(self.rnn_1.state_dict(), '{}/rnn1_{}.pth'.format(self.model_dir, step))
        torch.save(self.rnn_2.state_dict(), '{}/rnn2_{}.pth'.format(self.model_dir, step))

    def train(self):
        mse = nn.MSELoss()
        if self.num_gpu > 0:
            mse = mse.cuda()

        optimizer = torch.optim.Adam
        parameters = [self.rnn_1.parameters(), self.rnn_2.parameters()]
        optim_model = optimizer(parameters, lr=self.lr, betas=(self.beta1, self.beta2))

        data_loader = iter(self.data_loader)

        for step in trange(self.start_step, self.max_step):
            try:
                input, target = next(data_loader)
            except StopIteration:
                data_loader = iter(self.data_loader)
                input, target = next(data_loader)

            input = self._get_variable(input)
            target = self._get_variable(target)

            self.rnn_1.zero_grad()
            self.rnn_2.zero_grad()

            out = self.rnn_2(self.rnn_1(input))
            loss = mse(out, target)
            loss.backward()
            optim_model.step()

            if step % self.save_step == self.save_step - 1:
                self.save_model(step)

            if step % self.log_step == 0:
                print("[{}/{}] Loss: {:.4f}".format(step, self.max_step, loss))
