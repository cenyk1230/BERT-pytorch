import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import numpy as np
from sklearn.metrics import f1_score

from ..model import BERT, MultiLabelClassificationModel
from ..model.utils.scoring import construct_indicator
from .optim_schedule import ScheduledOptim

import tqdm


class FineTuningTrainer:
    """

    """

    def __init__(self, bert: BERT, hidden: int, class_size : int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None, valid_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.00, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, logger_name: str = None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the classification model, with BERT model
        self.model = MultiLabelClassificationModel(bert, hidden, class_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.valid_data = valid_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim = Adam(self.model.parameters(), lr=lr)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.BCELoss()

        self.log_freq = log_freq

        self.writer = SummaryWriter(logger_name)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        ret = self.iteration(epoch, self.test_data, str_code="test")
        self.model.train()
        return ret

    def valid(self, epoch):
        self.model.eval()
        ret = self.iteration(epoch, self.valid_data, str_code="valid")
        self.model.train()
        return ret

    def iteration(self, epoch, data_loader, str_code="train"):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param str_code: string value of is train or test or valid
        :return: None or score or loss
        """
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        y_scores = []
        y_labels = []

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            output = self.model.forward(data["bert_input"], data["segment_label"])[:,0,:]

            # 2-1. NLL(negative log likelihood) loss of classification result
            loss = self.criterion(output, data["bert_label"])

            # 3. backward and optimization only in train
            if str_code == "train":
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            else:
                y_scores.append(output.data.cpu().numpy())
                y_labels.append(data["bert_label"].cpu().numpy())

            avg_loss += loss.item()

            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item()
                }

                data_iter.write(str(post_fix))

                if str_code == "train":
                    y = data["bert_label"].cpu().numpy()
                    y_pred = construct_indicator(output.data.cpu().numpy(), y)
                    self.writer.add_scalar('finetune/train_loss', loss, epoch * len(data_iter) + i)
                    self.writer.add_scalar('finetune/train_accu', (y_pred == y).mean(axis=1).mean(), epoch * len(data_iter) + i)

        if str_code == "test" or str_code == "valid":
            y_score = np.concatenate(y_scores)
            y_label = np.concatenate(y_labels)
            y_pred = construct_indicator(y_score, y_label)
            self.writer.add_scalar(f'finetune/{str_code}_loss', avg_loss / len(data_iter), epoch)
            self.writer.add_scalar(f'finetune/{str_code}_accu', (y_pred == y_label).mean(axis=1).mean(), epoch)

            mi = f1_score(y_label, y_pred, average="micro")
            ma = f1_score(y_label, y_pred, average="macro")
            self.writer.add_scalar(f"score/{str_code}_micro_f1", mi, epoch)
            self.writer.add_scalar(f"score/{str_code}_macro_f1", ma, epoch)

            if str_code == "test":
                return mi, ma
            elif str_code == "valid":
                return avg_loss / len(data_iter)


    def save(self, epoch, file_path="output/finetuning.model"):
        """
        Saving the current Fine-Tuning model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
