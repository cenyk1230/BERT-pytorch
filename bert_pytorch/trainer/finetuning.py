import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import numpy as np
from sklearn.metrics import f1_score

from ..model import BERT, CLS_MODEL
from .optim_schedule import ScheduledOptim

import tqdm


class FineTuningTrainer:
    """

    """

    def __init__(self, bert: BERT, hidden: int, class_size : int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.00, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, data_name=""):
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
        self.model = CLS_MODEL(bert, hidden, class_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim = Adam(self.model.parameters(), lr=lr)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss()

        self.log_freq = log_freq

        self.data_num = data_name

        self.writer = SummaryWriter()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        ret = self.iteration(epoch, self.test_data, train=False)
        self.model.train()
        return ret

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

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
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            else:
                # print(output[:,1])
                # print(data["bert_label"])
                y_scores.append(output[:,1].data.cpu().numpy())
                y_labels.append(data["bert_label"].cpu().numpy())

            # prediction accuracy
            correct = output.argmax(dim=-1).eq(data["bert_label"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["bert_label"].nelement()

            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }

                data_iter.write(str(post_fix))

                if str_code == "train":
                    self.writer.add_scalar(self.data_num + "_loss/train", loss, epoch * len(data_iter) + i)
                    self.writer.add_scalar(self.data_num + "_accu/train", correct / data["is_next"].nelement(), epoch * len(data_iter) + i)

        if str_code == "test":
            self.writer.add_scalar(self.data_num + "_loss/test", avg_loss / len(data_iter), epoch)
            self.writer.add_scalar(self.data_num + "_accu/test", total_correct * 100.0 / total_element, epoch)

            # y_scores = torch.cat(y_scores).cpu().numpy()
            # y_labels = torch.cat(y_labels).cpu().numpy()

            y_scores = np.concatenate(y_scores)
            y_labels = np.concatenate(y_labels)

            y_sorts = np.argsort(y_scores)[::-1]
            y_preds = np.zeros_like(y_labels, dtype=np.int)
            y_preds[y_sorts[:np.sum(y_labels)]] = 1

            f1 = f1_score(y_labels, y_preds)
            self.writer.add_scalar(self.data_num + "_f1/test", f1, epoch)

            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
                  total_correct * 100.0 / total_element, "f1=", f1)

            tp = np.sum((y_labels == 1) * (y_preds == 1))
            fp = np.sum((y_labels == 0) * (y_preds == 1))
            fn = np.sum((y_labels == 1) * (y_preds == 0))

            return f1, tp, fp, fn

        else:
            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
                  total_correct * 100.0 / total_element)


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
