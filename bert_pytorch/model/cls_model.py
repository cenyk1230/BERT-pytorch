import torch.nn as nn

from .bert import BERT


class CLS_MODEL(nn.Module):
    """
    Classification Model
    """

    def __init__(self, bert: BERT, hidden, class_size):
        """
        :param bert: BERT model which has been trained
        :param hidden: output size of BERT model
        :param class_size: total class size
        """

        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(hidden, class_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.softmax(self.linear(x))
