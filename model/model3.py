from torchsummary import summary
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from model.model4 import resnet




class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  #
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU())

    def forward(self, input):
        return self.ConvNet(input)


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        #self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size*2 , output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        b, T, h = recurrent.size()
        recurrent = recurrent.reshape(b * T, h)
        output = self.linear(recurrent)  # batch_size x T x output_size
        output = output.view(b, T, -1)

        return output


class Model(nn.Module):
    def __init__(self, imgSize, inputChannel, outputChannel, rnnHiddenSize, numChars):

        super(Model, self).__init__()

        # Feature Extractor
        self.featureExtractor = VGG_FeatureExtractor(inputChannel, outputChannel)



        #b, c, h, w = self.featureExtractor(inputChannel,outputChannel).shape



        self.SequenceModel = nn.Sequential(
            nn.Linear(96, rnnHiddenSize),
            BidirectionalLSTM(rnnHiddenSize, rnnHiddenSize, rnnHiddenSize),
            BidirectionalLSTM(rnnHiddenSize, rnnHiddenSize, rnnHiddenSize),
            nn.Linear(rnnHiddenSize, numChars)
        )

    def forward(self, batch):

        batch = self.featureExtractor(batch)

        batch = batch.permute(0, 3, 1, 2)

        B = batch.size(0)
        T = batch.size(1)
        C = batch.size(2)
        H = batch.size(3)

        batch = batch.view(B, T, C * H)
        batch = self.SequenceModel(batch)
        batch = batch.permute(1, 0, 2)

        batch = batch.reshape(-1, 3 * 30)

        fc = nn.Linear(3 * 30, 3)

        batch=fc(batch)

        batch = F.log_softmax(batch, 1)

        return batch


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




