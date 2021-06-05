import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

##
# adapted from: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
##
class EncoderCNN(nn.Module):
    def __init__(self, hidden_size):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.hidden_size = hidden_size
        resnet = models.resnet18(pretrained=True)

        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 98)
        resnet.load_state_dict(torch.load("out/models/icon_image_classifier.pkl"))
        self.resnet = resnet
        self.linear = nn.Linear(98, self.hidden_size)
        self.bn = nn.BatchNorm1d(self.hidden_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers, max_seq_length=5):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.fc = nn.Linear(input_size*2, input_size)

    def forward(self, image_features, captions, lengths, context_features):
        """Decode image feature vectors and generates captions."""
        features = self.fc(torch.cat((image_features, context_features), 1))
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # batch*(max_len+1)*input
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # (batch_sum_seq_len X embedding_dim) packing discards <eos>

        hiddens, _ = self.lstm(packed)  # data: (batch_sum_seq_len X hidden_dim)

        outputs = self.linear(hiddens.data)
        return outputs

    def sample(self, image_features, context_features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        features = self.fc(torch.cat((image_features, context_features), 1))
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dropout_p=0.1, max_length=100): #vocab_size outputsize
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.input_size = input_size
        # self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, word, hidden, context):
        embeddings = self.embedding(word)

        embedded = embeddings.view(1, 1, -1) #1,1,100
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        context = torch.cat((context, torch.zeros(self.max_length-context.shape[0], context.shape[1]).to(self.device)), dim=0)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 context.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden) #, batch_first=True

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def sample(self, context, hidden=None):
        sampled_ids = []
        inputs = torch.tensor(1, device=self.device)
        if not hidden:
            hidden = self.initHidden()#image.unsqueeze(0)
        for i in range(5):
            out, hidden, weights = self.forward( inputs.unsqueeze(0).unsqueeze(0), hidden, context.squeeze(0))
            _, predicted = out.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = predicted[0]
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class EncoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, weights_matrix, glove_dim, device, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, contexts, con_lengths):  # b*2*8(words)
        output, (hidden, cell) = self.lstm(contexts)
        return output, hidden.squeeze()

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
