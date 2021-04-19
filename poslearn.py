from collections import Counter
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

MAX_SENTENCE_LENGTH = 100
BATCH_SIZE = 100
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
NUM_OF_LAYERS = 1
NUM_OF_EPOCHS = 20
DROPOUT_RATE = 0

PADDING_INDEX = 0
UNKNOWN_INDEX = 1

# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
class TaggerLSTM(nn.Module):
    def __init__(self, vocab_size, output_size,
                 embedding_size=EMBEDDING_SIZE,
                 hidden_size=HIDDEN_SIZE,
                 padding_idx=PADDING_INDEX,
                 num_of_layers=NUM_OF_LAYERS,
                 bidirectional=True,
                 dropout=DROPOUT_RATE):
        super(TaggerLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            bidirectional=bidirectional,
                            num_layers=num_of_layers,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, texts):
        # texts: (seq_len, batch)
        embedded = self.embedding(texts)    # embedded: (seq_len, batch, input_size)
        outputs, _ = self.lstm(embedded)    # outputs: (seq_len, batch, num_directions * hidden_size)
        pred = self.fc(outputs)             # pred: (seq_len, batch, output_size)

        return pred

class Sentence(Dataset):
    def __init__(self, sentences, tags, word_to_idx, tag_to_idx):
        self.len = len(sentences)
        self.X = [convert_seq(s, word_to_idx) for s in sentences]
        self.Y = [convert_seq(s, tag_to_idx) for s in tags]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.len

def import_tagged_data(data_path, max_length=MAX_SENTENCE_LENGTH):
    all_sentences = []
    all_tags = []
    word_counter = Counter()
    tag_set = set()

    with open(data_path, mode='r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            words, tags = [], []
            words_w_tags = line.split()
            for word_w_tag in words_w_tags:
                word, tag = word_w_tag.split('/')
                words.append(word.lower())
                tags.append(tag.upper())

            if len(words) > max_length:
                words = words[: max_length]
                tags = tags[: max_length]
            all_sentences.append(words)
            all_tags.append(tags)

            word_counter.update(words)
            tag_set.update(set(tags))
            line = f.readline()

    top1000 = word_counter.most_common(1000)
    word_to_idx = {w: i + 2 for i, (w, freq) in enumerate(top1000)}
    word_to_idx['<PADDING>'], word_to_idx['<UNKNOWN>'] = PADDING_INDEX, UNKNOWN_INDEX

    tag_to_idx = {t: i + 2 for i, t in enumerate(tag_set)}
    tag_to_idx['<PADDING>'], tag_to_idx['<UNKNOWN>'] = PADDING_INDEX, UNKNOWN_INDEX

    return all_sentences, all_tags, word_to_idx, tag_to_idx

# return a tensor of the word indices of a word sequence. e.g. [0, 2, 1, 3] for a 4 word sequence
def convert_seq(seq, to_idx):
    indices = [to_idx[w] if w in to_idx else to_idx['<UNKNOWN>'] for w in seq]
    return torch.from_numpy(np.array(indices))

# customized collate function to be passed into dataloader
def my_collate(batch):

    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = pad_sequence(inputs)
    targets = pad_sequence(targets)
    return [inputs, targets]

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

def calculate_f1(predictions, true_tags, padding_index=PADDING_INDEX):
    # predictions: (seq_len * batch, output_size)
    # true_tags: (seq_len * batch)
    final_pred = predictions.argmax(dim=1)      # (seq_len * batch)

    # remove paddings' results
    final_pred = final_pred[true_tags != PADDING_INDEX]
    true_tags = true_tags[true_tags != PADDING_INDEX]

    macro_f1 = f1_score(true_tags, final_pred, average='macro')

    return macro_f1

def train(model, tr_dl, optimizer, criterion):
    epoch_loss = 0

    model.train()

    for train_step, (sentences, tags) in enumerate(tr_dl):
        # sentences: (seq_len, batch)
        # tags: (seq_len, batch)
        optimizer.zero_grad()
        predictions = model(sentences)  # (seq_len, batch, output_size)

        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        loss = criterion(predictions, tags.type(torch.LongTensor)) # criterion ignores the padding index
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(tr_dl)

def validate(model, va_dl, criterion):
    epoch_loss = 0

    model.eval()

    predictions_all = torch.tensor([])
    tags_all = torch.tensor([])

    with torch.no_grad():
        for val_step, (sentences, tags) in enumerate(va_dl):
            predictions = model(sentences)  # (seq_len, batch, output_size)

            predictions = predictions.view(-1, predictions.shape[-1])   # (seq_len*batch, output_size)
            tags = tags.view(-1)

            predictions_all = torch.cat((predictions_all, predictions))
            tags_all = torch.cat((tags_all, tags))

            loss = criterion(predictions, tags.type(torch.LongTensor))

            epoch_loss += loss.item()

        # calculate macro f1 score on validation data
        val_f1_score = calculate_f1(predictions_all, tags_all)

    return epoch_loss / len(va_dl), val_f1_score

def main(tr_data_path, va_data_path):

    tr_sents, tr_tags, word_to_idx, tag_to_idx = import_tagged_data(tr_data_path)
    va_sents, va_tags, _, _ = import_tagged_data(va_data_path)

    tr_ds = Sentence(tr_sents, tr_tags, word_to_idx, tag_to_idx)
    va_ds = Sentence(va_sents, va_tags, word_to_idx, tag_to_idx)

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)
#    tr_features, tr_labels = next(iter(tr_dl))


    model = TaggerLSTM(len(word_to_idx), len(tag_to_idx))
    model.apply(init_weights)
    model.embedding.weight.data[PADDING_INDEX] = torch.zeros(EMBEDDING_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_INDEX)

    for e in range(NUM_OF_EPOCHS):
        print(f'Epoch {e + 1} ')
        train_loss = train(model, tr_dl, optimizer, criterion)
        val_loss, val_f1_score = validate(model, va_dl, criterion)

        print(f'\tTraining loss: {train_loss}')
        print(f'\tValidation loss: {val_loss} | Validation F1 score: {val_f1_score}')




    return

if __name__ == '__main__':
    training_data_path = 'train.txt'
    dev_data_path = 'dev.txt'
    if len(sys.argv) == 3:
        training_data_path = sys.argv[1]
        dev_data_path = sys.argv[2]
    main(training_data_path, dev_data_path)