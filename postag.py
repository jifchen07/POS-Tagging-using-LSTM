import sys
import pickle
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence

from poslearn import TaggerLSTM

MAX_SENTENCE_LENGTH = 100

def import_data(data_path, max_length=MAX_SENTENCE_LENGTH):
    all_sentences = []
    with open(data_path, mode='r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            words = line.split()
            if len(words) > max_length:
                words = words[: max_length]
            all_sentences.append(words)
            line = f.readline()

    return all_sentences

def convert_data(data, word_to_idx):
    matrix = [torch.from_numpy(np.array([word_to_idx[w] if w in word_to_idx
                                         else word_to_idx['<UNKNOWN>'] for w in seq]))
              for seq in data]      # (batch, various seq_len)

    return pad_sequence(matrix)     # (seq_len, batch)  84x2000

def write_output(orig_sentences, final_pred, tag_to_idx, output_path):
    # final_pred: (seq_len, batch)
    assert(len(orig_sentences) == final_pred.shape[1])  # number of sentences

    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, s in enumerate(orig_sentences):
            str_builder = []
            for j, w in enumerate(s):
                str_builder.append(w + '/' + idx_to_tag[int(final_pred[j][i])])
            f.write(' '.join(str_builder) + '\n')

def main(va_data_path, output_path):
    model = torch.load('model_params/model')
    with open('model_params/word_to_idx.pickle', 'rb') as f:
        word_to_idx = pickle.load(f)
    with open('model_params/tag_to_idx.pickle', 'rb') as f:
        tag_to_idx = pickle.load(f)

    sentences = import_data(va_data_path)
    sentences_padding = convert_data(sentences, word_to_idx)

    model.eval()

    predictions = model(sentences_padding)  # (seq_len, batch, output_size)
    final_pred = predictions.argmax(dim=2)    # (seq_len, batch)

    write_output(sentences, final_pred, tag_to_idx, output_path)

if __name__ == '__main__':
    validation_data_path = 'dev_notag.txt'
    tagged_output_path = 'dev_tagged.txt'
    if len(sys.argv) == 3:
        validation_data_path = sys.argv[1]
        tagged_output_path = sys.argv[2]
    main(validation_data_path, tagged_output_path)