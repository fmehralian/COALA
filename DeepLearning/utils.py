from collections import Counter
import json
import re
import nltk
import numpy as np
import torchtext.vocab as vocab
from sklearn.cluster import KMeans
from wordsegment import load, segment
import unicodedata
import math

load()


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, glove_dim):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.glove_dim = glove_dim
        self.glove = vocab.GloVe(name='6B', dim=glove_dim)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def glove_weights(self):

        matrix_len = len(self)
        weights_matrix = np.zeros((matrix_len, self.glove_dim))
        words_found = 0

        for i, word in enumerate(self.word2idx.keys()):
            try:
                weights_matrix[i] = self.glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.glove_dim,))
        return weights_matrix

    def glove_weights_custom(self, input):

        matrix_len = len(input)
        weights_matrix = np.zeros((matrix_len, self.glove_dim))
        words_found = 0

        for i, word_id in enumerate(input):
            try:
                word = self.idx2word[word_id]
                weights_matrix[i] = self.glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.glove_dim,))
        return weights_matrix

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<UNK>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def xml_values(xml_context, field):
    if not field:
        return ""
    if "-" in field:
        member = field.split("-")[0]
        tag = field.split("-")[1]
        return xml_values_family(xml_context, member, tag)
    else:
        return xml_context[field]


def xml_values_family(xml_context, family_member, tag):
    if family_member == 'father':
        return xml_context['family'][family_member][tag]
    str = ""
    for sibling in xml_context['family'][family_member]:
        str += sibling[tag]
    return str


def get_frequent_concept(param, vocab):
    sorted_params = sorted(param, key=lambda item: item[1], reverse=True)
    data = []
    for word in sorted_params:
        data.append(vocab.glove[word[0]].numpy())
    data = np.array(data)
    n_c = int(0.05 * len(data))
    kmeans = KMeans(n_clusters=n_c, n_init=n_c)
    y_pred_kmeans = kmeans.fit_predict(data)
    concepts = []
    for c in range(n_c):
        indices = [i for i, x in enumerate(y_pred_kmeans) if x == c]
        freq_in_cluster = sum([sorted_params[x][1] for x in indices])
        if freq_in_cluster > 90:
            concepts.append([sorted_params[x][0] for x in indices])
    return concepts


def build_vocab(filepath, threshold, glove_dim, context_fields):
    """Build a simple vocabulary wrapper."""
    with open(filepath, "r") as f:
        labels = json.load(f)
    counter_content = Counter()
    counter_context = Counter()
    category_vocab = ['None']
    for element in labels:
        caption = element['content']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter_content.update(tokens)

        element['context']['android_id'] = element['android-id']
        context_words = get_context_textual(element['context'], context_fields)
        counter_context.update(set(context_words))

        if element['context']['category'] not in category_vocab:
            if element['context']['category']:
                category_vocab.append(element['context']['category'])


    # If the word frequency is less than 'threshold', then the word is discarded.
    words_content = [word for word, cnt in counter_content.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab_content = Vocabulary(glove_dim)

    vocab_content.add_word('<PAD>')
    vocab_content.add_word('<SOS>')
    vocab_content.add_word('<EOS>')
    vocab_content.add_word('<UNK>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words_content):
        vocab_content.add_word(word)

    vocab_context = Vocabulary(glove_dim)
    concepts_context = [word for word, cnt in counter_context.items() if cnt >= threshold]
    vocab_context.add_word('<PAD>')
    vocab_context.add_word('<UNK>')
    vocab_context.add_word('.')
    for i, concept in enumerate(concepts_context):
        vocab_context.add_word(concept)

    return vocab_content, vocab_context, category_vocab


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def get_context_textual(context, context_fields):
    context_words = []
    for field in context_fields:
        if field == 'category' or field == 'location':
            continue
        for token in nltk.tokenize.word_tokenize(normalizeString(xml_values(context, field))):
            # if len(token)>3:

            context_words = context_words + segment(token)
        context_words.append(".")
    return context_words


def camel_to_snake(name):
    if not name:
        return ""
    tokens = re.split(" |_|-|/|:", name)
    if len(tokens) == 0:
        return ""
    if len(tokens) == 1:
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return str.replace("_", " ").strip()
    str = ""
    for n in tokens:
        str = str + " " + camel_to_snake(n)
    return str.replace("_", " ").strip()


def idx_to_words(sampled_ids, vocab):
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<EOS>':
            break
    sentence = ' '.join(sampled_caption)
    return sentence


def remove_extra_tokens(ids):
    try:
        end = (ids == 2).nonzero()[0][0] if len((ids == 2).nonzero()[0]) > 0 else len(ids)
        return ids[(ids == 1).nonzero()[0][0] + 1:end] if len(
            ids[(ids == 1).nonzero()[0][0] + 1:end]) > 0 else np.array([3])
    except:
        return np.array([3])


def sig(x, thresh):
    return (2 / (1 + math.exp(-1 * thresh * x))) - 1
