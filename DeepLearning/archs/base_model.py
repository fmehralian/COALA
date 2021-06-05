import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from .nn_models import EncoderCNN
from nlgeval import NLGEval
from overrides import EnforceOverrides, final
from DeepLearning.utils import idx_to_words, remove_extra_tokens
import time


def init_log():
    results_root = "out/results-"+time.strftime('%m%d%H%M%S')
    if not os.path.exists(results_root):
        os.mkdir(results_root)

    if not os.path.exists(results_root+"/failed"):
        os.mkdir(results_root+"/failed")

    if not os.path.exists(results_root+"/passed"):
        os.mkdir(results_root+"/passed")
    return results_root



class BaseModel(EnforceOverrides):
    def __init__(self, embed_size, label_vocab, device):
        self.device = device
        self.label_vocab = label_vocab
        self.encoder = EncoderCNN(embed_size).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self._get_optimizer_params(), lr=0.001)
        self.all_models = [self.encoder]
        self.eval = NLGEval(no_glove=True, no_skipthoughts=True)

    def _get_optimizer_params(self):
        raise NotImplementedError()

    @final
    def train(self, train_loader, val_loader, epochs, log_step, save_step, model_path, validation):
        for m in self.all_models:
            m.train()
        total_step = len(train_loader)
        best_val = 0
        for epoch in range(epochs):
            for i, (_, _, images, captions, cap_lengths, weights, contexts, con_lengths) in enumerate(train_loader):
                # Set mini-batch dataset
                images = images.to(self.device)
                captions = captions.to(self.device)
                contexts = contexts.to(self.device)
                weights = weights.to(self.device)
                targets = pack_padded_sequence(captions, cap_lengths, batch_first=True).data
                caption_weights = pack_padded_sequence(weights, cap_lengths, batch_first=True).data

                # Forward, backward and optimize
                img_features = self.encoder(images)
                outputs = self._compute_outputs(img_features, captions, cap_lengths, contexts, con_lengths)
                loss = self.criterion(outputs, targets)
                loss = sum(loss*caption_weights)/sum(caption_weights)
                self._zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print log info
                if i % log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                          .format(epoch, epochs, i, total_step, loss.item(), np.exp(loss.item())))

            if (epoch + 1) % save_step == 0 or epoch == epochs - 1:
                # validation results
                if validation:
                    cider = self._evaluate(val_loader, self.eval, None, False, 'CIDEr')
                    if cider > best_val:
                        self.save_model(model_path)
                        best_val = cider
                    for m in self.all_models:
                        m.train()
                else:
                    self.save_model(model_path)

    def _compute_outputs(self, img_features, captions, cap_lengths, contexts, con_lengths):
        raise NotImplementedError()

    def _zero_grad(self):
        self.encoder.zero_grad()

    def save_model(self, model_path):
        torch.save(self.encoder.state_dict(), os.path.join(
            model_path, 'encoder.ckpt'))

    def load_model(self, model_path):
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'encoder.ckpt'), map_location=self.device))

    @final
    def _evaluate(self, test_loader, eval, image_root, log, val_metric=None):
        for m in self.all_models:
            m.eval()
        data = {}
        for name in ['total', 'default', 'non-default']:
            data[name] = {'reference': [[]], 'candidate': [], 'count': 0, 'exact': 0}

        defaults = ['navigate up', 'more options', 'more option', 'close navigation drawer', 'open navigation drawer',
                    'clear query', 'search', 'next month', 'previous month', 'interstitial close',
                    'interstitial close button']

        if log:
            results_root = init_log()
            failed = open(results_root+'/failed.csv', 'w')
            passed = open(results_root+'/passed.csv', 'w')
            all = open(results_root + '/all.csv', 'w')
            all.write("id, gt, coala\n")
        for i, (img_ids, paths, images, captions, cap_lengths, weights, contexts, con_lengths) in enumerate(test_loader):
            images = images.to(self.device)
            contexts = contexts.to(self.device)
            for ind in range(images.size(0)):
                image_id = img_ids[ind]
                image_features = self.encoder(images[ind].unsqueeze(0))
                sampled_ids = self._sample(image_features, ind, contexts, con_lengths)
                sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

                sentence = idx_to_words(remove_extra_tokens(sampled_ids), self.label_vocab)
                target = idx_to_words(remove_extra_tokens(captions[ind].cpu().numpy()), self.label_vocab)
                data['total']['reference'][0].append(target)
                data['total']['candidate'].append(sentence)
                if target in defaults:
                    data['default']['reference'][0].append(target)
                    data['default']['candidate'].append(sentence)
                    data['default']['count'] += 1
                    data['default']['exact'] += 1 if target == sentence else 0
                else:
                    data['non-default']['reference'][0].append(target)
                    data['non-default']['candidate'].append(sentence)
                    data['non-default']['count'] += 1
                    data['non-default']['exact'] += 1 if target == sentence else 0
                data['total']['count'] += 1
                data['total']['exact'] += 1 if target.strip() == sentence else 0

                if log:
                    all.write("{}, {}, {}\n".format(str(image_id), target.strip(), sentence.strip()))
                    if target != sentence:
                        shutil.copy(os.path.join(image_root, paths[ind]), "{}/failed/{}.png".format(results_root, image_id))
                        failed.write("{}, {}, {}\n".format(str(image_id), target.replace("<UNK>", ""), sentence.replace("<UNK>", "")))
                    else:
                        shutil.copy(os.path.join(image_root, paths[ind]),
                                    "{}/passed/{}.png".format(results_root, image_id))
                        passed.write("{}, {}, {}\n".format(str(image_id), target.replace("<UNK>", ""), sentence.replace("<UNK>", "")))
        if log:
            failed.close()
            passed.close()
            all.close()
        if val_metric: # validation results on non-default metrics matter
            metrics = eval.compute_metrics(data['non-default']['reference'], data['non-default']['candidate'])
            if val_metric not in metrics.keys():
                raise Exception()
            return data['non-default']['exact']/data['non-default']['count']
            return metrics[val_metric]
        else:
            metrics = {}
            for name, d in data.items():
                metric = eval.compute_metrics(d['reference'], d['candidate'])
                metric['exact_match'] = d['exact'] / d['count']
                metric['count'] = d['count']
                metrics[name] = metric
            if log:
                with open(results_root+"/res.json", "w") as f:
                    json.dump(metrics, f)
            return metrics

    def _sample(self, image_features, ind, contexts, con_lengths):
        raise NotImplementedError()

    @final
    def report_metrics(self, test_loader, image_root, log):
        metrics = self._evaluate(test_loader, self.eval, image_root, log)
        print('all:', metrics['total'], 'non_defaults:', metrics['non-default'], "defaults:", metrics['default'])
