import os
from collections import defaultdict

import nltk
import torch
import torch.utils.data as data
import json
from PIL import Image
from random import random
from utils import get_context_textual, sig


def embed_context(context, cxt_fields, cxt_vocab, len_cat_vocab):
    second_dim = cxt_vocab.glove_dim

    context_tensor = torch.zeros(max(1, len(context)-1), second_dim)
    if len(context) == 0:
        return context_tensor

    if 'category' in cxt_fields:
        context_tensor[0][context[0]] = 1

    if 'location' in cxt_fields:
        context_tensor[0][len_cat_vocab+context[1]] = 1

    glove_weights = cxt_vocab.glove_weights_custom(context[2:])
    context_tensor[1:, :cxt_vocab.glove_dim] = torch.from_numpy(glove_weights)

    return context_tensor


class IconDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_path, cnt_vocab, cxt_vocab, cxt_fields, cat_vocab, transform=None, w_loss=False, mask_threshold=1):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json_path: annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.cxt_fields = cxt_fields
        self.root = root
        with open(json_path, "r") as json_file:
            caps = json.load(json_file)
            self.caps = {}
            freq = defaultdict(int)
            for item in caps:
                id = item['id']
                self.caps[id] = item
                caption = item['content']
                freq[caption] += 1

        # Weight calc
        self.weight = {}
        min_freq = min(freq.values())
        for k, v in freq.items():
            self.weight[k] = min_freq / v if w_loss else 1
            self.weight[k] = sig(self.weight[k], 100)
            # self.weight[k] = 1 / (1 + math.exp(-1*self.weight[k]))
        self.ids = list(self.caps.keys())
        self.cxt_vocab = cxt_vocab
        self.cnt_vocab = cnt_vocab
        self.cat_vocab = cat_vocab
        self.transform = transform
        self.mask_threshold = mask_threshold

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caps = self.caps
        ann_id = self.ids[index]
        caption = caps[ann_id]['content']
        weight = self.weight[caption]
        caps[ann_id]['context']['android_id'] = caps[ann_id]['android-id']
        cxt_fields = self.cxt_fields.copy()
        if random() < self.mask_threshold:
            for x in ['activity_name', 'android_id', "father-id", "siblings-id"]:
                cxt_fields.remove(x)

        context_concepts = get_context_textual(caps[ann_id]['context'], cxt_fields)
        img_id = caps[ann_id]['img_path']
        img_id = int(img_id[img_id.find("/") - 1:-4].replace("/", ""))
        category_id = self.cat_vocab.index(caps[ann_id]['context']['category']) if caps[ann_id]['context']['category'] in self.cat_vocab else 0
        region_id = caps[ann_id]['context']['location'] if caps[ann_id]['context']['location'] > 0 else 0
        path = os.path.join(self.root, "{}".format(caps[ann_id]['img_path']))

        image = Image.open(path).convert('RGB')
        if caps[ann_id]['rotation'] == "1":
            image = image.rotate(90)
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.cnt_vocab('<SOS>')]
        caption.extend([self.cnt_vocab(token) for token in tokens])
        caption.append(self.cnt_vocab('<EOS>'))
        caption_tensor = torch.Tensor(caption)

        weight_tensor = torch.tensor([weight]*len(caption))

        # Convert context to word ids
        context = []
        # if 'category' in self.cxt_fields:
        context.append(category_id)
        # if 'location' in self.cxt_fields:
        context.append(region_id)
        for concept in context_concepts:
            id = self.cxt_vocab(concept)
            if id != self.cxt_vocab('<UNK>'):
                context.append(id)
        context_tensor = embed_context(context, cxt_fields, self.cxt_vocab, len(self.cat_vocab))

        return img_id, caps[ann_id]['img_path'], image, caption_tensor, weight_tensor, context_tensor

    def __len__(self):
        return len(self.ids)


# Creates mini-batch tensors from the list of tuples (image, caption)
def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data = [x for x in data if not all(w <= 3 for w in x[3])]  # remove unknowns
    data.sort(key=lambda x: len(x[3]), reverse=True)
    img_ids, paths, images, captions, weight, contexts = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    caption_lengths = [len(cap) for cap in captions]
    caption_targets = torch.zeros(len(captions), max(caption_lengths)).long()
    weight_targets = torch.zeros(len(captions), max(caption_lengths)).float()
    for i, cap in enumerate(captions):
        end = caption_lengths[i]
        caption_targets[i, :end] = cap[:end]
        weight_targets[i, :end] = weight[i][:end]

    # Merge contexts (from tuple of 3D tensor to 4D tensor).
    contexts_lengths = [len(context) for context in contexts]
    context_targets = torch.zeros(len(contexts), max(contexts_lengths), contexts[0].shape[1]).float()
    for i, context in enumerate(contexts):
        end = contexts_lengths[i]
        context_targets[i,:end] = context[:end]

    return img_ids, paths, images, caption_targets, caption_lengths, weight_targets, context_targets, contexts_lengths


def get_loader(root, json, cnt_vocab, cxt_vocab, context_fields, cat_vocab, transform, batch_size, shuffle, num_workers, w_loss=False, mask_threshold=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # Icon caption dataset
    icon = IconDataset(root=root,
                       json_path=json,
                       cnt_vocab=cnt_vocab,
                       cxt_vocab=cxt_vocab,
                       cxt_fields=context_fields,
                       cat_vocab=cat_vocab,
                       transform=transform,
                       w_loss=w_loss,
                       mask_threshold=mask_threshold)


    data_loader = torch.utils.data.DataLoader(icon, batch_size=batch_size,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
