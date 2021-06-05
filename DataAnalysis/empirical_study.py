import csv
import json
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import nltk
import os
from PIL import Image
import random


def study_context(_args):
    icons = load_data(_args)
    fields = ['activity_name', 'title', 'android_id', "father-id", "siblings-id", "siblings-text",
              "category", "random", "location"]
    for field in fields:
        data1, data2 = pair_tokens_context(icons, field)
        corr, _ = pearsonr(data1, data2)
        res = normalized_mutual_info_score(data1, data2)
        print(field, corr, res)


def study_images(_args):
    icons = load_data(_args)
    contents = {}
    counter = 0
    tokens = {}
    store_classes = {}
    for icon in icons:
        if counter % 100 == 0:
            print('{}/{}'.format(counter, len(icons)))
        counter += 1
        path = os.path.join(_args.image_path, icon['img_path'])
        img_class = image_concept(path)
        store_classes[path] = img_class
        if img_class in contents:
            contents[img_class].add(icon['content'])
        else:
            contents[img_class] = set([icon['content']])
        for token in nltk.tokenize.word_tokenize(icon['content'].lower()):
            if img_class in tokens:
                tokens[img_class].add(token)
            else:
                tokens[img_class] = set([token])
    with open("data/emp_image_classes_v{}.json".format(_args.filter_version), "w") as f:
        json.dump(store_classes, f)

    for img_class in contents:
        print(img_class, ": #label=", len(contents[img_class]), " #tok=", len(tokens[img_class]), " samples:",
              list(contents[img_class])[0:min(3, len(contents[img_class]))])


def dataset(_args):
    icons = load_data(_args)
    app = set()
    screen = set()
    for icon in icons:
        app.add(icon['pkg_name'])
        screen.add("{}_{}".format(icon['pkg_name'], icon['img_id']))
    print(len(app), len(screen), len(icons))


def study_label_distribution(_args):
    icons = load_data(_args)
    content_freq = {}
    token_freq = {}
    stop_words = ['the', 'a', 'to', 'x', 'and', 'or', 'this', 'for', 'my', 'in', 'is', 'are', 'you', 'at']
    for icon in icons:
        tokens = nltk.tokenize.word_tokenize(icon['content'].lower())

        for token in tokens:
            if token in token_freq:
                token_freq[token] += 1
            else:
                token_freq[token] = 1

        if icon['content'] in content_freq:
            content_freq[icon['content']] += 1
        else:
            content_freq[icon['content']] = 1

    whole_match = 0
    partially_match = 0
    uniques = 0
    for icon in icons:
        if content_freq[icon['content']] > 1:
            continue
        uniques += 1
        tokens = nltk.tokenize.word_tokenize(icon['content'].lower())
        tokens = list(set(tokens)- set(stop_words))
        rep_tok = 0
        for token in tokens:
            if token_freq[token] > 1:
                rep_tok += 1
        if rep_tok == len(tokens):
            whole_match += 1
        if rep_tok > 0:
            partially_match += 1

    print("tot:{} / {} found all their tokens and {} found at least one".format(uniques, whole_match, partially_match))

    content_freq = dict(sorted(content_freq.items(), key=lambda item: item[1], reverse=True))
    with open('data/emp_content_freq_v{}.csv'.format(_args.filter_version), "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['content description', 'count'])
        for key in content_freq:
            csvwriter.writerow([key, content_freq[key]])

    token_freq = dict(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
    with open('data/emp_token_freq_v{}.csv'.format(_args.filter_version), "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['content description', 'count'])
        for key in token_freq:
            csvwriter.writerow([key, token_freq[key]])


def load_data(_args):
    with open("data/coala_data_v_{}_s{}.json".format(_args.filter_version, 0), "r") as f:
        icons_0 = json.load(f)
        for icon in icons_0:
            icon['img_path'] = "{}/{}".format("labelDroid_image_v{}_s{}".format(_args.filter_version, 0),
                                              icon['img_path'])
    with open("data/coala_data_v_{}_s{}.json".format(_args.filter_version , 1), "r") as f:
        icons_1 = json.load(f)
        for icon in icons_1:
            icon['img_path'] = "{}/{}".format("labelDroid_image_v{}_s{}".format(_args.filter_version , 1),
                                              icon['img_path'])
    with open("data/coala_data_v_{}_s{}.json".format(_args.filter_version, 2), "r") as f:
        icons_2 = json.load(f)
        for icon in icons_2:
            icon['img_path'] = "{}/{}".format("labelDroid_image_v{}_s{}".format(_args.filter_version, 2),
                                              icon['img_path'])
    with open("data/coala_data_v_{}_s{}.json".format(_args.filter_version , 3), "r") as f:
        icons_3 = json.load(f)
        for icon in icons_3:
            icon['img_path'] = "{}/{}".format("labelDroid_image_v{}_s{}".format(_args.filter_version , 3),
                                              icon['img_path'])
    icons = icons_0 + icons_1 + icons_2 + icons_3
    selected_items = {}
    for icon in icons:
        if "{}_{}".format(icon['pkg_name'], icon['content']) in selected_items:
            continue
        selected_items["{}_{}".format(icon['pkg_name'], icon['content'])] = icon
    return list(selected_items.values())


# {'add': 0, 'arrow_backward': 1, 'arrow_downward': 2, 'arrow_forward': 3, 'arrow_upward': 4, 'attach_file': 5, 'av_forward': 6, 'av_rewind': 7, 'avatar': 8, 'bluetooth': 9, 'book': 10, 'bookmark': 11, 'build': 12, 'call': 13, 'cart': 14, 'chat': 15, 'check': 16, 'close': 17, 'compare': 18, 'copy': 19, 'dashboard': 20, 'date_range': 21, 'delete': 22, 'description': 23, 'dialpad': 24, 'edit': 25, 'email': 26, 'emoji': 27, 'expand_less': 28, 'expand_more': 29, 'explore': 30, 'facebook': 31, 'favorite': 32, 'file_download': 33, 'filter': 34, 'filter_list': 35, 'flash': 36, 'flight': 37, 'folder': 38, 'follow': 39, 'font': 40, 'fullscreen': 41, 'gift': 42, 'globe': 43, 'group': 44, 'help': 45, 'history': 46, 'home': 47, 'info': 48, 'label': 49, 'launch': 50, 'layers': 51, 'list': 52, 'location': 53, 'location_crosshair': 54, 'lock': 55, 'menu': 56, 'microphone': 57, 'minus': 58, 'more': 59, 'music': 60, 'national_flag': 61, 'navigation': 62, 'network_wifi': 63, 'notifications': 64, 'pause': 65, 'photo': 66, 'play': 67, 'playlist': 68, 'power': 69, 'redo': 70, 'refresh': 71, 'repeat': 72, 'reply': 73, 'save': 74, 'search': 75, 'send': 76, 'settings': 77, 'share': 78, 'shop': 79, 'skip_next': 80, 'skip_previous': 81, 'sliders': 82, 'star': 83, 'swap': 84, 'switcher': 85, 'thumbs_down': 86, 'thumbs_up': 87, 'time': 88, 'twitter': 89, 'undo': 90, 'videocam': 91, 'visibility': 92, 'volume': 93, 'wallpaper': 94, 'warning': 95, 'weather': 96, 'zoom_out': 97}
def image_concept(input):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(input)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    resnet = models.resnet18(pretrained=True)

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 98)
    resnet.load_state_dict(torch.load("../DeepLearning/out/models/icon_image_classifier.pkl"))
    resnet.eval()
    output = resnet(batch_t)
    _, pred = torch.max(output, 1)
    return pred.item()


def pair_tokens_context(icons, field):
    tok_vocab = []
    context_vocab = []
    data1 = []
    data2 = []
    for icon in icons:
        icon['context']['android_id'] = icon['android-id']
        tokens = nltk.tokenize.word_tokenize(icon['content'].lower())
        set_content_tokens = set(tokens)
        set_context_values = ["random"]
        if field != "random":
            set_context_values = set(xml_values(icon['context'], field))
        for tok in set_content_tokens:
            for ctx in set_context_values:
                if tok in tok_vocab:
                    tok_idx = tok_vocab.index(tok)
                else:
                    tok_vocab.append(tok)
                    tok_idx = tok_vocab.index(tok)
                if field == "random":
                    data1.append(tok_idx)
                    data2.append(random.choice(range(1, 100)))
                    continue
                if ctx in context_vocab:
                    ctx_idx = context_vocab.index(ctx)
                else:
                    context_vocab.append(ctx)
                    ctx_idx = context_vocab.index(ctx)
                data1.append(tok_idx)
                data2.append(ctx_idx)
    return data1, data2


def xml_values(xml_context, field):
    if "-" in field:
        member = field.split("-")[0]
        tag = field.split("-")[1]
        return xml_values_family(xml_context, member, tag)
    else:
        if field == 'category' or field == 'location':
            return [xml_context[field]]
        return xml_context[field].split()


def xml_values_family(xml_context, family_member, tag):
    if family_member == 'father':
        return xml_context['family']['father'][tag].split()
    str = ""
    for father in xml_context['family'][family_member]:
        str += father[tag]
    return str.split()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='split icons for train/test/val')
    parser.add_argument('--filter-version', type=int, required=True, help='icons version')
    parser.add_argument('--image-dist', dest="image", action="store_true")
    parser.add_argument('--label-dist', dest="label", action="store_true")
    parser.add_argument('--context-dist', dest="context", action="store_true")
    parser.add_argument('--image-path', type=str)
    parser.set_defaults(image=False)
    parser.set_defaults(label=False)
    parser.set_defaults(context=False)
    _args = parser.parse_args()
    print("Arguments", _args)
    if _args.image:
        study_images(_args)
    if _args.label:
        study_label_distribution(_args)
    if _args.context:
        study_context(_args)
    dataset(_args)