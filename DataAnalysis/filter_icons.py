import json
import os
import pickle
import re
from time import gmtime, strftime
from PIL import Image

from cleaning_functions import *
from datetime import datetime
from random import random


def analyze(_args):
    print('initializing the analysis at {}...'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    icons, meaningless, predefined, statistics = initialize(_args)
    print('initialization done!')

    dup = set()  # checks duplicate icons in different screens of the same app
    image_id_counter = 0
    image_uid_counter = 0
    unlabeled = []
    for i, icon in enumerate(icons):
        if i % 1000 == 0:
            print('{}/{} -- '.format(i, len(icons)), statistics)
        dup_key = "{}_{}_{}".format(icon.app, icon.bounds, icon.content_desc)
        ignore_duplicate = 0 if dup_key in dup else 1
        dup.add(dup_key)
        statistics['total'][ignore_duplicate] += 1

        try:
            # checkout steps in icon_extractor.py
            prefix = "tmp" if _args.step == 0 else "UI5K"

            img_path = "{}png".format(icon.file_path[:-3])
            img_path = img_path[img_path.find(prefix):]
            img_path = "{}/{}".format(_args.data_path, img_path)

            image = Image.open(img_path)
        except:
            if statistics['no-image'][ignore_duplicate] == statistics['total'][ignore_duplicate] \
                    and statistics['total'][ignore_duplicate] > 10:
                print("Check the data path! So far, more than 10 icons had no image...")
                exit(0)
            statistics['no-image'][ignore_duplicate] += 1
            continue

        if not icon.content_desc:
            statistics['unlabeled'][ignore_duplicate] += 1
            if _args.unlabeled:
                if ignore_duplicate == 1 and random() < 0.2 and image_uid_counter < 2000:
                    cor = icon.bounds
                    if icon.rotation == "1":
                        cor[0], cor[2], cor[1], cor[3] = image.width - cor[3], image.width - cor[1], cor[0], cor[2]
                    try:
                        area = image.crop(cor)
                        area.save(
                            "out/unlabeled_image_v{}_s{}/{}.png".format(_args.filter_version, _args.step,
                                                                                    image_uid_counter))
                    except:
                        continue

                    icon.set_id(image_id_counter)
                    image_uid_counter += 1
                    unlabeled.append(icon)
                continue

        statistics['labeled'][ignore_duplicate] += 1
        if not is_english(icon.content_desc):
            statistics['non-english'][ignore_duplicate] += 1
            continue

        if bool(re.search("^[A-Za-z0-9_\-]*$", icon.content_desc)):
            statistics['special-char'][ignore_duplicate] += 1
        clean_content = clean_string_basic(icon.content_desc)
        if clean_content in predefined:
            statistics['predefined'][ignore_duplicate] += 1
        elif clean_content in meaningless:
            statistics['meaningless'][ignore_duplicate] += 1
            continue
        statistics['avg_length'][ignore_duplicate] += len(clean_content.split())

        cor = icon.bounds
        if icon.rotation == "1": # check if the device was rotated
            cor[0], cor[2], cor[1], cor[3] = image.width - cor[3], image.width - cor[1], cor[0], cor[2]
        try:
            area = image.crop(cor)
        except:
            statistics['crop'][ignore_duplicate] += 1
            continue
        if area.size[0] * area.size[1] > 0.05 * image.size[0] * image.size[1]:  # skip large pictures
            statistics['large-pic'][ignore_duplicate] += 1
            continue

        if area.size[0] * area.size[1] <= 0:  # skip invisible icons!
            statistics['zero-surf'][ignore_duplicate] += 1
            continue

        if max(area.size[0], area.size[1]) / min(area.size[0], area.size[1]) > 2:  # skip narrow images
            statistics['narrow'][ignore_duplicate] += 1
            continue

        # save a good icon!
        if ignore_duplicate == 1:
            area.save(
                "out/image_v{}_s{}/{}.png".format(_args.filter_version, _args.step, image_id_counter))
            icon.set_id(image_id_counter)
            icon.content_desc = clean_content
            clean_context(icon)
            append_icon(icon, "out/coala_data_v{}_s{}.json".format(_args.filter_version, _args.step))
            image_id_counter += 1

    # remove the last comma
    with open("out/coala_data_v{}_s{}.json".format(_args.filter_version, _args.step), 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()

    # close the list
    with open("out/coala_data_v{}_s{}.json".format(_args.filter_version, _args.step), "a") as f:
        f.write("]")

    if _args.unlabeled:
        with open('out/unlabeled_icons_v{}_s{}.pkl'.format(_args.filter_version, _args.step), 'wb') as f:
            pickle.dump(unlabeled, f)
    print('dumped {} at {}'.format(len(unlabeled),strftime("%Y-%m-%d %H:%M:%S", gmtime())), statistics)


# load the icons, make required directories and initialize the variables
def initialize(_args):
    if os.path.exists('out/image_v{}_s{}'.format(_args.filter_version, _args.step)) or \
            os.path.exists("out/coala_data_v{}_s{}.json".format(_args.filter_version, _args.step)) or \
            os.path.exists('out/unlabeled_image_v{}_s{}'.format(_args.filter_version, _args.step)):
        print('output exists!')
        exit(0)
    os.mkdir('out/image_v{}_s{}'.format(_args.filter_version, _args.step))
    if _args.unlabeled:
        os.mkdir('out/unlabeled_image_v{}_s{}'.format(_args.filter_version, _args.step))
    with open("out/coala_data_v{}_s{}.json".format(_args.filter_version, _args.step), "w") as f:
        f.write("[")
    predefined = ['navigate up', 'more options', 'more option', 'close navigation drawer', 'open navigation drawer',
                  'clear query',
                  'search', 'next month', 'previous month', 'interstitial close button', 'interstitial close']
    with open("aux/meaningless_label.txt", "r") as f:
        a = f.read()
        meaningless = [x for x in a.split("\n") if x and not x.startswith("=")]
    print('start loading data {}'.format(datetime.now()))
    icons_path = "out/icons_v{}_s{}.pkl".format(_args.icon_version, _args.step)
    with open(icons_path, "rb") as f:
        icons = pickle.load(f)
        print('loaded data {}'.format(datetime.now()))
    statistics = {'total': [0, 0], 'unlabeled': [0, 0], 'labeled': [0, 0], 'non-english': [0, 0], 'meaningless': [0, 0],
                  'predefined': [0, 0], 'avg_length': [0, 0], 'special-char': [0, 0], 'large-pic': [0, 0],
                  'zero-surf': [0, 0], 'narrow': [0, 0], 'no-image': [0, 0], 'crop': [0, 0]}
    return icons, meaningless, predefined, statistics


def icon2json(icon, mode='coala'):
    if mode == 'coco':
        return json.dumps({'annotation': {"id": icon.id, "caption": icon.content_desc, "image_id": icon.id},
                'image': {"id": icon.id, "filename": "{}.png".format(icon.id),
                          'height': icon.bounds[3] - icon.bounds[1],
                          'width': icon.bounds[2] - icon.bounds[1]}})
    elif mode == 'coala':
        return json.dumps({'id': icon.id, 'img_id': icon.image_id,
                'img_path': "{}.png".format(icon.id),
                'pkg_name': icon.app,
                'activity_name': icon.activity, 'content': icon.content_desc,
                'bounds': icon.bounds,
                'android-id': icon.android_id, 'context': icon.xml_context,
                'type': icon.ancestor, 'rotation': icon.rotation})


def append_icon(icon, output):
    with open(output, 'a') as f:
        f.write("{},".format(icon2json(icon)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='data parser for LabelDroid')
    parser.add_argument('--icon-version', type=int, required=True, help='icons version')
    # due to the limited memory on machines, we considered four steps at icon_extractor.py
    parser.add_argument('--step', type=int, required=True, help='icons step')
    parser.add_argument('--filter-version', type=int, required=True, help='result version')
    parser.add_argument('--data-path', type=str, required=True, help='the root path for screenshots')
    parser.add_argument('--unlabeled', dest='unlabeled', action='store_true', help='save unlabeled icons as well')
    parser.set_defaults(unlabeled=False)
    _args = parser.parse_args()
    print("Arguments", _args)
    analyze(_args)
