import json
import random
import csv
import os


def create_dir_if_not_exists(img_path):
    if not os.path.exists(img_path):
        os.makedirs(img_path)


# writing the data to files to be used by deep learning models
def write_to_files(root, split, apps, selected_items, split_version):

    my_ann_files = []
    coco_ann_files = []

    create_dir_if_not_exists("{}/{}/v{}/ann".format(root, "labelDroid", split_version))
    create_dir_if_not_exists("{}/{}/v{}/coco".format(root, "labelDroid", split_version))
    my_ann_files.append("{}/{}/v{}/ann/{}.json".format(root, "labelDroid", split_version, split))
    coco_ann_files.append("{}/{}/v{}/coco/captions_{}.json".format(root, "labelDroid", split_version, split))
    my_anns = []
    coco_anns = {"annotations": [], "images": []}

    count_icons = 0
    for app in apps:
        for element in selected_items[app]:
            count_icons += 1
            if _args.filter_version == 103:
                element['img_path'] = "{}/{}.png".format(element['img_path'].split("/")[0], element['id']) # filter v103
            element['id'] = int(element['img_path'][element['img_path'].find("/")-1]+str(element['id']))
            my_anns.append(element)  # component_type doesn't exist for label droid
            coco_item_ann = {"id": element['id'], "caption": element['content'], "image_id": element['id']}

            coco_item_image = {"id": element['id'], "filename": element['img_path'],
                               'height': element['bounds'][3] - element['bounds'][1],
                               'width': element['bounds'][2] - element['bounds'][1]}
            coco_anns['annotations'].append(coco_item_ann)
            coco_anns['images'].append(coco_item_image)

    for idx, my_ann_file in enumerate(my_ann_files):
        with open(my_ann_file, "w") as f:
            f.write(json.dumps(my_anns))
    for idx, coco_ann_file in enumerate(coco_ann_files):
        with open(coco_ann_file, "w") as f:
            f.write(json.dumps(coco_anns))


def get_ratio(pkg_to_data, pkg_names):
    p, np = 0, 0
    for k in pkg_names:
        v = pkg_to_data[k]
        p += len(v['p'])
        np += len(v['np'])
    return p, np, p/(np+p)


# check if the new split has the same distribution as the whole dataset
def new_dist(_args):
    from collections import defaultdict
    import random
    icons = load_icons(_args)
    unique_icons = []
    app_dup = set()
    selected_items = defaultdict(list)
    for icon in icons:
        if not _args.duplicate:
            if "{}_{}".format(icon['pkg_name'], icon['content']) in app_dup:
                continue
            app_dup.add("{}_{}".format(icon['pkg_name'], icon['content']))
        unique_icons.append(icon)
        selected_items[icon['pkg_name']].append(icon)
    pkg_to_data = defaultdict(lambda: defaultdict(list))
    defaults = ['navigate up', 'more options', 'more option', 'close navigation drawer', 'open navigation drawer',
                'clear query', 'search', 'next month', 'previous month', 'interstitial close',
                'interstitial close button']
    for x in unique_icons:
        key = "p" if x['content'].strip() in defaults else "np"
        pkg_to_data[x['pkg_name']][key].append(x)
    p, np = 0, 0
    for k, v in pkg_to_data.items():
        p += len(v['p'])
        np += len(v['np'])
    print(f"P: {p}, NP: {np}, Ratio: {p/(p+np)}")
    s_pkg_names = list(pkg_to_data.keys())
    random.shuffle(s_pkg_names)
    test_ratio = 0.1
    valid_ratio = 0.1
    test_size = int(len(s_pkg_names)*test_ratio)
    valid_size = int(len(s_pkg_names) * valid_ratio)
    test_apps = s_pkg_names[:test_size]
    val_apps = s_pkg_names[test_size: test_size+valid_size]
    train_apps = s_pkg_names[test_size+valid_size:]
    for x in [test_apps, val_apps, train_apps]:
        print(get_ratio(pkg_to_data, x))

    selected_items = {}
    for icon in icons:
        if not _args.duplicate:
            if "{}_{}".format(icon['pkg_name'], icon['content']) in app_dup:
                continue
            app_dup.add("{}_{}".format(icon['pkg_name'], icon['content']))
        if icon['pkg_name'] in selected_items:
            selected_items[icon['pkg_name']].append(icon)
        else:
            selected_items[icon['pkg_name']] = [icon]

    root = "out"

    write_to_files(root, "train", train_apps, selected_items, _args.split_version)
    write_to_files(root, "test", test_apps, selected_items, _args.split_version)
    write_to_files(root, "val", val_apps, selected_items, _args.split_version)


def main(_args):
    icons = load_icons(_args)
    app_dup = set()
    selected_items = {}
    for icon in icons:
        if not _args.duplicate:
            if "{}_{}".format(icon['pkg_name'], icon['content']) in app_dup:
                continue
            app_dup.add("{}_{}".format(icon['pkg_name'], icon['content']))
        if icon['pkg_name'] in selected_items:
            selected_items[icon['pkg_name']].append(icon)
        else:
            selected_items[icon['pkg_name']] = [icon]

    freq = {}
    for icons in selected_items.values():
        for icon in icons:
            if icon['content'] in freq:
                freq[icon['content']] += 1
            else:
                freq[icon['content']] = 1

    freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
    with open('out/freq_v{}.csv'.format(_args.split_version), "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['content description', 'count'])
        for key in freq:
            csvwriter.writerow([key, freq[key]])

    app_list = list(selected_items.keys())
    random.shuffle(app_list)
    num_test = int(0.1 * len(app_list))
    test_apps, val_apps, train_apps = app_list[0:num_test], app_list[num_test:num_test * 2], app_list[num_test * 2:]
    print("test apps:", len(test_apps), "val apps:", len(val_apps), "train apps:", len(train_apps))
    total = len(selected_items.values())

    prob = {}
    for key in freq:
        prob[key] = total / freq[key]

    root = "data"

    write_to_files(root, "train", train_apps, selected_items, _args.split_version)
    write_to_files(root, "test", test_apps, selected_items, _args.split_version)
    write_to_files(root, "val", val_apps, selected_items, _args.split_version)

    print("done splitting")


def load_icons(_args):
    with open("{}/coala_data_v{}_s{}.json".format(_args.data_path, _args.filter_version, 0), "r") as f:
        icons_0 = json.load(f)
        for icon in icons_0:
            icon['img_path'] = "{}/{}".format("image_v{}_s{}".format(_args.filter_version, 0),
                                              icon['img_path'])
    with open("{}/coala_data_v{}_s{}.json".format(_args.data_path, _args.filter_version, 1), "r") as f:
        icons_1 = json.load(f)
        for icon in icons_1:
            icon['img_path'] = "{}/{}".format("image_v{}_s{}".format(_args.filter_version, 1),
                                              icon['img_path'])
    with open("{}/coala_data_v{}_s{}.json".format(_args.data_path, _args.filter_version, 2), "r") as f:
        icons_2 = json.load(f)
        for icon in icons_2:
            icon['img_path'] = "{}/{}".format("image_v{}_s{}".format(_args.filter_version, 2),
                                              icon['img_path'])
    with open("{}/coala_data_v{}_s{}.json".format(_args.data_path, _args.filter_version, 3), "r") as f:
        icons_3 = json.load(f)
        for icon in icons_3:
            icon['img_path'] = "{}/{}".format("image_v{}_s{}".format(_args.filter_version, 3),
                                              icon['img_path'])
    icons = icons_0 + icons_1 + icons_2 + icons_3
    return icons


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='split icons for train/test/val')
    parser.add_argument('--data-path', type=str, required=True, help='Base Path')
    parser.add_argument('--filter-version', type=int, required=True, help='icons version')
    parser.add_argument('--split-version', type=int, required=True, help='split version')
    parser.add_argument('--ignore-dup', dest='duplicate', action='store_false')
    parser.set_defaults(duplicate=True)
    _args = parser.parse_args()
    print("Arguments", _args)
    main(_args)
