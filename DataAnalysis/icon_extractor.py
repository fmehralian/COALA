import xml.etree.ElementTree as ET
import os
import pickle
import csv
from time import gmtime, strftime
import size_configs
from models import Node, Root


# In a hierarchical xml layout, find the parent and siblings of the node
def find_family(node):
    father = {'id': '', 'text': ''}
    if node.father.android_id:
        father['id'] = node.father.android_id
    if node.father.text:
        father['text'] = node.father.text

    siblings = []
    for child in node.father.children:
        if child == node:
            continue
        if child.android_id or child.text:
            siblings.append({'id': child.android_id if child.android_id else '', 'text': child.text})

    return {'father': father, 'siblings': siblings}


# Find the zone, in which an icon resides.
def find_location(node):
    zones = size_configs.zones
    for zone, zone_bounds in zones.items():
        if zone <= 3:
            if zone_bounds[0] <= node.bounds[0] and node.bounds[2] <= zone_bounds[2] :
                return zone
        elif zone >= 7:
            if zone_bounds[1] <= node.bounds[1] and node.bounds[3] <= zone_bounds[3]:
                return zone
        else:
            if zone_bounds[0] <= node.bounds[0] and node.bounds[2] <= zone_bounds[2] :
                return zone
    return 0


# Parse the layout to fill node objects
def find_nodes(node, father, file_path, activity_name, rotation):
    value = xml2node(node, file_path, activity_name, rotation)
    value.set_father(father)
    for child in node.findall("node"):
        value.add_child(find_nodes(child, value, file_path, activity_name, rotation))

    return value


def str2bool(str):
    return True if str == 'true' else False


def get_image_id(file_path):
    return file_path[file_path.rindex("/") + 1:-4]


def get_android_id(resource_id):
    return resource_id.split("/")[-1] if resource_id else ''


def xml2node(node, file_path, activity_name, rotation):
    value = Node(node.get('text'), get_android_id(node.get('resource-id')),
                 node.get('class'), node.get('package'), node.get('content-desc'), str2bool(node.get('clickable')),
                 str2bool(node.get('long-clickable')),
                 [int(x) for x in node.get('bounds').replace("][", ",")[1:-1].split(",")],
                 file_path, activity_name, get_image_id(file_path), rotation)
    return value


def load_activity_names(all_states):
    activity_names = {}
    with open(all_states, "r") as f:
        s = f.read()
        e = s.split("\n\n-----------\n\n")
        for e1 in e:
            if "\n" not in e1:
                continue
            a = e1.split("\n")
            page_id = a[0][a[0].rindex("/") + 1:]
            if not page_id.startswith("S"):
                continue
            main_activity = a[1][a[1].rindex(".") + 1:]
            activity_names[page_id] = main_activity
    return activity_names


def find_title(root_node):
    stack = []
    for child in root_node.children:
        stack.append(child)
    left_most = ""
    left_bound = size_configs.dev_topbar_max_width
    while stack:
        node = stack.pop()

        if node.text and node.bounds[3] < size_configs.dev_topbar_max_height and node.bounds[2] < left_bound:
            left_most = node.text
            left_bound = node.bounds[2]

        for child in node.children:
            if child:
                stack.append(child)
    return left_most


app_dict = {}


def get_category(app):
    if not app_dict:
        with open("aux/description.txt", "r") as f:
            app_categories_reader = csv.reader((x.replace('\0', '') for x in f), delimiter='\t')
            # row[0]: package_name row[1]: play store name

            for row in app_categories_reader:
                app_dict[row[0]] = row[1]
    set_categories = list(set(app_dict.values()))

    return set_categories.index(app_dict[app]) + 1 if app in app_dict else 0


def extract_icons(screen):
    stack = []
    elements = []
    for parent in screen.children:
        stack.append(parent)
    while stack:
        node = stack.pop()

        if ('ImageButton' in node.ancestor or 'ImageView' in node.ancestor) \
                and (
                node.clickable or node.long_clickable):
            context_xml = {"activity_name": screen.activity_name, "title": screen.title,
                           "family": find_family(node), "category": get_category(screen.app),
                           "location": find_location(node)}
            node.set_context(context_xml)
            elements.append(node)
        for child in node.children:
            stack.append(child)
    return elements


def parse_layouts(_args):
    data_path = _args.data_path
    # the apps in the raw dataset are distributed in 4 folders
    sub_dirs = ["{}/tmp".format(data_path), "{}/UI5K/verbo1/top_10000_google_play_20170510_cleaned_outputs_verbose_xml".format(data_path), "{}/UI5K/verbo2/verbo_2".format(data_path),
                "{}/UI5K/verbo3/verbo3".format(data_path)]

    # With limited memory, extract icons in 4 steps
    sub_dirs = [sub_dirs[_args.step]]
    icons = []
    for i, app_dir in enumerate(sub_dirs):
        print('started: {} at {}'.format(app_dir, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        base_ui = app_dir
        base_state = app_dir
        apps = os.listdir(base_ui)
        counter = 0
        not_exist = 0
        for i in range(len(apps)):
            app = apps[i]
            counter += 1
            # filter apps without proper states or ui directory
            if not os.path.exists("{}/{}/stoat_fsm_output/allstates.txt".format(base_state, app)) or \
                    not os.path.exists("{}/{}/stoat_fsm_output/ui".format(base_state, app)):
                not_exist += 1
                continue
            if counter % 100 == 0:
                print(counter, "/", len(apps), " :", app_dir)
            if app == 'stoat_fsm_output':
                continue
            try:
                screens = [x for x in os.listdir("{}/{}/stoat_fsm_output/ui".format(base_ui, app)) if x.endswith('.xml')]
                activity_names = load_activity_names("{}/{}/stoat_fsm_output/allstates.txt".format(base_state, app))
                if len(activity_names) == 0:
                    not_exist += 1
                    continue
                for screen in screens:
                    file_path = "{}/{}/stoat_fsm_output/ui/{}".format(base_ui, app, screen)
                    if not os.path.exists(file_path[:-3] + "png"):
                        continue
                    if screen not in activity_names:
                        continue
                    root_xml = ET.parse("{}/{}/stoat_fsm_output/ui/{}".format(base_ui, app, screen)).getroot()
                    rotation = root_xml.get('rotation')
                    root_node = Root(file_path, app[:app.rindex("_")], activity_names[screen], rotation)
                    parent_xml = root_xml.findall('node')[0]
                    parent_node = xml2node(parent_xml, file_path, activity_names[screen], rotation)
                    if parent_node.bounds[0] > 0:
                        continue
                    root_node.add_child(parent_node)
                    for child in parent_xml.findall('node'):
                        parent_node.add_child(find_nodes(child, parent_node, file_path, activity_names[screen], rotation))

                    root_node.set_title(find_title(root_node))
                    screen_icons = extract_icons(root_node)

                    icons += screen_icons

            except Exception as e:
                print(e, app)

    with open("out/icons_v{}_s{}.pkl".format(_args.version, _args.step), "wb") as f:
        pickle.dump(icons, f)
    print("dumped {} icons at {} in {}".format(len(icons), strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                                               "out/icons_v{}_s{}.pkl".format(_args.version, _args.step)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='data parser for LabelDroid')
    parser.add_argument('--data-path', type=str, required=True, help='The root path of the data')
    parser.add_argument('--step', type=int, required=True, help='Subfolder index')
    parser.add_argument('--version', type=int, required=True, help='Icons version')
    _args = parser.parse_args()
    print("Arguments", _args)
    parse_layouts(_args)

