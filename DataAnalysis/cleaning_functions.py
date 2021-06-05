import re
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
import numpy as np
spell = SpellChecker()
wnl = WordNetLemmatizer()


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


def clean_string_basic(str):
    str = camel_to_snake(str)
    str = str.lower().replace("_", " ")

    # remove special chars
    str = re.sub('[^A-Za-z0-9\s]+', ' ', str)

    # correct spelling
    tokens = [wnl.lemmatize(spell.correction(token)) for token in str.split()]

    str = " ".join(tokens)
    str = str.replace("nav ", "navigation ")

    return str


def clean_context_basic(xml_context):
    cleaned_context = xml_context
    cleaned_context['activity_name'] = clean_string_basic(xml_context['activity_name'])
    cleaned_context['title'] = clean_string_basic(xml_context['title'])

    xml_context['family']['father']['id'] = clean_string_basic(xml_context['family']['father']['id'])
    xml_context['family']['father']['text'] = clean_string_basic(xml_context['family']['father']['text'])

    for item in xml_context['family']['siblings']:
        item['id'] = clean_string_basic(item['id'])
        item['text'] = clean_string_basic(item['text'])

    return cleaned_context


def clean_content(content):
    non_inf_tokens = ['picture', 'button', 'image', 'content', 'desc', 'description', 'icon', 'images', 'default', 'view',
               'src', 'btn', 'alt', 'null', 'none', 'text', 'textt', 'test', 'hint', 'unknown', 'untitled',
               'placeholder', 'photo']

    for item in non_inf_tokens:
        content = content.replace(item, "")
    content = content.strip()

    if content.startswith("click to") or content.startswith("touch to"):
        content = content[9:]
    if content.startswith("click for") or content.startswith("touch for"):
        content = content[10:]

    return content.strip()


non = ['scrollview', 'fragment', 'root', 'bottom', 'up', 'left', 'right', 'home', 'input', 'row', 'column', 'slide',
           'this', 'wrapper', 'switcher', 'footer', 'header', 'grid', 'tab', 'appbar', 'dialog', 'first', 'second',
           'third', 'list', 'smooth', 'wizard','decor', 'content', 'view', 'scroll', 'bar', 'action', 'pane', 'layout',
           'drawer', 'container', 'parent', 'toolbar', 'root', 'main', 'activity', 'action', 'image', 'button', 'text',
           'tabularbar', 'titlebar', 'frame', 'sheet', 'menu', 'my', 'picker', 'custom', 'panel', 'actionbar',
           'mainlayout', 'and', 'btn', 'app', 'list']


def clean_context(element):
    activity_name = element.xml_context['activity_name']
    activity_name = camel_to_snake(activity_name)
    activity_name = activity_name.replace("activity", "")
    activity_name = activity_name.replace("main", "")
    element.xml_context['activity_name'] = activity_name.strip()

    title = element.xml_context['title']
    if title is None:
        title = ""
    element.xml_context['title'] = title.lower().strip()

    family = element.xml_context['family']

    family['father']['id'] = clean_id(family['father']['id'])

    text = camel_to_snake(family['father']['text'])
    text = [token for token in text.split() if token not in non and len(token) > 2 and not '0' <= token[0] <= '9']
    family['father']['text'] = " ".join(text).strip()

    for item in family['siblings']:
        item['id'] = clean_id(item['id'])

        text = camel_to_snake(item['text'])
        text = [token for token in text.split() if token not in non and len(token) > 2 and not '0' <= token[0] <= '9']
        item['text'] = " ".join(text).strip()


def clean_id(android_id):
    if not android_id:
        return ""
    id = camel_to_snake(android_id)
    id = [token for token in id.split() if token not in non and len(token) > 2 and not '0' <= token[0] <= '9']
    return " ".join(id).strip()


def is_english(string):
    # tokens = nltk.tokenize.word_tokenize(string.lower())
    # for token in tokens:
    #     if token not in words.words():
    #         return False
    # return True
    try:
        string.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


if __name__ == '__main__':
    print(clean_string_basic("close button"))