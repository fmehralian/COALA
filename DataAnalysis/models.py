class Node:
    def __init__(self, text, resource_id, ancestor, package, content_desc, clickable, long_clickable, bounds
                 , file_path, activity, image_id, rotation):
        self.activity = activity
        self.text = text
        self.android_id = resource_id
        self.ancestor = ancestor
        self.app = package
        self.content_desc = content_desc
        self.clickable = clickable
        self.long_clickable = long_clickable
        self.bounds = bounds
        self.file_path = file_path
        self.father = None
        self.children = []
        self.xml_context = {}
        self.image_id = image_id
        self.id = None  # same as cropped image name
        self.rotation = rotation

    def set_id(self, id):
        self.id = id

    def set_father(self, father):
        self.father = father

    def add_child(self, child):
        self.children.append(child)

    def set_context(self, xml_context):
        self.xml_context = xml_context



class Root:
    def __init__(self, file_path, app, activity_name, rotation):
        self.children = []
        self.file_path = file_path
        self.app = app
        self.activity_name = activity_name
        self.title = ""
        self.rotation = rotation

    def add_child(self, child):
        self.children.append(child)

    def set_title(self, title):
        self.title = title