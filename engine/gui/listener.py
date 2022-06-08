class Gui_listener:
    def __init__(self, name='listener', identifier=42, listener_class=42, to_notify=[]):
        self.name  = name
        self.id  = identifier
        self.listener_class = listener_class
        self.to_notify = to_notify
        self.misc_info = None # can store anything

    def delete(self):
        self.to_notify = None
        self.misc_info = None

    def add_notify(self, targets):
        try:
            self.to_notify.extend(targets)
        except:
            self.to_notify.append(targets)

    # can get a new value from the notified elements
    def notify(self, value, to_redraw=[]):
        for e in self.to_notify:
            if self.misc_info is not None:
                e.get_notified(self.listener_class, self.id, (value, self.misc_info), to_redraw)
            else:
                e.get_notified(self.listener_class, self.id, value, to_redraw)

    def copy(self, new_id=None, new_name=None):
        identifier = new_id
        if new_id is None:
            identifier = self.id
        cpy_name = new_name
        if new_name is None:
            cpy_name = self.name
        return Gui_listener(cpy_name, identifier=identifier, listener_class=self.listener_class, to_notify=self.to_notify)


class Listener(Gui_listener): #shortcut
    def __init__(self, listener_class, to_notify):
        super(Listener, self).__init__(name='listener', identifier=42, listener_class=listener_class, to_notify=to_notify)
