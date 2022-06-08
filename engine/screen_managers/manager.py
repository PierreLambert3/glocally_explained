

class Manager():
    def __init__(self, name, initial_state = False):
        self.name = name
        self.to_redraw_on_next_iter = []
        self.asked_a_redraw         = False
        self.active                 = initial_state
        self.main_window = None
        # self.pressed_keys           = {K_CTRL : False}

    def sleep(self):
        self.active = False

    def wake_up(self, prev_manager): # called when passing from another screen to the main screen
        self.active  = True
        if self.main_window is not None:
            self.ask_redraw(self.main_window)

    def end(self):
        self.active = False

    def get_notified(self, event_class, id, value, to_redraw = []):
        pass

    def ask_redraw(self, element):
        if self.active:
            element.schedule_draw(self.to_redraw_on_next_iter)
            self.asked_a_redraw = True
