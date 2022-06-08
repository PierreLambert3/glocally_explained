from engine.gui.event_ids import WINDOW_CLOSE
from engine.gui.element import Element

class Container(Element):
    def __init__(self, position, dimension, name, parent, uid_generator, show_border=False, filled=True, color=None, background_color=None, border_size=1,\
        listening_hover=False, listening_Lclick=False, listening_Rclick=False, listening_scroll=False, attach_to_parent=False):
        super(Container, self).__init__(position=position, dimension=dimension, name=name, parent=parent, uid=uid_generator.get(),\
            color=color, background_color=background_color,listening_hover=listening_hover, listening_Lclick=listening_Lclick, listening_Rclick=listening_Rclick, listening_scroll=listening_scroll)
        self.listen_to(["hover", "Lclick", "Rclick", "scroll"])
        self.containers = []
        self.leaves     = []
        self.filled       = filled
        if filled:
            self.add_rect([(0,0), (1,1), 0, self.background_color])
        self.show_border = show_border
        if show_border:
            self.add_rect([(0,0), (1,1), border_size, self.color])
        if attach_to_parent:
            self.parent.add_container(self)

    def copy(self, uid_generator, with_content):
        cpy = Container(self.pos_pct, self.dim_pct, self.name, self.parent, uid=uid_generator.get(), show_border=self.shows_border,\
                filled=self.filled, color=self.color, background_color=self.background_color)
        if with_content:
            cpy.containers = [c for c in self.containers]
            cpy.leaves     = [l for l in self.leaves]
        return cpy

    def update_color(self, new_color, new_background_color = None):
        self.color = new_color
        if new_background_color is not None:
            self.background_color = new_background_color
        for leaf in self.leaves:
            leaf.update_color(self.color, self.background_color)
        for c in self.containers:
            c.update_color(self.color, self.background_color)

    def empty(self):
        for c in self.containers:
            if c is not None:
                c.delete()
        self.containers = []
        for l in self.leaves:
            if l is not None:
                l.delete()
        self.leaves = []

    def delete(self):
        for l in range(len(self.leaves)):
            if l is not None:
                self.leaves[l].delete()
                self.leaves[l] = None
        self.leaves = []

        for c in range(len(self.containers)):
            if c is not None:
                self.containers[c].delete()
                self.containers[c] = None
        self.containers = []
        super(Container, self).delete()

    def enable(self):
        for c in self.containers:
            c.enable()
        for l in self.leaves:
            l.enable()
        super(Container, self).enable()

    def disable(self):
        for c in self.containers:
            c.disable()
        for l in self.leaves:
            l.disable()
        super(Container, self).disable()

    def reset(self):
        for c in self.containers:
            c.reset()
        for l in self.leaves:
            l.reset()

    def remove_leaf(self, idx, delete_it = False):
        prev_leaves = [l for l in self.leaves]
        self.leaves = []
        for i in range(len(prev_leaves)):
            if i != idx:
                self.leaves.append(prev_leaves[i])
            elif delete_it:
                if prev_leaves[i] is not None:
                    prev_leaves[i].delete()

    def change_leaf(self, index, new_leaf, delete_old = True):
        prev = self.leaves[index]
        if delete_old:
            prev.delete()
        self.leaves[index] = new_leaf
        new_leaf.parent    = self

    def change_container(self, index, new_cont, delete_old = True):
        prev = self.containers[index]
        if delete_old:
            prev.delete()
        self.containers[index] = new_cont
        new_cont.parent        = self

    def add_leaf(self, leaf):
        leaf.parent = self
        self.leaves.append(leaf)

    def add_container(self, container):
        container.parent = self
        self.containers.append(container)

    def update_pos_and_dim(self):
        super(Container, self).update_pos_and_dim()
        for l in self.leaves:
            l.update_pos_and_dim()
        for c in self.containers:
            c.update_pos_and_dim()

    def draw(self, screen):
        if super(Container, self).draw(screen): # " if self.not_hidden ":
            for l in self.leaves:
                if not l.hidden:
                    l.draw(screen)
            for c in self.containers:
                if not c.hidden:
                    c.draw(screen)
            return True
        return False

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        for l in self.leaves:
            if l.listening_hover and l.point_is_inside(mouse_pos) and l.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
                return True
        for c in self.containers:
            if c.listening_hover and c.point_is_inside(mouse_pos) and c.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
                return True
        return False

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        for l in self.leaves:
            if l.listening_Lclick and l.point_is_inside(mouse_pos) and l.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
                return True
        for c in self.containers:
            if c.listening_Lclick and c.point_is_inside(mouse_pos) and c.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
                return True
        return False

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if mouse_button_status[0]:
            return True
        for l in self.leaves:
            if l.listening_Rclick and l.point_is_inside(mouse_pos) and l.propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
                return True
        for c in self.containers:
            if c.listening_Rclick and c.point_is_inside(mouse_pos) and c.propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
                return True
        if self.listening_Rclick is not None and self.notify_on_Rclick is not None and self.on_Rclick_listener is not None:
            return self.on_Rclick_listener.notify((self.name, windows, mouse_pos), to_redraw)
        return False

    def propagate_scroll(self, to_redraw, mouse_pos, scroll, pressed_special_keys):
        for l in self.leaves:
            if l.listening_scroll and l.point_is_inside(mouse_pos) and l.propagate_scroll(to_redraw, mouse_pos, scroll, pressed_special_keys):
                return True
        for c in self.containers:
            if c.listening_scroll and c.point_is_inside(mouse_pos) and c.propagate_scroll(to_redraw, mouse_pos, scroll, pressed_special_keys):
                return True
        return False
