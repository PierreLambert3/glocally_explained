from engine.gui.element import Element
from utils import luminosity_change
from engine.gui.window import Window
from engine.gui.listener import Listener, Gui_listener
import pygame as pg

class Element_with_value(Element):
    def __init__(self, position, dimension, name, parent, init_value, uid_generator, color=None, background_color=None, \
        on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None,\
        attach_to_parent=False):
        self.default_value = init_value
        self.value = init_value
        self.notify_on_value_change = False
        self.on_value_change = on_value_change_listener
        super(Element_with_value, self).__init__(position, dimension, name, parent, uid=uid_generator.get(), color=color, background_color=background_color,\
            on_draw_listener=on_draw_listener, on_Lclick_listener=on_Lclick_listener, on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener)
        if attach_to_parent:
            self.parent.add_leaf(self)

    def delete(self):
        self.default_value = None
        self.value = None
        if self.on_value_change is not None:
            self.on_value_change.delete()
            self.on_value_change = None
        super(Element_with_value, self).delete()

    def reset(self):
        self.value = self.default_value

    def update_listener_booleans(self):
        super(Element_with_value, self).update_listener_booleans()
        if self.on_value_change is not None:
            self.notify_on_value_change = True

    def add_listener(self, event_type, listener):
        if event_type in ["value", "value_change", "value change"]:
            self.on_value_change = listener
            self.notify_on_value_change = True
        else:
            super(Element_with_value, self).add_listener(event_type, listener)

    def get_value(self):
        return self.value

class Button(Element_with_value): # value is true or false
    def __init__(self, position, dimension, name, parent, uid_generator, init_value=False, show_txt=True, color=None, background_color=None,\
        on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None, attach_to_parent=False):
        super(Button, self).__init__(position, dimension, name, parent, uid_generator=uid_generator, init_value=init_value, color=color, background_color=background_color, \
            on_draw_listener=on_draw_listener, on_Lclick_listener=on_Lclick_listener, on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener, on_value_change_listener=on_value_change_listener, attach_to_parent=attach_to_parent)
        self.listen_to(["hover", "Lclick", "Rclick"])
        self.clickable = True
        self.beeing_pressed = False
        self.beeing_hovered = False
        if self.value: # if button is turned on: invert colors
            self.flip_colors()
        self.add_rect([(0,0), (1,1), 0, self.background_color])
        self.add_rect([(0,0), (1,1), 1, self.color])
        if show_txt:
            self.add_text([(0.5, 0.5), 9, (1,1), 18, name, self.color])  # [pos_pct, anchor_id, max_dim, font_size, string , color]

    def delete(self):
        super(Button, self).delete()

    def disable(self):
        super(Button, self).disable()
        self.beeing_hovered = False
        self.beeing_pressed = False
        self.update_appearance([])

    def set_unclickable(self, to_redraw):
        self.disable()
        self.hidden    = False
        self.clickable = False
        self.update_appearance(to_redraw)

    def set_clickable(self, to_redraw):
        self.enable()
        self.clickable = True
        self.update_appearance(to_redraw)

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.clickable and not self.beeing_hovered and not self.hidden:
            self.beeing_hovered = True
            self.update_appearance(to_redraw)
            self.schedule_awaiting(awaiting_mouse_move)
        return True

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.clickable and not self.hidden:
            self.beeing_pressed = True
            self.update_appearance(to_redraw)
            self.schedule_awaiting(awaiting_mouse_release)
        return True


    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        if self.stop_awaiting_hover or not self.clickable:
            self.stop_awaiting_hover = False
            return True
        elif self.beeing_pressed:
            return False
        elif not self.point_is_inside(mouse_positions[1]):
            if self.beeing_hovered:
                self.beeing_hovered = False
                self.update_appearance(to_redraw)
            return True
        return False

    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        self.beeing_pressed = False
        if self.stop_awaiting_click or not self.clickable:
            self.stop_awaiting_click = False
            return True
        elif not self.point_is_inside(release_pos):
            self.beeing_hovered = False
        else: # button was clicked
            if self.notify_on_value_change: # let the notifier change the value and appearance of the button
                self.on_value_change.notify(self.value, to_redraw)
            else: # toggle the button value and appearance
                self.value = not self.value
                self.flip_colors()
        if not self.deleted: # can get deleted during the call to: self.on_value_change.notify(self.value, to_redraw)
            self.update_appearance(to_redraw)
        return True

    def toggle(self, to_redraw=[]):
        self.value = not self.value
        self.flip_colors()
        self.update_appearance(to_redraw)

    def set(self, value):
        if not value == self.value:
            self.toggle()

    def update_appearance(self, to_redraw): # updates appearance and enqueues self to be redrawn
        if not self.clickable:
            background_c = luminosity_change(self.background_color, -180)
            main_c = luminosity_change(self.color, -120)
        elif self.beeing_pressed:
            background_c = luminosity_change(self.color, 180)
            main_c = luminosity_change(self.background_color, -180)
        elif self.beeing_hovered:
            background_c = luminosity_change(self.background_color, 40)
            main_c = luminosity_change(self.color, -180)
        else:
            background_c = self.background_color
            main_c = self.color
        self.rects_schematics[0][3] = background_c
        self.rects_schematics[1][3] = main_c
        if self.texts_schematics:
            self.texts_schematics[0][5] = main_c
            self.texts[0].color_change(main_c)
        self.schedule_draw(to_redraw)




class Number_selector(Element_with_value):
    def __init__(self, position, dimension, name, parent, uid_generator, nb_type, min_val, max_val, step, init_val, default, color=None, background_color=None,\
        on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None,attach_to_parent=False):
        # btn_uids = uid_generator.get_N(3)
        super(Number_selector, self).__init__(position, dimension, name, parent, init_val, uid_generator=uid_generator, color=color, background_color=background_color, \
            on_draw_listener=on_draw_listener, on_Lclick_listener=on_Lclick_listener, on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener, on_value_change_listener=on_value_change_listener, attach_to_parent=attach_to_parent)
        self.listen_to(["hover", "Lclick", "Rclick"])
        self.is_int = False
        if nb_type == "int":
            self.is_int = True
        self.default = default # can be of other type than int
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.init_val = init_val
        if not (isinstance(init_val, int) or isinstance(init_val, float)):
            self.init_val = min_val + int(0.5*(max_val-min_val)/step)*step
        self.cursor_value = self.init_val
        self.real_value = default # can be a string or a number
        self.is_default = True
        self.add_text([(0.05, 0.2), 1, (0.65, 0.4), 16, name+" :", self.color])
        if type(self.real_value) == float and self.real_value < 0.01:
            self.add_text([(0.75, 0.2), 1, (0.65, 0.4), 16, str("{:.3e}".format(self.real_value)), self.color])
        else:
            self.add_text([(0.75, 0.2), 1, (0.65, 0.4), 16, str(self.real_value), self.color])
        self.add_rect([(0,0), (1,1), 0, self.background_color])
        self.add_rect([(0.07, 0.8), (0.84, 0.05), 1, self.color])
        l_btn_listener = Gui_listener("<", identifier=0, listener_class=0, to_notify = [self])
        r_btn_listener = Gui_listener(">", identifier=1, listener_class=0, to_notify = [self])
        default_btn_listener = Gui_listener("default", identifier=1, listener_class=1, to_notify = [self])
        self.l_btn       = Button((0.1, 0.4),  (0.15, 0.2), "<", self, uid_generator=uid_generator,on_value_change_listener=l_btn_listener)
        self.r_btn       = Button((0.75, 0.4), (0.15, 0.2), ">", self, uid_generator=uid_generator, on_value_change_listener=r_btn_listener)
        self.default_btn = Button((0.4, 0.4), (0.2, 0.2), "default", self, uid_generator=uid_generator, on_value_change_listener=default_btn_listener)
        self.cursor_mid = self.abs_pos[0] + self.dim[0]*(0.07 + (0.84*(self.init_val-self.min_val)/(self.max_val-self.min_val)))
        self.cursor_top = self.abs_pos[1]+0.6*self.dim[1]
        self.cursor_dim = (max(8,self.dim[0]*0.04), self.dim[1]*0.4)
        self.x_delta_from_cursor_mid = 0
        self.cursor_hovered = False
        self.beeing_dragged = False
        self.disable_buttons = False

    def delete(self):
        if self.l_btn is not None:
            self.l_btn.delete()
            self.l_btn = None
        if self.r_btn is not None:
            self.r_btn.delete()
            self.r_btn = None
        if self.default_btn is not None:
            self.default_btn.delete()
            self.default_btn = None
        super(Number_selector, self).delete()

    def disable(self):
        super(Number_selector, self).disable()
        self.l_btn.disable()
        self.r_btn.disable()
        self.default_btn.disable()

    def enable(self):
        super(Number_selector, self).enable()
        if not self.disable_buttons:
            self.l_btn.enable()
            self.r_btn.enable()
            self.default_btn.enable()

    def get_value(self):
        return self.real_value

    def draw(self, screen):
        if self.hidden:
            return False
        super(Number_selector, self).draw(screen)
        self.l_btn.draw(screen)
        self.r_btn.draw(screen)
        self.default_btn.draw(screen)
        if not self.cursor_hovered:
            c1 = self.background_color
            c2 = self.color
        else:
            c1 = self.color
            c2 = self.background_color
        pg.draw.rect(screen, c1, pg.Rect((self.cursor_mid-0.5*self.cursor_dim[0], self.cursor_top), self.cursor_dim), 0)
        pg.draw.rect(screen, c2, pg.Rect((self.cursor_mid-0.5*self.cursor_dim[0], self.cursor_top), self.cursor_dim), 1)
        pg.draw.rect(screen, c2, pg.Rect((self.cursor_mid-0.5*self.cursor_dim[0]+3, self.cursor_top+3),( self.cursor_dim[0]-6, self.cursor_dim[1]-6)), 0)
        return True

    def get_notified(self, listener_class, listener_id, listener_value, to_redraw):
        if listener_class == 0:
            if listener_id == 0:
                self.cursor_value = max(self.min_val, self.cursor_value-self.step)
            else:
                self.cursor_value = min(self.max_val, self.cursor_value+self.step)
            self.real_value = self.cursor_value
            self.is_default = False
        else: # set to default value
            self.cursor_value = self.init_val
            self.real_value = self.default
            self.is_default = True
        self.update_cursor_pos()
        self.update_text()
        self.schedule_draw(to_redraw)
        if self.notify_on_value_change:
            self.on_value_change.notify(self.get_value(), to_redraw)

    def update_cursor_pos(self):
        self.cursor_mid = self.abs_pos[0] + self.dim[0]*(0.07 + (0.84*(self.cursor_value-self.min_val)/(self.max_val-self.min_val)))


    def update_text(self):
        if type(self.real_value) == float and self.real_value < 0.01:
            self.texts_schematics[1][4] = str("{:.3e}".format(self.real_value))
        elif type(self.real_value) == float:
            self.texts_schematics[1][4] = str(round(self.real_value, 3))
        else:
            self.texts_schematics[1][4] = str(self.real_value)
        self.rebuild_texts()

    def update_pos_and_dim(self):
        super(Number_selector, self).update_pos_and_dim()
        self.update_cursor_pos()
        self.cursor_top = self.abs_pos[1]+0.6*self.dim[1]
        self.cursor_dim = (max(8,self.dim[0]*0.04), self.dim[1]*0.4)
        self.l_btn.update_pos_and_dim()
        self.r_btn.update_pos_and_dim()
        self.default_btn.update_pos_and_dim()


    def cursor_movement(self, mouse_x, notify = False):
        self.is_default = False
        self.cursor_mid = min(max(self.abs_pos[0]+self.dim[0]*0.07, self.cursor_mid + (mouse_x-self.x_delta_from_cursor_mid)-(self.cursor_mid)), self.abs_pos[0]+self.dim[0]*0.93)
        steps_in_interval = int((self.max_val-self.min_val)/self.step)
        adj_mid = self.cursor_mid+0.5*(0.86*self.dim[0]/steps_in_interval) # because of integer rounding, adj mid puts and offset
        ratio = min(max(0, (adj_mid-(self.abs_pos[0]+0.07*self.dim[0]))/(0.84*self.dim[0])),1)
        nb_steps = int(ratio*steps_in_interval)
        if self.is_int:
            self.cursor_value = int(self.min_val+nb_steps*self.step)
        else:
            self.cursor_value = self.min_val+nb_steps*self.step
        self.real_value = self.cursor_value
        self.cursor_mid = self.abs_pos[0]+self.dim[0]*0.07+((self.cursor_value-self.min_val)/(self.max_val-self.min_val))*self.dim[0]*0.84
        self.update_text()
        if notify:
            self.on_value_change.notify(self.get_value(), [])

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if not self.cursor_hovered and mouse_pos[1] > self.cursor_top:
            if mouse_pos[0] > self.cursor_mid-0.5*self.cursor_dim[0] and mouse_pos[0] < self.cursor_mid+0.5*self.cursor_dim[0]:
                self.cursor_hovered = True
                self.schedule_awaiting(awaiting_mouse_move)
                self.schedule_draw(to_redraw)
        else:
            if self.l_btn.point_is_inside(mouse_pos) and self.l_btn.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
                return True
            if self.r_btn.point_is_inside(mouse_pos) and self.r_btn.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
                return True
            if self.default_btn.point_is_inside(mouse_pos) and self.default_btn.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
                return True
        return True

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        if self.beeing_dragged:
            self.cursor_movement(mouse_positions[1][0], notify=False)
            self.schedule_draw(to_redraw)
            return False
        if mouse_positions[1][1] > self.cursor_top and mouse_positions[1][1] < self.cursor_top+self.cursor_dim[1]:
            if mouse_positions[1][0] > self.cursor_mid-0.5*self.cursor_dim[0] and mouse_positions[1][0] < self.cursor_mid+0.5*self.cursor_dim[0]:
                return False
        self.cursor_hovered = False
        self.schedule_draw(to_redraw)
        return True

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.l_btn.point_is_inside(mouse_pos) and self.l_btn.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
            return True
        elif self.r_btn.point_is_inside(mouse_pos) and self.r_btn.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
            return True
        elif self.default_btn.point_is_inside(mouse_pos) and self.default_btn.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
            return True
        elif (not self.beeing_dragged or not self.cursor_hovered) and mouse_pos[1] > self.cursor_top:
            if mouse_pos[0] > self.cursor_mid-0.5*self.cursor_dim[0] and mouse_pos[0] < self.cursor_mid+0.5*self.cursor_dim[0]:
                self.x_delta_from_cursor_mid = mouse_pos[0] - self.cursor_mid
                self.beeing_dragged = True
                self.schedule_awaiting(awaiting_mouse_move)
                self.schedule_awaiting(awaiting_mouse_release)
                self.schedule_draw(to_redraw)

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.l_btn.point_is_inside(mouse_pos) and self.l_btn.propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
            return True
        elif self.r_btn.point_is_inside(mouse_pos) and self.r_btn.propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
            return True
        elif self.default_btn.point_is_inside(mouse_pos) and self.default_btn.propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
            return True
        elif (not self.beeing_dragged or not self.cursor_hovered) and mouse_pos[1] > self.cursor_top:
            self.x_delta_from_cursor_mid = 0
            self.cursor_movement(mouse_pos[0], notify=False)
            self.schedule_draw(to_redraw)
            self.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release)


    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        self.beeing_dragged = False
        if self.notify_on_Lclick:
            self.on_Lclick_listener.notify(self.get_value(), to_redraw)
        if self.notify_on_Rclick:
            self.on_Rclick_listener.notify(self.get_value(), to_redraw)
        if self.notify_on_value_change:
            self.on_value_change.notify(self.get_value(), to_redraw)
        return True

class String_selector(Element_with_value):
    def __init__(self, position, dimension, name, parent, uid_generator, values, default_value, color=None, background_color=None,\
        on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None,attach_to_parent=False):
        super(String_selector, self).__init__(position, dimension, name, parent, default_value, uid_generator=uid_generator, color=color, background_color=background_color, \
            on_draw_listener=on_draw_listener, on_Lclick_listener=on_Lclick_listener,on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener, on_value_change_listener=on_value_change_listener, attach_to_parent=attach_to_parent)
        self.listen_to(["hover", "Lclick"])
        self.uid_generator = uid_generator
        self.values = values
        self.default_value = default_value
        self.selected_idx = 0
        for idx, v in enumerate(values):
            if v == default_value:
                self.selected_idx = idx
        self.add_text([(0.05, 0.2), 1, (0.65, 0.4), 16, name+" :", color])
        other_options = [v for v in values if v != default_value]
        self.options_window = Window((0, 0), (250, 300), "options_window", uid_generator=uid_generator, close_on_click_outside=True, close_on_notify=True, color=self.color, background_color=self.background_color)
        # the pane that opens when we click on the value
        mutex_on_value_change = Listener(listener_class=3, to_notify=[self])
        self.mutex_in_window = Mutex_choice((0,0), (1,1), "options", self.options_window, nb_col=1, labels=other_options, uid_generator=uid_generator, selected_idx=-1, on_value_change_listener=mutex_on_value_change)
        self.options_window.add_leaf(self.mutex_in_window)
        # the selected value that opens a pane when clicked
        self.currently_selected_btn = Button((0.05, 0.5), (0.9, 0.3), self.default_value, self,uid_generator=uid_generator)
        self.options_window.abs_pos = (self.currently_selected_btn.abs_pos[0], self.currently_selected_btn.abs_pos[1]+self.currently_selected_btn.dim[1])
        self.options_window.update_pos_and_dim()

    def delete(self):
        if self.options_window is not None:
            self.options_window.delete()
            self.options_window = None
        if self.mutex_in_window is not None:
            self.mutex_in_window.delete()
            self.mutex_in_window = None
        if self.currently_selected_btn is not None:
            self.currently_selected_btn.delete()
            self.currently_selected_btn = None
        super(String_selector, self).delete()

    def disable(self):
        super(String_selector, self).disable()
        self.mutex_in_window.disable()
        self.currently_selected_btn.disable()

    def enable(self):
        super(String_selector, self).enable()
        self.mutex_in_window.enable()
        self.currently_selected_btn.enable()

    def get_value(self):
        return self.values[self.selected_idx]

    def update_pos_and_dim(self):
        super(String_selector, self).update_pos_and_dim()
        self.currently_selected_btn.update_pos_and_dim()
        if self.options_window is not None and not self.options_window.deleted:
            self.options_window.abs_pos = (self.currently_selected_btn.abs_pos[0], self.currently_selected_btn.abs_pos[1]+self.currently_selected_btn.dim[1])
            self.options_window.dim = (self.currently_selected_btn.dim[0], (len(self.values)-1)*self.currently_selected_btn.dim[1])
            self.options_window.update_pos_and_dim()

    def draw(self, screen):
        if self.hidden:
            return False
        self.texts[0].draw(screen)
        self.currently_selected_btn.draw(screen)
        return True

    def get_notified(self, listener_class, listener_id, listener_value, to_redraw):
        if listener_class == 3 and not listener_value is None: # choice done, and window closed
            for idx, v in enumerate(self.values):
                if v == listener_value:
                    self.selected_idx = idx
                    self.currently_selected_btn.texts_schematics[0][4] = v
                    self.currently_selected_btn.beeing_pressed = False
                    self.currently_selected_btn.beeing_hovered = False
                    self.currently_selected_btn.rebuild_texts()
                    self.currently_selected_btn.update_appearance([])
                    other_options = [vv for vv in self.values if vv != v]
                    self.options_window.close(to_redraw, listener_id)
                    # self.options_window.leaves[0].delete() # necessary in case some remain in awaiting_X
                    del self.options_window.leaves[0]
                    mutex_on_value_change = Listener(listener_class=3, to_notify=[self])
                    self.mutex_in_window  = Mutex_choice((0,0), (1,1), "options", self.options_window, nb_col=1, labels=other_options, uid_generator=self.uid_generator, selected_idx=-1, on_value_change_listener=mutex_on_value_change)
                    self.options_window.add_leaf(self.mutex_in_window)
                    self.schedule_draw(to_redraw)

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.currently_selected_btn.point_is_inside(mouse_pos) and self.currently_selected_btn.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
            return True
        return True

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.currently_selected_btn.point_is_inside(mouse_pos) and self.currently_selected_btn.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
            # self.options_window.reset()
            self.options_window = Window((0, 0), (250, 300), "options_window", uid_generator=self.uid_generator, close_on_click_outside=True, close_on_notify=True, color=self.color, background_color=self.background_color)
            self.options_window.abs_pos = (self.currently_selected_btn.abs_pos[0], self.currently_selected_btn.abs_pos[1]+self.currently_selected_btn.dim[1])
            self.options_window.dim = (self.currently_selected_btn.dim[0], (len(self.values)-1)*self.currently_selected_btn.dim[1])
            if self.mutex_in_window is not None:
                self.mutex_in_window.delete()
            other_options = [vv for vv in self.values if vv != self.values[self.selected_idx]]
            mutex_on_value_change = Listener(listener_class=3, to_notify=[self])
            self.mutex_in_window = Mutex_choice((0,0), (1,1), "options", self.options_window, nb_col=1, labels=other_options, uid_generator=self.uid_generator, selected_idx=-1, on_value_change_listener=mutex_on_value_change)
            self.options_window.add_leaf(self.mutex_in_window)

            self.options_window.update_pos_and_dim()
            windows.append(self.options_window)
        return True

class Mutex_with_title(Element_with_value):
    def __init__(self, position, dimension, name, parent, uid_generator, nb_col, labels, selected_idx=0,\
        color=None, background_color=None, on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None, attach_to_parent=False):
        super(Mutex_with_title, self).__init__(position, dimension, name, parent, labels[selected_idx], uid_generator=uid_generator, color=color, background_color=background_color, \
            on_draw_listener=on_draw_listener, on_Lclick_listener=on_Lclick_listener, on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener, on_value_change_listener=on_value_change_listener, attach_to_parent=attach_to_parent)
        self.listen_to(["hover", "Lclick"])
        on_mutex_value_change = Gui_listener("mutex_value_changed", identifier=9, listener_class=73964, to_notify=[self])
        self.mutex = Mutex_choice((0, 0.4), (1, 0.6), "mutex", self, nb_col=nb_col, labels=labels, uid_generator=uid_generator, selected_idx=selected_idx, \
            color=color, background_color=background_color, on_draw_listener=None, on_Rclick_listener=None, on_Lclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=on_mutex_value_change, attach_to_parent=False)
        self.add_text([(0.05, 0.2), 1, (0.65, 0.4), 16, name+" :", self.color])

    def delete(self):
        self.mutex.delete()
        super(Mutex_with_title, self).delete()

    def disable(self):
        super(Mutex_with_title, self).disable()
        self.mutex.disable()

    def enable(self):
        super(Mutex_with_title, self).enable()
        self.mutex.enable()

    def get_value(self):
        return self.mutex.get_value()

    def get_notified(self, listener_class, listener_id, listener_value, to_redraw):
        if listener_class == 73964: # my mutex changed value
            if self.notify_on_value_change:
                self.on_value_change.notify(self.get_value(), to_redraw)

    def draw(self, screen):
        self.already_drawn = True
        if self.hidden:
            return False
        self.texts[0].draw(screen)
        self.mutex.draw(screen)
        return True

    def reset(self):
        self.mutex.reset()

    def update_pos_and_dim(self):
        super(Mutex_with_title, self).update_pos_and_dim()
        self.mutex.update_pos_and_dim()

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.mutex.point_is_inside(mouse_pos):
            self.mutex.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move)
        return True

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.mutex.point_is_inside(mouse_pos):
            self.mutex.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release)
        return True

class Mutex_choice(Element_with_value):
    def __init__(self, position, dimension, name, parent, uid_generator, nb_col, labels, selected_idx=0, nb_row = 0, ignore_with_ctrl=True, \
        color=None, background_color=None, on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None, attach_to_parent=False):
        init_value = None
        if not selected_idx == -1:
            init_value = labels[selected_idx]
        super(Mutex_choice, self).__init__(position, dimension, name, parent, uid_generator=uid_generator, init_value=init_value, color=color, background_color=None, \
            on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=on_value_change_listener,\
            attach_to_parent=attach_to_parent)
        self.listen_to(["hover", "Lclick", "Rclick"])
        self.buttons  = []
        self.labels   = labels
        self.initially_selected_idx = selected_idx
        self.selected_idx = selected_idx
        self.ignore_with_ctrl = ignore_with_ctrl
        if nb_col > 0:
            nb_row = max(1, int(len(labels)/nb_col))
            if len(labels) % nb_col > 0:
                nb_row += 1
        else:
            nb_col = len(labels)
            nb_row = 1
        btn_nb = 0
        for r in range(nb_row):
            for c in range(nb_col):
                if btn_nb < len(self.labels):
                    btn_listener = Gui_listener(identifier=btn_nb, listener_class=73963, to_notify=[self])
                    self.buttons.append(Button((c/nb_col, r/nb_row), (1/nb_col,1/nb_row), self.labels[btn_nb], self,uid_generator=uid_generator,\
                        on_value_change_listener=btn_listener, attach_to_parent=False,  color=self.color, background_color=self.background_color))
                btn_nb += 1
        self.value = None
        if not selected_idx == -1:
            self.buttons[selected_idx].set(True)
            self.value = self.labels[selected_idx]

    def disable_option(self, option_name):
        for b in self.buttons:
            if b.name == option_name:
                b.set_unclickable([])

    def enable_option(self, option_name):
        for b in self.buttons:
            if b.name == option_name:
                b.set_clickable([])

    def update_color(self, new_color, new_background_color = None):
        self.color = new_color
        if new_background_color is not None:
            self.background_color = new_background_color
        for btn in self.buttons:
            btn.color = new_color
            btn.background_color = self.background_color

    def set(self, value):
        for i in range(len(self.labels)):
            if value == self.labels[i]:
                self.change_value(i, [])

    def delete(self):
        for b in range(len(self.buttons)):
            if self.buttons[b] is not None:
                self.buttons[b].delete()
                self.buttons[b] = None
        self.labels = None
        self.value  = None
        super(Mutex_choice, self).delete()

    def disable(self):
        for b in self.buttons:
            if b is not None:
                b.disable()

    def enable(self):
        for b in self.buttons:
            if b is not None:
                b.enable()

    def draw(self, screen):
        self.already_drawn = True
        if self.hidden:
            return False
        for b in self.buttons:
            b.draw(screen)
        return True

    def update_pos_and_dim(self):
        super(Mutex_choice, self).update_pos_and_dim()
        for b in self.buttons:
            b.update_pos_and_dim()

    def unselect_all(self):
        self.selected_idx = -1
        for b in self.buttons:
            if b.value:
                b.toggle()

    def change_value(self, new_value, to_redraw):
        if not self.selected_idx == -1:
                self.buttons[self.selected_idx].toggle(to_redraw)
        self.buttons[new_value].toggle(to_redraw)
        self.selected_idx = new_value
        self.value = self.labels[self.selected_idx]

    def get_notified(self, listener_class, listener_id, listener_value, to_redraw):
        if listener_id == self.selected_idx:
            return
        else: # value change
            self.change_value(listener_id, to_redraw)
            if self.notify_on_value_change:
                self.on_value_change.notify(self.value, to_redraw)

    def propagate_Lmouse_down(self,to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.ignore_with_ctrl and pressed_special_keys[0]:
            return False
        for b in self.buttons:
            if b.listening_Lclick and b.point_is_inside(mouse_pos):
                b.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release)
                return True

    def propagate_Rmouse_down(self,to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.ignore_with_ctrl and pressed_special_keys[0]:
            return False
        for b in self.buttons:
            if b.listening_Rclick and b.point_is_inside(mouse_pos):
                b.propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release)
                if self.notify_on_Rclick:
                    self.on_Rclick_listener.notify(b.name, to_redraw)
                return True

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.ignore_with_ctrl and pressed_special_keys[0]:
            return False
        for b in self.buttons:
            if b.listening_hover and b.point_is_inside(mouse_pos):
                b.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move)
                return True
        return False

    def reset(self):
        self.selected_idx = self.initially_selected_idx
        if not self.selected_idx == -1:
            for idx, b in enumerate(self.buttons):
                b.set(idx == self.selected_idx)
            self.value = self.labels[self.selected_idx]
        else:
            for idx, b in enumerate(self.buttons):
                b.set(False)
            self.value = None

class Scrollable_bundle(Element):
    def __init__(self, position, dimension, name, parent, uid_generator, items=[], color=None, background_color=None,\
        on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None,attach_to_parent=False):
        uid = uid_generator.get()
        super(Scrollable_bundle, self).__init__(position, dimension, name, parent, uid=uid, color=color, background_color=background_color, \
            on_draw_listener=on_draw_listener, on_Lclick_listener=on_Lclick_listener, on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener, attach_to_parent=attach_to_parent)
        self.listen_to(["hover", "Lclick", "Rclick", "scroll"])
        self.items = items
        self.background_rect = pg.Rect(self.abs_pos, self.dim)
        self.add_text([(0.5,0.03), 2, (0.8,0.3), 18, self.name, self.color])
        self.add_rect([(0,0), (1,0.1), 0, self.background_color]) # top mask
        self.add_rect([(0,0.9), (1,0.1), 0, self.background_color]) # bottom mask

    def delete(self):
        for itm in self.items:
            itm.delete()
        self.items = None
        super(Scrollable_bundle, self).delete()

    def disable(self):
        for i in self.items:
        	i.disable()
        super(Scrollable_bundle, self).disable()


    def set_items(self, items, new_title="new title"):
        for idx in range(len(items)):
            items[idx].dim_pct = (1, 0.1)
            items[idx].pos_pct = (0, 0.1 + idx*0.1)
            items[idx].update_pos_and_dim()
        self.items = items
        self.already_drawn = False
        self.texts_schematics[0][4] = new_title
        self.name = new_title
        self.rebuild_texts()

    def get_values(self):
        values = {}
        for itm in self.items:
            values[itm.name] = itm.get_value()
        return values

    def get_notified(self, var_class, var_id, var_value, to_redraw):
        self.schedule_draw(to_redraw)

    def draw(self, screen):
        self.already_drawn = True
        if self.hidden:
            return False
        pg.draw.rect(screen, self.background_color, self.background_rect,  0) # background rect is necessary
        for itm in self.items:
            if itm.pos_pct[1] >= 0 and itm.pos_pct[1] < 0.9:
                itm.draw(screen)
        super(Scrollable_bundle, self).draw(screen) # draw the masks last
        return True

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if mouse_pos[1] >= self.pct_to_px((0,0.1))[1] and mouse_pos[1] <= self.pct_to_px((0,0.9))[1]:
            for itm in self.items:
                if itm.point_is_inside(mouse_pos) and itm.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
                    return True
        return True

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if mouse_pos[1] >= self.pct_to_px((0,0.1))[1] and mouse_pos[1] <= self.pct_to_px((0,0.9))[1]:
            for itm in self.items:
                if itm.point_is_inside(mouse_pos) and itm.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
                    return True
        return True

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if mouse_pos[1] >= self.pct_to_px((0,0.1))[1] and mouse_pos[1] <= self.pct_to_px((0,0.9))[1]:
            for itm in self.items:
                if itm.point_is_inside(mouse_pos) and itm.propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
                    return True
        return True

    def propagate_scroll(self, to_redraw, mouse_pos, scroll, pressed_special_keys):
        if len(self.items) == 0:
            return True
        scroll_multiplier = 0.1 # 0.035
        if scroll > 0: # scrolling up (items going down)
            if self.items[0].pos_pct[1] >= 0.1:
                return True
            if self.items[0].pos_pct[1]+scroll*scroll_multiplier > 0.1:
                scroll = 0.1 - self.items[0].pos_pct[1]
            else:
                scroll = scroll*scroll_multiplier
        if scroll < 0 : # scrolling down (items going up)
            if self.items[-1].pos_pct[1] <= 0.8:
                return True
            if self.items[-1].pos_pct[1]+scroll*scroll_multiplier < 0.8:
                scroll = 0.8 - self.items[-1].pos_pct[1]
            else:
                scroll = scroll*scroll_multiplier
        for itm in self.items:
            itm.pos_pct = (0, itm.pos_pct[1]+scroll)
            if itm.pos_pct[1] >= 0 and itm.pos_pct[1] < 0.9:
                itm.enable()
                itm.update_pos_and_dim()
            else:
                itm.disable()
        self.schedule_draw(to_redraw)
        return True
