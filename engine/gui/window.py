from engine.gui.event_ids import WINDOW_CLOSE, CLOSE_YOURSELF
from engine.gui.listener import Listener
from engine.gui.container import Container
import pygame


class Window(Container):
    def __init__(self, pixel_pos, pixel_dim, name, close_on_click_outside, uid_generator, color=(0,255,100), background_color=(200,100,10),\
        listening_hover=False, listening_Lclick=False, listening_Rclick=False, listening_scroll=False, close_on_notify = True):
        super(Window, self).__init__(pixel_pos, pixel_dim, name, None, uid_generator=uid_generator, color=color, background_color=background_color, listening_hover=listening_hover, listening_Lclick=listening_Lclick, listening_Rclick=listening_Rclick, listening_scroll=listening_scroll)
        self.close_on_click_outside = close_on_click_outside
        self.close_on_notify = close_on_notify
        self.awaiting_mouse_release = []
        self.awaiting_mouse_move    = []
        self.awaiting_key_press     = []
        self.on_close_listener      = Listener(listener_class=WINDOW_CLOSE, to_notify = [])
        self.closing = False

    def set_initial_positions(self, position, dimension):
        self.dim_pct = (1, 1)
        self.dim     = dimension
        self.pos_pct = (0,0)
        self.rel_pos = (0,0)
        self.abs_pos = position

    def open(self, abs_pos, windows_list):
        if not self.is_already_in(windows_list):
            self.closing = False
            self.abs_pos = abs_pos
            self.update_pos_and_dim()
            self.reset()
            windows_list.append(self)

    def delete(self):
        for e in range(len(self.awaiting_mouse_release)):
            if self.awaiting_mouse_release[e] is not None:
                self.awaiting_mouse_release[e].delete()
                self.awaiting_mouse_release[e] = None
        for e in range(len(self.awaiting_mouse_move)):
            if self.awaiting_mouse_move[e] is not None:
                self.awaiting_mouse_move[e].delete()
                self.awaiting_mouse_move[e] = None
        for e in range(len(self.awaiting_key_press)):
            if self.awaiting_key_press[e] is not None:
                self.awaiting_key_press[e].delete()
                self.awaiting_key_press[e] = None
        if self.on_close_listener is not None:
            self.on_close_listener.delete()
            self.on_close_listener = None
        self.awaiting_mouse_move = []
        self.awaiting_mouse_release = []
        self.awaiting_key_press = []
        super(Window, self).delete()

    def close(self, to_redraw, value):
        self.closing = True
        # self.awaiting_mouse_release = []
        # self.awaiting_mouse_move    = []
        # self.awaiting_key_press     = []
        # if self.on_close_listener is not None:
        #     self.on_close_listener.notify(value, to_redraw)

    def get_notified(self, listener_class, listener_id, listener_value, to_redraw):
        if self.close_on_notify:
            self.close(to_redraw, listener_value)
        # if listener_class == CLOSE_YOURSELF:
        #     if listener_value != 'dont notify':
        #         self.close(to_redraw, listener_value)
        # else:
        #     print("Window class -> unrecognised event class :", listener_class)

    def update_pos_and_dim(self):
        self.bounding_rect = pygame.Rect(self.abs_pos, self.dim)
        for l in self.leaves:
            l.update_pos_and_dim()
        for c in self.containers:
            c.update_pos_and_dim()

    def draw(self, screen):
        pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
        for l in self.leaves:
            l.draw(screen)
        for c in self.containers:
            c.draw(screen)
        return True

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys):
        super(Window, self).propagate_hover(to_redraw, mouse_pos, pressed_special_keys, self.awaiting_mouse_move)

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys):
        if self.close_on_click_outside and not self.point_is_inside(mouse_pos):
            self.close(to_redraw, None)
        else:
            super(Window, self).propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, self.awaiting_mouse_move, self.awaiting_mouse_release)

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys):
        if self.close_on_click_outside and not self.point_is_inside(mouse_pos):
            self.close(to_redraw, None)
        else:
            super(Window, self).propagate_Rmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, self.awaiting_mouse_move, self.awaiting_mouse_release)

    def propagate_scroll(self, to_redraw, mouse_pos, scroll, pressed_special_keys):
        super(Window, self).propagate_scroll(to_redraw, mouse_pos, scroll, pressed_special_keys)

    def on_awaited_mouse_release(self, to_redraw,release_pos, released_buttons, pressed_special_keys):
        idxs = [] # to be removed from awaiting_mouse_release
        for i, e in enumerate(self.awaiting_mouse_release):
            # if not e.deleted and e.on_awaited_mouse_release(to_redraw, release_pos, released_buttons, pressed_special_keys):
            if e.stop_awaiting_click:
                e.stop_awaiting_click = False
                idxs.append(i)
            elif e.on_awaited_mouse_release(to_redraw, release_pos, released_buttons, pressed_special_keys):
                idxs.append(i)
        for i in range(1, len(idxs)+1): # remove the ones that should be removed
            del self.awaiting_mouse_release[idxs[-i]]

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        idxs = []
        for i, e in enumerate(self.awaiting_mouse_move):
            # if not e.deleted and e.on_awaited_mouse_move(to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
            if e.stop_awaiting_hover:
                e.stop_awaiting_hover = False
                idxs.append(i)
            elif e.on_awaited_mouse_move(to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
                idxs.append(i)
        for i in range(1, len(idxs)+1):
            del self.awaiting_mouse_move[idxs[-i]]

    def on_awaited_key_press(self, to_redraw, windows, pressed_keys, pressed_special_keys):
        if "es" in pressed_keys:
            self.close(to_redraw, None)
        else:
            idxs = []
            for i, e in enumerate(self.awaiting_key_press):
                if not e.deleted and e.on_awaited_key_press(to_redraw, pressed_keys, pressed_special_keys):
                    idxs.append(i)
            for i in range(1, len(idxs)+1):# remove the ones that should be removed
                del self.awaiting_key_press[idxs[-i]]
