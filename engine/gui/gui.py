import numpy as np
import pygame
import time
from engine.gui.event_ids import CTRL_KEY_CHANGE
import pygame

class Id_generator:
    def __init__(self):
        self.counter = 0

    def get(self):
        self.counter += 1
        return self.counter

    def get_N(self, N):
        ids = []
        for i in range(N):
            self.counter += 1
            ids.append(self.counter)
        return ids

class Gui:
    # resolution parameter is overritten by hardware screen resolution if fullscreen!!
    def __init__(self, screen, theme, config, running_flag, main_screen):
        self.theme        = theme
        self.frame_time   = config["frame time"]
        self.resolution   = config["resolution"]
        self.ctrl_pressed = False
        self.screen       = screen
        self.shd_running_bool = running_flag

        self.main_manager, self.main_window  = main_screen["manager"], main_screen["window"]

        self.key_to_manager  = {
                                config["main screen key"]  : (self.main_manager, self.main_window, "main"),
                                }
        self.screen_change_keys = [key for key in self.key_to_manager]
        self.name_to_manager = {
                                "main"  : (self.main_manager, self.main_window),
                                }
        self.current_manager = self.main_manager

    def change_manager(self, name):
        self.current_manager.sleep()
        prev_manager = self.current_manager
        self.current_manager = self.name_to_manager[name][0]
        self.current_manager.wake_up(prev_manager)

    # whatever needs to be done for gracefull exit
    def end(self):
        self.shd_running_bool.set(False)
        self.main_manager.end()
        pygame.quit()


    def routine(self):
        self.change_manager("main")
        open_windows    = [self.name_to_manager["main"][1]] # works as a stack
        nb_open_windows = len(open_windows)
        open_windows_for_each_screen = {}
        for screen_name in self.name_to_manager:
            open_windows_for_each_screen[screen_name] = [self.name_to_manager[screen_name][1]]
        top_window = open_windows[-1]

        usr_events = User_events()
        open_windows[0].draw(self.screen)
        pygame.display.flip()
        iter_num = 0
        to_redraw = []
        iter_time = 100*self.frame_time
        while not usr_events.stop_signal:
            t = time.time()
            # user events
            mouse_mvd, mouse_press, mouse_rel_lmb, mouse_rel_rmb, key_press, spec_key_rel, wheeled  = usr_events.update()

            # change the shown screen if appropriate shortcut is pressed
            if key_press and (usr_events.pressed_keys[0] in self.screen_change_keys):
                new_manager, new_window, new_name = self.key_to_manager[usr_events.pressed_keys[0]]
                if not new_name == self.current_manager.name:
                    prev_screen   = open_windows[0]
                    prev_name     = self.current_manager.name
                    prev_screen.disable()
                    new_window.enable()
                    self.change_manager(new_name)
                    open_windows_for_each_screen[prev_name] = [e for e in open_windows] # save the previously opened windows in previous screen
                    open_windows = [e for e in open_windows_for_each_screen[new_name]]
                    top_window = open_windows[-1]
                    nb_open_windows = len(open_windows)
                    to_redraw.append(top_window)

            # ctrl key: general event
            if usr_events.ctrl_pressed:
                if not self.ctrl_pressed:
                    self.ctrl_pressed = True
                    self.current_manager.get_notified(CTRL_KEY_CHANGE, True,(True, usr_events.curr_mouse_pos), to_redraw)
            elif self.ctrl_pressed: # ctrl released
                self.ctrl_pressed = False
                self.current_manager.get_notified(CTRL_KEY_CHANGE, False, (False, usr_events.curr_mouse_pos), to_redraw)

            # manager asked for some redraws but didn't have to_redraw array (occurs when notified by another thread)
            if self.current_manager.asked_a_redraw:
                to_redraw.extend(self.current_manager.to_redraw_on_next_iter)
                self.current_manager.asked_a_redraw = False
                self.current_manager.to_redraw_on_next_iter.clear()
                if len(open_windows) > 1: # if right click window: redaw it on top of the newly drawn thing
                    to_redraw.append(top_window)

            # propagating events under the mouse, this can add elements to awaiting_* lists
            if mouse_mvd: #  not redundant with on_awaited_mouse_move: in propagate_ we send to elements aimed by the mouse that aren't necessarily in awaiting_* list
                top_window.propagate_hover(to_redraw=to_redraw,mouse_pos=usr_events.curr_mouse_pos, pressed_special_keys=usr_events.pressed_special_keys())
            if mouse_press:
                mouse_btns = usr_events.pressed_mouse_buttons()
                if mouse_btns[0]:
                    top_window.propagate_Lmouse_down(to_redraw=to_redraw,windows=open_windows,mouse_pos=usr_events.curr_mouse_pos, mouse_button_status=mouse_btns, pressed_special_keys=usr_events.pressed_special_keys())
                elif mouse_btns[1]:
                    top_window.propagate_Rmouse_down(to_redraw=to_redraw,windows=open_windows,mouse_pos=usr_events.curr_mouse_pos, mouse_button_status=mouse_btns, pressed_special_keys=usr_events.pressed_special_keys())
            if wheeled:
                top_window.propagate_scroll(to_redraw=to_redraw,mouse_pos=usr_events.curr_mouse_pos,scroll=usr_events.scroll, pressed_special_keys=usr_events.pressed_special_keys()) #scroll is an integer, positive is up, negative is down
            # awaited events (can become awaiting_* after a propagate)
            if mouse_mvd:
                top_window.on_awaited_mouse_move(to_redraw=to_redraw,mouse_positions=(usr_events.prev_mouse_pos,usr_events.curr_mouse_pos), mouse_button_status=usr_events.pressed_mouse_buttons(), pressed_special_keys=usr_events.pressed_special_keys())
            if mouse_rel_lmb or mouse_rel_rmb:
                top_window.on_awaited_mouse_release(to_redraw=to_redraw,release_pos=usr_events.curr_mouse_pos, released_buttons=(mouse_rel_lmb, mouse_rel_rmb), pressed_special_keys=usr_events.pressed_special_keys())
            if key_press:
                top_window.on_awaited_key_press(to_redraw=to_redraw,windows=open_windows,pressed_keys=usr_events.pressed_keys, pressed_special_keys=usr_events.pressed_special_keys())

            if iter_num % 600 == 0:
                print("iter: {iter:n}\t\t, waiting_mv: {mv:n}, waiting_rel: {rel:n}, waiting_key: {key:n}, to_redraw: {redraw:n}".format(iter=iter_num,mv=len(top_window.awaiting_mouse_move),rel=len(top_window.awaiting_mouse_release),key=len(top_window.awaiting_key_press),redraw=len(to_redraw)))
                print("iter time:", "{:.3f}".format(iter_time, 3)+"ms", "      sleep:", str("{:.3f}".format(1000*self.frame_time))+"ms", "      work/sleep for gui thread:", str("{:.3f}".format(100*float(iter_time/(1000*self.frame_time))))+"%")

            # render the parts of the screen that were changed
            if to_redraw:
                self.redraw(to_redraw)
                to_redraw.clear()

            # if a window was added/removed by on_awaited_mouse_release
            if not len(open_windows) == nb_open_windows:
                nb_open_windows = len(open_windows)
                if nb_open_windows == 0:
                    usr_events.stop_signal = True
                else:
                    top_window = open_windows[-1]
                    top_window.draw(self.screen)
                    pygame.display.update(top_window.bounding_rect)
            if top_window.closing:
                if len(open_windows) == 1:
                    usr_events.stop_signal = True
                else:
                    top_window.delete()
                    del open_windows[-1]
                    top_window = open_windows[-1]
                    top_window.draw(self.screen)
                    pygame.display.update(top_window.bounding_rect)
                    nb_open_windows = len(open_windows)

            iter_num = (iter_num+1) % 9000
            iter_time = iter_time*0.7 + (time.time()-t)*0.3*1000
            time.sleep(self.frame_time)
        self.end()

    def redraw(self, to_redraw):
        rects = []
        for element in to_redraw:
            if not element.already_drawn:
                element.draw(self.screen)
                rects.append(element.bounding_rect)
        pygame.display.update(rects)


class User_events:
    def __init__(self):
        self.prev_mouse_pos = (0, 0)
        self.curr_mouse_pos = (0, 0)
        self.pressed_keys = np.array(['no', 'no', 'no'], dtype='<U2')
        self.ctrl_pressed = False
        self.shift_pressed = False
        self.enter_pressed = False
        self.lmb_pressed = False
        self.rmb_pressed = False
        self.scroll = 0  # positive = up, negative = down
        self.stop_signal = False

    def pressed_special_keys(self):
        return (self.ctrl_pressed, self.shift_pressed, self.enter_pressed)

    def pressed_mouse_buttons(self):
        return (self.lmb_pressed, self.rmb_pressed)

    def mouse_travel_dist(self): # euclidian distance, in pixels
        dx = self.prev_mouse_pos[0] - self.curr_mouse_pos[0]
        dy = self.prev_mouse_pos[1] - self.curr_mouse_pos[1]
        return int(sqrt(dx*dx + dy*dy))

    def update(self):
        self.prev_mouse_pos = self.curr_mouse_pos
        self.pressed_keys[0] = 'no'
        self.pressed_keys[1] = 'no'
        self.pressed_keys[2] = 'no'
        self.scroll = 0
        moved_mouse = False
        mouse_got_pressed = False
        mouse_released_rmb = False
        mouse_released_lmb = False
        key_got_pressed = False
        special_key_got_released = False
        wheel_got_wheeled = False
        key_idx = 0
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                moved_mouse = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.lmb_pressed = True
                    mouse_got_pressed = True
                elif event.button == 3:
                    self.rmb_pressed = True
                    mouse_got_pressed = True
                elif event.button == 4:
                    self.scroll += 1
                    wheel_got_wheeled = True
                elif event.button == 5:
                    self.scroll -= 1
                    wheel_got_wheeled = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.lmb_pressed = False
                    mouse_released_lmb = True
                elif event.button == 3:
                    self.rmb_pressed = False
                    mouse_released_rmb = True
            elif event.type == pygame.KEYDOWN:
                key_got_pressed = True
                # non-special key press
                if pygame.key.name(event.key) not in ['left shift', 'left ctrl', 'return']:
                    self.pressed_keys[key_idx] = pygame.key.name(event.key)
                    key_idx = (key_idx+1) % 3
                # special key press (shift, ctrl, enter)
                else:
                    if pygame.key.name(event.key) == 'left ctrl':
                        self.ctrl_pressed = True
                    elif pygame.key.name(event.key) == 'left shift':
                        self.shift_pressed = True
                    elif pygame.key.name(event.key) == 'return':
                        self.enter_pressed = True
            elif event.type == pygame.QUIT:
                self.stop_signal = True
                return False, False, False, False, False, False, False
        #check if special keys have been released
        if self.shift_pressed or self.ctrl_pressed or self.enter_pressed:
            pressd = pygame.key.get_pressed()
            if self.ctrl_pressed and not pressd[pygame.K_LCTRL]:
                self.ctrl_pressed = False
                special_key_got_released = True
            if self.shift_pressed and not pressd[pygame.K_LSHIFT]:
                self.shift_pressed = False
                special_key_got_released = True
            if self.enter_pressed and not pressd[pygame.K_RETURN]:
                self.enter_pressed = False
                special_key_got_released = True
        if moved_mouse:
            self.curr_mouse_pos = pygame.mouse.get_pos()
        return moved_mouse, mouse_got_pressed, mouse_released_lmb, mouse_released_rmb,\
                key_got_pressed, special_key_got_released, wheel_got_wheeled
