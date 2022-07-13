from engine.screen_managers.manager import Manager
from engine.gui.listener import Listener
from engine.gui.event_ids import *
from utils import random_colors
import threading
import numpy as np
from engine.screens.main_screen import Main_screen
from expl_model import Local_explanation_wrapper
from sklearn.neighbors import KDTree
from numba import njit
import Multi_BIOT
import BIOT
from scipy.linalg import norm

@njit
def thoughtfull_name(Xld, colours, orig_colours, errors, err_min, span):
    for pt in range(Xld.shape[0]):
        colours[pt] = orig_colours[pt] * (errors[pt]-err_min)/span

class Node():
    def __init__(self, idxs, is_leaf, split_axis, name):
        self.idxs = idxs
        self.is_leaf = is_leaf
        self.split_axis = split_axis
        self.name = name


def split_embedding(Xhd, Xld, threshold, min_support):
    N, M = Xhd.shape
    idxs = np.arange(N)
    root = Node(idxs, is_leaf=False, split_axis=0, name=0)

    nodes = [root]
    kept_explanations = []
    i = 0
    while(nodes):
        node = nodes.pop()
        tmp_explanation = Local_explanation_wrapper(node.idxs, Xld, Xhd, method = 'biot')
        expl_err = np.mean(tmp_explanation.compute_errors(Xhd[node.idxs], Xld[node.idxs]))
        print(expl_err, '(thresh=',threshold,')')
        if node.idxs.shape[0] <= min_support*2+1 or expl_err < threshold:
            node.is_leaf = True
            yield tmp_explanation
            # kept_explanations.append(tmp_explanation)
        else:
            node.is_leaf = False
            cut = np.median(Xld[node.idxs, node.split_axis], axis=0)

            idx1 = node.idxs[np.where(Xld[node.idxs, node.split_axis] < cut)[0]]
            idx2 = node.idxs[np.where(Xld[node.idxs, node.split_axis] >= cut)[0]]

            n1 = Node(idx1, is_leaf=False, split_axis = 1 - node.split_axis, name=i)
            n2 = Node(idx2, is_leaf=False, split_axis = 1 - node.split_axis, name=i)


            nodes.append(n1)
            nodes.append(n2)
            i+=1
    # return kept_explanations


class Main_manager(Manager):
    def __init__(self, config, main_window, theme, uid_generator):
        super(Main_manager, self).__init__("main", initial_state=True)
        # self.type_identifier = "main manger"
        self.active  = True
        self.deleted = False
        self.theme = theme
        self.method = 'biot'

        self.main_window   = main_window
        self.main_window.awaiting_key_press.append(self)
        self.uid_generator = uid_generator
        self.screen        = Main_screen(theme, main_window, self) # builds the screen
        self.lock = threading.Lock()
        self.main_view = self.main_window.containers[0]

        self.an_explanation_is_selected = False

        self.scatterplot_redraw_listener = Listener(REDRAW_THINGS, [self])
        self.scatterplot = self.main_view.leaves[0]
        self.ax1_explanation = self.main_view.containers[0].leaves[0]
        self.ax2_explanation = self.main_view.containers[0].leaves[1]
        self.ctrl_pressed = False
        self.Xhd, self.Xld, self.KDtree_ld, self.Y, self.Y_colours = None,None,None,None,None
        self.has_dataset = False
        self.K_select = 150
        self.draw_grid = np.zeros((150, 150), dtype = int)

        self.selected_feature = None


    def receive_dataset(self, Xhd, Xld, Y, Y_colours, Y_colours_expl, feature_names=None):
        self.Xhd = Xhd
        self.Xld = Xld
        self.KDtree_ld = KDTree(Xld, leaf_size=2)
        self.Y = Y
        self.has_dataset = True
        feature_colours1 = np.tile(np.array((161., 94., 249.)), Xhd.shape[1]).reshape((-1, 3))
        feature_colours2 = np.tile(np.array((249., 151., 94.)), Xhd.shape[1]).reshape((-1, 3))
        if feature_names is None:
            feature_names = ["variable "+str(i) for i in range(Xhd.shape[1])]
        self.scatterplot.set_points(Xld, Y_colours, Y_colours_expl, feature_names)
        self.ax1_explanation.receive_features(feature_names, feature_colours1)
        self.ax2_explanation.receive_features(feature_names, feature_colours2)
        self.feature_names = feature_names

    def new_explanation(self, neighbours_idx):
        explanation = Local_explanation_wrapper(neighbours_idx, self.Xld, self.Xhd, method=self.method)
        self.scatterplot.add_explanation(explanation)

    def explain_full_dataset_splitting(self, threshold, min_support):
        for explanation in split_embedding(self.Xhd, self.Xld, threshold, min_support):
            self.scatterplot.add_explanation(explanation)
            self.redraw_things()

    def explain_full_dataset_Kmeans(self, threshold, min_support, K):
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=K, random_state=0).fit(self.Xld)
        centers = model.cluster_centers_
        labels = model.labels_
        for i in range(centers.shape[0]):
            print(np.round(float(i/centers.shape[0]),2))
            idxs = np.where(labels == i)[0]
            Nidx = idxs.shape[0]
            if Nidx > min_support:
                explanation = Local_explanation_wrapper(idxs, self.Xld, self.Xhd, method = 'biot')
                self.scatterplot.add_explanation(explanation)

    def merge_explanations(self, threshold):

        for pass_i in range(3):
            explanations = self.scatterplot.local_explanations
            N_expla = len(explanations)
            marked = np.zeros((N_expla,), dtype=np.bool)

            to_remove = []
            to_add = []

            for i, expl_wrapper in enumerate(explanations):
                if marked[i]:
                    continue
                expl = expl_wrapper.model
                center1 = expl.center_LD
                R_1  = expl.R
                W_1  = expl.W
                w0_1 = expl.w0
                sample1 = expl_wrapper.sample_idx

                d = np.zeros(N_expla)
                for u, expl_wrapper2 in enumerate(explanations):
                    d[u] = np.sum((expl_wrapper2.model.center_LD - center1)**2)
                neighbours = np.argsort(d)[1:]

                for r, neigh in enumerate(neighbours):
                    if marked[neigh]:
                        continue
                    expl2 = explanations[neigh].model

                    center2 = expl2.center_LD
                    R_2  = expl2.R
                    W_2  = expl2.W
                    w0_2 = expl2.w0
                    sample2 = explanations[neigh].sample_idx

                    full_sample = np.hstack((sample1, sample2))


                    mu_R = (R_1 + R_2) / 2
                    mu_R /= norm(mu_R, axis=1)
                    mu_w0 = (w0_1 + w0_2) / 2
                    mu_W  = (W_1 + W_2) / 2
                    new_center = np.mean(self.Xld[full_sample], axis=0)

                    new_explanation = Local_explanation_wrapper(full_sample, self.Xld, self.Xhd, method = 'biot', fit=False)
                    new_explanation.model.center_LD = new_center
                    new_explanation.model.center_HD = np.mean(self.Xhd[full_sample], axis=0)
                    new_explanation.model.axis2d = mu_R
                    new_explanation.model.features_coeffs_ax1 = mu_W[:,0]
                    new_explanation.model.features_coeffs_ax2 = mu_W[:,1]
                    new_explanation.model.W = mu_W
                    new_explanation.model.w0 = mu_w0
                    new_explanation.model.R = mu_R
                    new_explanation.sample_idx = full_sample
                    new_explanation.axis2d = new_explanation.model.axis2d


                    centrd = self.Xhd[full_sample] - new_explanation.model.center_HD
                    Yhat_new = new_center + ((np.tile(mu_w0, (centrd.shape[0], 1)) + (centrd @ mu_W)) @ mu_R.T)
                    err = np.mean(np.sqrt(np.mean((Yhat_new - self.Xld[full_sample])**2, axis=1)))

                    print('err, threshold ', err, threshold)
                    if err < threshold:

                        marked[i] = True
                        marked[neigh] = True
                        to_remove.append(i)
                        to_remove.append(neigh)
                        to_add.append(new_explanation)
                        break
            import time
            to_remove = np.sort(np.array(to_remove))
            for i in range(to_remove.shape[0]):
                print(to_remove[-1-i])
                self.scatterplot.delete_explanation(to_remove[-1-i])

            for expl in to_add:
                self.scatterplot.add_explanation(expl)

            self.redraw_things()
            time.sleep(1.5)
        self.redraw_things()


    def last_biot(self):
        explanations = self.scatterplot.local_explanations
        for i, expl_wrapper in enumerate(explanations):
            expl_wrapper.model.fit(self.Xld[expl_wrapper.sample_idx], self.Xhd[expl_wrapper.sample_idx])
            expl_wrapper.axis2d = expl_wrapper.model.axis2d
        self.redraw_things()
        print("done")


    def explain_full_dataset(self, threshold=10., min_support=10):
        self.explain_full_dataset_splitting(threshold=threshold, min_support=min_support)
        # self.explain_full_dataset_Kmeans(threshold=threshold, min_support=min_support, K=10)

        print('todo: réparer merge_explanations(): le "mean model" marche bien, mais il y a un bug à trouver.')
        # self.merge_explanations(threshold=threshold)

        self.last_biot()

        # print("MULTI BIOT")
        # import pandas as pd
        # max_lam = BIOT.calc_max_lam(self.Xhd, self.Xld)
        # n_lam = 10
        # lam_values = max_lam*(10**np.linspace(-1, 0, num=n_lam, endpoint=True, retstep=False, dtype=None))
        # lam_list = lam_values.tolist()
        # initial_clusters = np.zeros(self.Xhd.shape[0], dtype=int)
        # for i, expl in enumerate(self.scatterplot.local_explanations):
        #     initial_clusters[expl.sample_idx] = i
        # Yhat, W_list, w0_list, R_list, clusters = Multi_BIOT.CV_Multi_BIOT(
        #     X_train = pd.DataFrame(self.Xhd), X_test = pd.DataFrame(self.Xhd), Y_train = pd.DataFrame(self.Xld), lam_list = lam_list,
        #     K_list = None, clusters = initial_clusters, rotation = True)
        #
        # print(Yhat)
        # print(W_list[0].shape)
        # print(w0_list)
        # print(R_list)
        # print(clusters)


    def select_explanation(self, explanation_idx):
        if explanation_idx == -1:
            return
        self.an_explanation_is_selected = True
        self.scatterplot.selected_explanation = explanation_idx
        explanation = self.scatterplot.local_explanations[explanation_idx]
        errors = explanation.compute_errors(self.Xhd, self.Xld)
        goodness = -errors
        score_min, score_max = np.min(goodness), np.max(goodness)
        span = score_max - score_min

        colours      = self.scatterplot.Y_colours
        orig_colours = self.scatterplot.orig_Y_colours
        thoughtfull_name(self.Xld, colours, orig_colours, goodness, score_min, span)

        self.ax1_explanation.receive_explanation(explanation, 0)
        self.ax2_explanation.receive_explanation(explanation, 1)


    def click(self, pos, left_click):
        if not self.has_dataset:
            return

        if left_click:
            closest = self.scatterplot.closest_explanation(self.scatterplot.px_pos_to_LD(pos))
            self.select_explanation(closest)
        else:
            pos_in_LD = self.scatterplot.px_pos_to_LD(pos)
            if not self.ctrl_pressed:
                pass
            else:
                self.scatterplot.delete_explanation(pos_in_LD)

    def wake_up(self, prev_manager):
        super(Main_manager, self).wake_up(prev_manager)

    def select_feature(self, feature_name, left=True):
        self.ax1_explanation.select_feature(feature_name, left_click=True)
        self.ax2_explanation.select_feature(feature_name, left_click=True)
        self.scatterplot.selected_feature = feature_name

    def unselect_feature(self, left=True):
        self.ax1_explanation.unselect_features(left_click=True)
        self.ax2_explanation.unselect_features(left_click=True)
        self.scatterplot.selected_feature = None

    def get_notified(self, event_class, id, value, to_redraw = []):
        with self.lock:
            if event_class == REDRAW_THINGS:
                self.redraw_things()
            elif event_class == CTRL_KEY_CHANGE:
                is_pressed, pos = value
                self.ctrl_pressed = is_pressed
                self.an_explanation_is_selected = False
                self.scatterplot.selected_explanation = -1
                self.scatterplot.Y_colours = self.scatterplot.orig_Y_colours.copy()
                self.redraw_things()

            elif event_class == SCATTERPLOT_LEFT_CLICK:
                self.unselect_feature()
                self.click(value[0], left_click=True)
                self.redraw_things()
            elif event_class == SCATTERPLOT_RIGHT_CLICK:
                self.unselect_feature()
                self.click(value[0], left_click=False)
                self.redraw_things()
            elif event_class == SCATTERPLOT_HOVER:
                return
            elif event_class == SCATTERPLOT_DRAWING_DONE:
                neighbours = value
                self.new_explanation(neighbours)
            elif event_class == AXIS_LEFT_CLICK:
                var_name = value[0]
                if var_name is None:
                    self.unselect_feature()
                    return
                print(var_name)
                self.select_feature(var_name, left = True)
                self.redraw_things()
            elif event_class == AXIS_RIGHT_CLICK:
                var_name = value[0]
                if var_name is None:
                    self.unselect_feature()
                    return
                print(var_name)
                self.select_feature(var_name, left = False)
                self.redraw_things()
            else:
                print("unrecognised event received by main_manager: ", event_class, "  (see correspondance in event_ids.py)")



    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        if pressed_keys[0] == 'o':
            print("pressed o")
        return False

    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        1/0
        return True


    def redraw_things(self):
        self.ask_redraw(self.main_view)
