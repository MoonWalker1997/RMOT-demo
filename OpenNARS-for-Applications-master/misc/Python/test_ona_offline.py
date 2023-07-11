"""
The file is named with "offline", since it works offline to get the tracking, it is not about whether "the method is
offline" or not. It is still classified as "online tracking", since only previous frames are used to generate the
tracking.

It is called "offline" since it needs the result from multiple trackers in advance. It is okay to use such a method on
MOT17 challenge.
"""
import os
import sys
import time

import cv2
import lap
import numpy as np
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))

import NAR as reasoner
from yolox.tracker import matching
from yolox.tracker.kalman_filter import KalmanFilter
from yolox.utils.visualize import plot_tracking

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(2, suppress=True)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..", "exps")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..", "exps/example/mot")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..", "OpenNARS-for-Applications-master")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "TrackEval-master")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Demo/pretrained")))

min_correspondence_thresh = 5  # > 0, integer
max_lost_track_tolerance = 15  # > 0, integer
appearance_model_keeping_rate = 0.1  # [0, 1]
non_detected_speed_penalty = 0.01  # [0, 1]
iou_similarity_thresh_DRC = 0.5  # [0, 1]
appearance_similarity_thresh = 0.7  # [0, 1]
iou_similarity_thresh_GLC = 0.2  # [0, 1]
iou_similarity_thresh_GT = 0.9  # [0, 1]
box_score_thresh = 0.85  # [0, 1]


def tlwh_to_tlbr(tlwh):
    ret = np.array(tlwh.copy())
    ret[2:] += ret[:2]
    return ret


def generate_color_hist(img_patch):
    # BRG color hist
    if img_patch.shape[0] * img_patch.shape[1] == 0:
        return None
    img = cv2.resize(img_patch, (64, 64))
    blue_track = cv2.calcHist([img], [0], None, [25], [0, 256])
    blue_track = blue_track / blue_track.sum()
    red_track = cv2.calcHist([img], [1], None, [25], [0, 256])
    red_track = red_track / red_track.sum()
    green_track = cv2.calcHist([img], [2], None, [25], [0, 256])
    green_track = green_track / green_track.sum()

    return [blue_track, red_track, green_track]


def img_patch(tlbr, img):
    return img[
           min(img.shape[0], max(0, int(tlbr[1]))):
           min(img.shape[0], max(0, int(tlbr[3]))),
           min(img.shape[1], max(0, int(tlbr[0]))):
           min(img.shape[1], max(0, int(tlbr[2]))), :]


def cleanness(tlbr_tracking, tlbr_trackings, inter_iou_similarity, I, img):
    area = abs(tlbr_tracking[0] - tlbr_tracking[2]) * abs(tlbr_tracking[1] - tlbr_tracking[3]) + 1

    mask = np.zeros([img.shape[0], img.shape[1]])

    mask[
    min(img.shape[0], max(0, int(tlbr_tracking[1]))): min(img.shape[0], max(0, int(tlbr_tracking[3]))),
    min(img.shape[1], max(0, int(tlbr_tracking[0]))): min(img.shape[1], max(0, int(tlbr_tracking[2])))] \
        = 1

    for j in range(inter_iou_similarity.shape[1]):
        if inter_iou_similarity[I, j] > 0 and j != I:
            mask[
            min(img.shape[0], max(0, int(tlbr_trackings[j][1]))): min(img.shape[0], max(0, int(tlbr_trackings[j][3]))),
            min(img.shape[1], max(0, int(tlbr_trackings[j][0]))): min(img.shape[1], max(0, int(tlbr_trackings[j][2])))] \
                = 0

    return mask.sum() / area


def compare_color_hist(hist1, hist2):
    # multiply the color-hist similarities of three channels
    return cv2.compareHist(hist1[0], hist2[0], 0) * cv2.compareHist(hist1[1], hist2[1], 0) * cv2.compareHist(
        hist1[2], hist2[2], 0)


def linear_assignment(cost_matrix, thresh = 0.8):
    # the return are just indices
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


class outside_track:
    """
    An outside (outside the original tracker) copy. It can predict automatically (with a KF).
    But its updating is controlled by hard-coded rules or reasoners outside.
    """

    def __init__(self, ID, label_ID, tlwh, score, appearances = None, appearance_score = 0.0):
        # data
        self.ID = ID
        self.label_ID = label_ID

        self.kalman_filter = KalmanFilter()
        self.mean, self.cov = self.kalman_filter.initiate(self.tlwh_to_xyah(tlwh))

        self.score = score  # the quality of a box (generated by NN)

        self.appearances = appearances  # appearance model, color-hist is used here, but can be extended
        self.appearance_score = appearance_score

        self.max_life = 10
        self.life = self.max_life  # how many frames can a track stand with no updates

        self.initialized = False
        self.updated = True  # whether it is just updated
        self.updated_appearance = True  # whether the appearance is just updated

        self.to_update = [[tlwh, score, appearances, appearance_score]]

    def retire(self, t = 1):
        self.life -= t

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.cov = self.kalman_filter.predict(mean_state, self.cov)

    def initialize(self, tlwh, score, appearances = None, appearance_score = 0):
        #  the KF is not updated, but re-initialized

        # activate
        self.life = self.max_life
        self.initialized = True
        self.updated = True
        self.updated_appearance = True  # whether the appearance is just updated

        # KF initialization
        self.kalman_filter = KalmanFilter()
        self.mean, self.cov = self.kalman_filter.initiate(self.tlwh_to_xyah(tlwh))

        # update score
        self.score = score  # the quality of a box (generated by NN)

        # update appearance
        self.appearances = appearances  # appearance model, color-hist is used here, but can be extended
        self.appearance_score = appearance_score

        self.to_update = []

    def update(self, tlwh, score, appearances = None, appearance_score = 0):
        # re-active
        self.life = self.max_life
        self.updated = True

        # update the KF
        mean, cov = self.kalman_filter.update(self.mean, self.cov, self.tlwh_to_xyah(tlwh))

        # update score
        self.score = score

        # update appearance
        if appearances is not None:
            self.updated_appearance = True
            self.appearances = appearances
            self.appearance_score = appearance_score
            # EMA, observation centric, higher weight for the new box
            mean = self.mean * 0.1 + mean * 0.9
        self.mean = mean
        self.cov = cov

        self.to_update = []

    @property
    def tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class external_tracker_manager:

    def __init__(self, path, video_length):
        self.frame = 1
        self.index = 0
        self.length = video_length
        with open(path) as file:
            self.data = file.readlines()
        self.correspondence = {}  # format: {tracker ID: [outside_track ID, count]}

    def next_frame(self):
        if self.frame > self.length:
            return None
        ret = []
        for i in range(self.index, len(self.data)):
            if int(self.data[i].split(",")[0]) == self.frame:
                ret.append([float(each) for each in self.data[i].strip("\n").split(",")])
            else:
                self.index = i
                self.frame += 1
                break
        # format: [frame, object ID, top left x, top left y, width, height, score, -, -, -]
        # tlwh in short
        return ret

    def update_correspondence(self, external_tracker_ID, outside_track_ID):
        if external_tracker_ID in self.correspondence:
            if self.correspondence[external_tracker_ID][0] == outside_track_ID:
                self.correspondence[1] += 1
        else:
            self.correspondence.update({external_tracker_ID: [outside_track_ID, 1]})


class ground_truth_manager:

    def __init__(self, path, video_length):
        self.frame = 1
        self.index = 0
        self.length = video_length
        with open(path) as file:
            tmp = file.readlines()
        self.data = []
        for i in range(len(tmp)):
            self.data.append([float(each) for each in tmp[i].strip("\n").split(",")])
        self.data.sort(key=lambda x: x[0])
        self.correspondence = {}  # format: {tracker ID: [outside_track ID, count]}

    def next_frame(self):
        if self.frame > self.length:
            return None
        ret = []
        for i in range(self.index, len(self.data)):
            if self.data[i][0] == self.frame:
                ret.append(self.data[i])
            else:
                self.index = i
                self.frame += 1
                break
        # format: [frame, object ID, top left x, top left y, width, height, score, -, -, -]
        # tlwh in short
        return ret


class video_manager:

    def __init__(self, path, video_length):
        self.frame_id = 1
        self.path = path + "/"
        self.length = video_length

    def next_frame(self):
        if self.frame_id > self.length:
            return None
        path = self.path + str(self.frame_id).zfill(6) + ".jpg"
        self.frame_id += 1
        return cv2.imread(path)


video_length = 1050

# cv2.videoCap might be better
imgs_path = os.path.abspath(os.path.join(os.getcwd(), "../../../", "data/MOT17/MOT17/train/MOT17-04-DPM/img1/"))

external_trackers = {
    "yoloxl": os.path.abspath(os.path.join(os.getcwd(), "../../../", "data/trackers/MOT17-04-DPM-1.txt")),
    "yoloxm": os.path.abspath(os.path.join(os.getcwd(), "../../../", "data/trackers/MOT17-04-DPM-2.txt"))}

video_manager = video_manager(imgs_path, video_length)
for each in external_trackers:
    external_trackers[each] = external_tracker_manager(external_trackers[each], video_length)
ground_truth = ground_truth_manager(
    os.path.abspath(os.path.join(os.getcwd(), "../../../", "data/MOT17/MOT17/train/MOT17-04-SDP/gt/gt.txt")),
    video_length)

frame_id = 1
time_a = time.time()
outside_track_ID = 1
outside_tracks = {}
results = []  # for the txt file
vid_writer = cv2.VideoWriter(os.path.abspath(os.path.join(os.getcwd(), "YOLOX_outputs/hai/res.mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             30,
                             (int(1920), int(1080)))
reasoner.Reset()

show_reasoning = False
fps = 0

for _ in range(video_length):

    if frame_id == 201:
        print(1)

    # processing each frame

    if frame_id % 1 == 0:
        time_b = time.time()
        fps = 1. / max(1e-5, time_b - time_a)
        logger.info("Processing frame {} ({:.2f} fps)".format(frame_id, fps))
        time_a = time_b

    # get the frame
    img = video_manager.next_frame()

    # pop outdated outside tracks
    tmp = []
    for each in outside_tracks:
        outside_tracks[each].retire()
        if outside_tracks[each].life < -max_lost_track_tolerance:
            tmp.append(each)
    for each in tmp:
        outside_tracks.pop(each)

    # ground truth
    gt = ground_truth.next_frame()
    tlbr_gt = np.array([tlwh_to_tlbr(each[2: 6]) for each in gt])

    for each_external_tracker in external_trackers:
        tracking = external_trackers[each_external_tracker].next_frame()

        print(len(tracking) * len(gt))

        tlbr_tracking = np.array([tlwh_to_tlbr(each[2: 6]) for each in tracking])
        tlbr_outside_tracks = np.array([outside_tracks[each].tlbr for each in outside_tracks])

        # comprehensive similarity used for bipartite matching
        comprehensive_similarity = np.zeros((img.shape[0], img.shape[1]))

        # iou similarities between external tracker results and outside tracks, the higher the similar
        iou_similarity = 1 - matching.iou_distance(tlbr_tracking,
                                                   tlbr_outside_tracks,
                                                   img.shape)

        # iou similarities between external trackers, used to find these "pure" (without overlapping) boxes
        inter_iou_similarity = 1 - matching.iou_distance(tlbr_tracking,
                                                         tlbr_tracking,
                                                         img.shape)

        # iou similarity between external trackers and the ground truth
        gt_iou_similarity = 1 - matching.iou_distance(tlbr_tracking,
                                                      tlbr_gt,
                                                      img.shape)

        # storage for save some complexity
        cls = [None for _ in range(len(tracking))]  # cleanness score
        hists_target = [None for _ in range(len(tracking))]

        I = []

        # try to evaluate each pair of outside tracks and tracking
        for i, each in enumerate(tracking):

            if each[6] < box_score_thresh:  # does not track low quality box
                continue

            # get the cleanness
            if cls[i] is not None:
                cl = cls[i]
            else:
                cl = cleanness(tlbr_tracking[i], tlbr_tracking, inter_iou_similarity, i, img)
                cls[i] = cl

            # get the appearance (color-hist currently)
            if hists_target[i] is not None:
                hist_target = hists_target[i]
            else:
                hist_target = generate_color_hist(img_patch(tlbr_tracking[i], img))
                hists_target[i] = hist_target

            for j, each_id in enumerate(outside_tracks):

                if hist_target is None:
                    # this is to avoid some boxes outside the scope, there is still one bbox, but its area is 0
                    # considering the resolution rate of the video
                    continue

                # the first question: whether the target looks like the track
                hist_similarity = compare_color_hist(hist_target, outside_tracks[each_id].appearances)
                reasoner.AddInput("similar. :|: %" + str(min(1, max(0, hist_similarity))) + ";0.9%", show_reasoning)

                # the second question: whether the target is close to the track
                reasoner.AddInput("*concurrent", show_reasoning)
                reasoner.AddInput("close. :|: %" + str(min(1, max(0, iou_similarity[i, j]))) + ";0.9%", show_reasoning)

                # the third question: whether the tracking is trustful
                # reasoner.AddInput("*concurrent", show_reasoning)
                # if each[1] not in external_trackers[each_external_tracker].correspondence:
                #     c = 0
                # else:
                #     c = external_trackers[each_external_tracker].correspondence[each[1]][0]
                # reasoner.AddInput(
                #     "<(" + each_external_tracker + str(each[1]) + "," + str(each_id) + ") --> trust>. %" + str(
                #         c / (c + 1)) + ";0.9%", show_reasoning)

                # check the ground truth
                # ------------------------------------------------------------------------------------------------------
                gt_idx = np.where([tmp[1] for tmp in gt] == outside_tracks[each_id].label_ID)[0]
                if gt_idx.size != 0 and gt_iou_similarity[i, gt_idx[0]] > iou_similarity_thresh_GT:
                    # this external tracker and this track should be matched
                    reasoner.AddInput("match. :|: %1.0;0.9%", show_reasoning)
                else:
                    # this external tracker and this track should not be matched
                    reasoner.AddInput("match. :|: %0.0;0.9%", show_reasoning)
                # ------------------------------------------------------------------------------------------------------

                # use the reasoner to "learn from the ground truth"
                reasoner.AddInput("1", show_reasoning)

                # ask whether they should be matched (even this has been mentioned in the learning process)
                r = reasoner.AddInput("match?", show_reasoning)
                for each_answer in r["answers"]:
                    if each_answer["term"] == "match":
                        comprehensive_similarity[i, j] = float(each_answer["truth"]["confidence"]) * (
                                float(each_answer["truth"]["frequency"]) - 0.5) + 0.5

        matches, unmatched_a, unmatched_b = linear_assignment(1 - comprehensive_similarity, thresh=0.6)

        for each_match in matches:
            I.append(each_match[0])
            ot_id = list(outside_tracks.keys())[each_match[1]]
            outside_tracks[ot_id].to_update.append([each[2: 6],
                                                    each[6],
                                                    hist_target,
                                                    max(0.1, cl)])

        gt_matches, _, _ = linear_assignment(1 - gt_iou_similarity, thresh=0.9)

        for i, each in enumerate(tracking):

            if each[6] < box_score_thresh or i in I:  # does not track low quality and considered box
                continue

            # get the cleanness
            if cls[i] is not None:
                cl = cls[i]
            else:
                cl = cleanness(tlbr_tracking[i], tlbr_tracking, inter_iou_similarity, i, img)
                cls[i] = cl

            # get the appearance (color-hist currently)
            if hists_target[i] is not None:
                hist_target = hists_target[i]
            else:
                hist_target = generate_color_hist(img_patch(tlbr_tracking[i], img))
                hists_target[i] = hist_target

            if i in [tmp[0] for tmp in gt_matches]:
                label_ID = gt[[tmp[0] for tmp in gt_matches].index(i)][1]
                outside_tracks.update({outside_track_ID:
                                           outside_track(ID=outside_track_ID,
                                                         label_ID=label_ID,
                                                         tlwh=each[2:6],
                                                         score=each[6],
                                                         appearances=hist_target,
                                                         appearance_score=max(0.1, cl))})
            external_trackers[each_external_tracker].correspondence.update({each[1]: [outside_track_ID, 1]})
            outside_track_ID += 1

    # pick the best to update/initialize
    for each in outside_tracks:
        if len(outside_tracks[each].to_update) != 0:
            outside_tracks[each].to_update.sort(key=lambda x: x[1] * x[3])
            d = outside_tracks[each].to_update[-1]
            if not outside_tracks[each].initialized:
                outside_tracks[each].initialize(*d)
            else:
                if outside_tracks[each].life >= 0:
                    if d[-1] > outside_tracks[each].appearance_score or d[-1] > 0.7:
                        outside_tracks[each].update(*d)
                    else:
                        outside_tracks[each].update(tlwh=d[0], score=d[1])
                else:
                    outside_tracks[each].initialize(*d)

    # ==================================================================================================================

    # txt writer
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for each in outside_tracks:

        if outside_tracks[each].life >= 0:

            tlwh = outside_tracks[each].tlwh
            tid = each
            # if not outside_tracks[each].touched_appearance:
            #     tid = -tid
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 10 and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(outside_tracks[each].score)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},"
                    f"{outside_tracks[each].score:.2f},-1,-1,-1\n"
                )
    online_im = plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=fps)

    for each in outside_tracks:

        if outside_tracks[each].life >= 0:
            # make the speed of the aspect ratio 0 when not updated (trick often used in many MOT codes)
            if not outside_tracks[each].updated:
                outside_tracks[each].mean[7] = 0
            outside_tracks[each].predict()
            outside_tracks[each].updated = False
            outside_tracks[each].updated_appearance = False

    vid_writer.write(online_im)
    frame_id += 1

res_file = os.path.abspath(os.path.join(os.getcwd(), "YOLOX_outputs/hai/res.txt"))

with open(res_file, "w") as f:
    f.writelines(results)
logger.info(f"save results to {res_file}")
