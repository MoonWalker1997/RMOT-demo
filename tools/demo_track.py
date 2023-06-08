import argparse
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
from cython_bbox import bbox_overlaps as bbox_ious
from loguru import logger
from pynars.NARS.Control.Reasoner import Reasoner
from pynars.utils.Print import PrintType
from sty import bg, fg
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker import matching
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

IDE = True
video_name = "street"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/" + video_name + ".mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file = None,
            decoder = None,
            device = torch.device("cpu"),
            fp16 = False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # set to a pre-defined format (h, w, c), fixed size
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16
        img_info["actual_img"] = img.squeeze()

        # all above are pre-processing
        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)  # RPN
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            # after this post-processing, only high quality bboxes can remain
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def track_name(track):
    return "".join(str(track).split("(")[0][:-1].split("_"))


def target_property_to_Narsese(online_target):
    # need more information
    return []


def iou_distance_to_Narsese(online_target: STrack, box_id: int, outside_track: STrack):
    iou = bbox_ious(
        np.ascontiguousarray([online_target.tlbr], dtype=float),
        np.ascontiguousarray([outside_track.tlbr], dtype=float)
    ).squeeze()
    return "<(*, box" + str(box_id) + ", " + track_name(online_target) + ") --> close>. %" + str(iou.item()) + "; 0.9%"


def histogram_similarity(online_target, outside_track, img):
    img_patch = img[min(max(0, int(online_target.tlbr[1])), 720): min(max(0, int(online_target.tlbr[3])), 720),
                min(max(0, int(online_target.tlbr[0])), 1280): min(max(0, int(online_target.tlbr[2])), 1280), :]

    blue_target = cv2.calcHist([img_patch], [0], None, [256], [0, 256])
    red_target = cv2.calcHist([img_patch], [1], None, [256], [0, 256])
    green_target = cv2.calcHist([img_patch], [2], None, [256], [0, 256])

    blue_track = cv2.calcHist([outside_track.img_patch], [0], None, [256], [0, 256])
    red_track = cv2.calcHist([outside_track.img_patch], [1], None, [256], [0, 256])
    green_track = cv2.calcHist([outside_track.img_patch], [2], None, [256], [0, 256])

    return cv2.compareHist(blue_target, blue_track, 0) * cv2.compareHist(red_target, red_track, 0) * cv2.compareHist(
        green_target, green_track, 0)


def histogram_similarities(online_targets, outside_tracks, img):
    ret = np.zeros((len(online_targets), len(outside_tracks)))
    for i, each_online_target in enumerate(online_targets):
        for j, each_outside_track in enumerate(outside_tracks):
            ret[i, j] = histogram_similarity(each_online_target, outside_tracks[each_outside_track], img)
    return ret


def appearance_similarity_to_Narsese(online_target, box_id, outside_track, img):
    return "<(*, box" + str(box_id) + ", " + track_name(online_target) + ") --> similar>. % " + str(
        histogram_similarity(online_target, outside_track, img)) + "; 0.9%"


def comparison_to_Narsese(online_target, box_id, outside_track, img):
    ret = []
    ret.append(iou_distance_to_Narsese(online_target, box_id, outside_track))
    ret.append(appearance_similarity_to_Narsese(online_target, box_id, outside_track, img))
    return ret


def print_out(type, content, p: float = None, d: float = None, q: float = None, comment_title: str = None,
              end: str = None):
    # show_budget = True
    # if isinstance(p, float) and isinstance(d, float) and isinstance(q, float):
    #     if p<0 or p>1 or q<0 or q>1 or d<0 or d>1:
    #         show_budget = False
    # else:
    #     show_budget = False

    if isinstance(p, float) and p >= 0 and p <= 1:
        bg1 = bg(min(255, int(255 * p / 2 + 10)), 10, 10)
        p = f'{round(p, 2):.2f}'
    else:
        p = '    '
        bg1 = ''
    if isinstance(d, float) and d >= 0 and d <= 1:
        bg2 = bg(10, min(255, int(255 * d / 2 + 10)), 10)
        d = f'{round(d, 2):.2f}'
    else:
        d = '    '
        bg2 = ''
    if isinstance(q, float) and q >= 0 and q <= 1:
        bg3 = bg(10, 10, min(255, int(255 * q / 2 + 10)))
        q = f'{round(q, 2):.2f}'
    else:
        q = '    '
        bg3 = ''

    # print(F'{bg(int(256*p),0,0)} {p} {bg(0,int(256*q),0)} {q} {bg(0,0,int(256*d))} {d} {bg.rs}{type.value} {str(content)}')

    if type is PrintType.COMMENT and comment_title is not None:
        print(f'{fg.da_grey}{comment_title}: {str(content)}{fg.rs}', end=end)
    elif type is PrintType.INFO:
        print(f'{bg1} {p} {bg.rs}{bg2} {d} {bg.rs}{bg3} {q} {bg.rs} {type.value} {fg.grey}{str(content)}{fg.rs}',
              end=end, )
    else:
        print(f'{bg1} {p} {bg.rs}{bg2} {d} {bg.rs}{bg3} {q} {bg.rs} {type.value} {str(content)}', end=end, )


def imageflow_demo(predictor, vis_folder, current_time, args):
    # No reID included.
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    if IDE:
        cap = cv2.VideoCapture(
            "C:\\Users\\TORY\\OneDrive - Temple University\\AGI research\\ByteTrack-main\\videos\\" + video_name +
            ".mp4")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    """
    Design:
    
    1. I will ONLY use the result from the tracker, and I will NEVER change the pipeline how the result is generated.
    2. In each frame, a tracking result is expected from an existed tracker.
    3. The result is used to create several Kalman filters OUTSIDE of the tracker, say I am going to have "another
    tracker" outside of the existed tracker.
    4. Since most of the existed tracker will follow a pattern that "detection and tracking", therefore, if there are no
    detections (no bounding boxes), it is impossible for it to have a trajectory. As experimented, this is a common 
    issue of detectors.
    5. We can pre-define, or learn gradually the positions of "entrance" and "exit" in the view, and so when a tracking
    generated by the tracker is broken, we can continue it under assumptions in the outside tracker.
    """

    # main process starts here
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    box_id = 0
    results = []

    # reasoner included
    reasoner = Reasoner(50000, 50000)

    # a-priori knowledge, variable supported
    reasoner.input_narsese("<<(*, $x, $y) --> similar> ==> <(*, $x, $y) --> belong>>.")
    reasoner.input_narsese("<<(*, $x, $y) --> close> ==> <(*, $x, $y) --> belong>>.")
    # ...

    # a dictionary of tracks {track name: track object}, managed outside the "byte tracker"
    outside_tracks = {}

    while True:  # for each frame

        if frame_id % 20 == 0:  # data logging
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        if frame_id // 20 == 5:
            break

        ret_val, frame = cap.read()  # imread
        if ret_val:
            # only high quality (class quality) bboxes remain in outputs
            # the filtering is based on a pre-defined threshold (0.001)

            outputs, img_info = predictor.inference(frame, timer)  # predictor inference, tracks and ids
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

                # the "online targets" are all "active (just updated)" tracks

                # update outside_tracks
                for each in outside_tracks:
                    STrack.multi_predict([outside_tracks[each]])

                dists = matching.iou_distance(online_targets, [outside_tracks[each] for each in outside_tracks])
                dists2 = histogram_similarities(online_targets, outside_tracks, img_info["raw_img"])
                print("")

                # compare these online targets with existed outside tracks
                # x_index = {}
                # similarity = np.zeros((len(online_targets), len(outside_tracks)))
                # for i, each_online_target in enumerate(online_targets):
                #     box_id += 1
                #     x_index.update({box_id: i})
                #     judgments = []
                #     # translate the property generated by some DL models
                #     # see by segmentation, skeleton detection, and object detection, we know a person has a red hat
                #     # the information will be translated into Narsese
                #     judgments.extend(target_property_to_Narsese(each_online_target))
                #     for each_outside_track in outside_tracks:
                #         judgments.extend(
                #             comparison_to_Narsese(each_online_target, box_id, outside_tracks[each_outside_track],
                #                                   img_info["raw_img"]))
                #         question = "<(*, box" + str(box_id) + ", " + str(
                #             track_name(each_online_target)) + ") --> belong>?"
                #         for each_Narsese in judgments + [question]:
                #             # print(each_Narsese)
                #             success, task, _ = reasoner.input_narsese(each_Narsese, go_cycle=False)
                #         tasks_all_cycles = []
                #         for _ in range(1):
                #             tasks_all = reasoner.cycle()
                #             tasks_all_cycles.append(deepcopy(tasks_all))
                #         # output_information, TODO, not just for outputting
                #         for tasks in tasks_all_cycles:
                #             tasks_derived, judgement_revised, goal_revised, answers_question, answers_quest, (
                #                 task_operation_return, task_executed) = tasks
                #             if answers_question is not None:  # if a question is answered
                #                 for answer in answers_question:
                #                     tmp = [str(each) for each in answer.term.subject.terms]
                #                     box_id, outside_track_id = int(tmp[0][3:]), int(tmp[1][2:]) - 1
                #                     similarity[x_index[box_id], outside_track_id] = answer.truth.f
                #                     # print_out(PrintType.ANSWER, answer.sentence.repr(), *answer.budget)

                # find the "continued" tracks, say the track of one object is "continued"
                continued_outside_tracks = []
                for each in online_targets:
                    if track_name(each) in outside_tracks:
                        continued_outside_tracks.append(track_name(each))
                        outside_tracks[track_name(each)] = each
                        outside_tracks[track_name(each)].img_patch = img_info["raw_img"][
                                                                     min(max(0, int(each.tlbr[1])), 720): min(max(0, int(each.tlbr[3])), 720),
                                                                     min(max(0, int(each.tlbr[0])), 1280): min(max(0, int(each.tlbr[2])), 1280), :]
                        if outside_tracks[track_name(each)].img_patch.shape[1] == 0:
                            print(1)

                # so there are some non-continued tracks
                non_continued_outside_tracks = []
                for each in outside_tracks:
                    if each not in continued_outside_tracks:
                        non_continued_outside_tracks.append(outside_tracks[each])

                # find the "new" tracks, say "a new object" is found, and it is given a track
                new_outside_tracks = []
                for each in online_targets:
                    if track_name(each) not in continued_outside_tracks:
                        new_outside_tracks.append(each)

                # for those outside tracks, check them with all stored tracks, if they are "close", then it is
                # not "a new object"
                dists = matching.iou_distance(non_continued_outside_tracks, new_outside_tracks)
                print(dists.shape)
                if not tracker.args.mot20:
                    dists = matching.fuse_score(dists, new_outside_tracks)

                # find matches the "new tracks" and "old non-continued tracks"
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=tracker.args.match_thresh)

                # process each match
                # matches: that means one "new track" is actually an "old non-continued track", then the old one is
                # continued with the new one
                for itracked, idet in matches:
                    track = non_continued_outside_tracks[itracked]
                    det = new_outside_tracks[idet]
                    track.update(det, tracker.frame_id)

                # process each u_track
                # u_track: previous tracks are not continued, they just "update themselves"
                # if len(u_track) != 0:
                #     STrack.multi_predict([non_continued_outside_tracks[each] for each in u_track])
                # update them in "outside_tracks"
                for each in u_track:
                    outside_tracks[track_name(non_continued_outside_tracks[each])] = non_continued_outside_tracks[each]
                    outside_tracks[track_name(non_continued_outside_tracks[each])].img_patch = img_info["raw_img"][
                                                                                               min(max(0, int(non_continued_outside_tracks[each].tlbr[1])), 720):
                                                                                               min(max(0, int(non_continued_outside_tracks[each].tlbr[3])), 720),
                                                                                               min(max(0, int(non_continued_outside_tracks[each].tlbr[0])), 1280):
                                                                                               min(max(0, int(non_continued_outside_tracks[each].tlbr[2])), 1280),
                                                                                               :]
                    # if outside_tracks[track_name(each)].img_patch.shape[1] == 0:
                    #     print(1)

                # process each u_detection
                # u_detections: actual new_tracks
                for each in u_detection:
                    outside_tracks.update({track_name(new_outside_tracks[each]): new_outside_tracks[each]})
                    outside_tracks[track_name(new_outside_tracks[each])].img_patch = img_info["raw_img"][int(
                        new_outside_tracks[each].tlbr[1]): int(new_outside_tracks[each].tlbr[3]), int(
                        new_outside_tracks[each].tlbr[0]): int(new_outside_tracks[each].tlbr[2]), :]

                online_tlwhs = []
                online_ids = []
                online_scores = []

                # ======================================================================================================
                # use the outside tracks
                for each in outside_tracks:
                    tlwh = outside_tracks[each].tlwh
                    tid = outside_tracks[each].track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(outside_tracks[each].score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},"
                            f"{outside_tracks[each].score:.2f},-1,-1,-1\n "
                        )
                # ======================================================================================================
                # use the original tracks
                # for t in online_targets:  # for each target, plot tracking
                #     tlwh = t.tlwh
                #     tid = t.track_id
                #     vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                #     if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                #         online_tlwhs.append(tlwh)
                #         online_ids.append(tid)
                #         online_scores.append(t.score)
                #         results.append(
                #             f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},"
                #             f"{t.score:.2f},-1,-1,-1\n "
                #         )
                # ======================================================================================================
                timer.toc()
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        # change the save folder
        if IDE:
            vis_folder = "C:\\Users\\TORY\\OneDrive - Temple University\\AGI research" \
                         "\\ByteTrack-main\\YOLOX_outputs\\yolox_x_mix_det\\track_vis"
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        if IDE:
            ckpt_file = "C:/Users/TORY/OneDrive - Temple University/AGI research/ByteTrack-main/" + ckpt_file
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        # main process starts at here
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
