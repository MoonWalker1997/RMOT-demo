import argparse
import copy
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
from loguru import logger

import NAR as reasoner
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker import matching
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

"""
这个文件的宗旨就是，“简单”。要使用最少的改变生成一个足够优秀的视频。太深的东西不必要加。必须包含推理。

原理：

在深度学习tracker之外，加一个新的数据结构，这并不是outside track，因为它只是对深度学习生成类的复制。
当在新一轮中，深度学习给出新的结构，在名称上一致的，直接更新，比如在上一帧我获得了名为OT1的STrack，然后在这一帧中我获得了一个新的名叫OT1的
STrack，那么直接用新的覆盖旧的。

核心考虑在于当某个STrack丢失以后，即在新的一帧中，没有新的同名STrack时，存在两种操作：1）外部track自动更新，2）搜索同名track之外的STrack，
这由推理完成。

update要用EMA，但是目前没用到. !!!!!!!!!!!!!!!!!!!!!

和draft4差不多，只不过没用推理，而且虽然NN会生成很鬼畜且错误的东西，我先相信，后面再改

"""

IDE = True
video_name = "street"
np.set_printoptions(2, suppress=True)
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


def distance_to_Narsese(online_target, outside_track_id, distance_similarity):
    A = "<(*, T" + str(online_target.track_id) + ", " + outside_track_id + ") --> close>. %" + str(
        distance_similarity) + ";0.8%"
    B = "(--, <(*, T" + str(online_target.track_id) + ", " + outside_track_id + ") --> close>). %" + str(
        1 - distance_similarity) + ";0.8%"
    return [A, B]


def appearance_model_to_Narsese(online_target, outside_track_id, appearance_similarity):
    A = "<(*, T" + str(online_target.track_id) + ", " + outside_track_id + ") --> similar>. %" + str(
        appearance_similarity) + ";0.8%"
    B = "(--, <(*, T" + str(online_target.track_id) + ", " + outside_track_id + ") --> similar>). %" + str(
        1 - appearance_similarity) + ";0.8%"
    return [A, B]


def question_belong_to_Narsese(online_target, outside_track_id):
    return ["<(*, T" + str(online_target.track_id) + ", " + outside_track_id + ") --> belong>)?"]


def question_occluded_to_Narsese(outside_track_id):
    return ["<" + outside_track_id + " --> occluded>?"]


def appearance_similarity(online_targets, outside_tracks, img):
    ret = np.zeros((len(online_targets), len(outside_tracks)))
    for i in range(len(online_targets)):
        for j in range(len(outside_tracks)):
            sim = 0
            img_patch_online_target = \
                img[
                min(img.shape[0], max(0, int(online_targets[i].tlbr[1]))):
                min(img.shape[0], max(0, int(online_targets[i].tlbr[3]))),
                min(img.shape[1], max(0, int(online_targets[i].tlbr[0]))):
                min(img.shape[1], max(0, int(online_targets[i].tlbr[2]))), :]

            blue_target = cv2.calcHist([img_patch_online_target], [0], None, [256], [0, 256])
            red_target = cv2.calcHist([img_patch_online_target], [1], None, [256], [0, 256])
            green_target = cv2.calcHist([img_patch_online_target], [2], None, [256], [0, 256])

            for k in range(len(outside_tracks[j].appearance_models)):
                tmp = 1
                tmp *= cv2.compareHist(blue_target,
                                       outside_tracks[j].appearance_models[k][0], 0)
                tmp *= cv2.compareHist(red_target,
                                       outside_tracks[j].appearance_models[k][1], 0)
                tmp *= cv2.compareHist(green_target,
                                       outside_tracks[j].appearance_models[k][2], 0)

                sim += tmp

            sim /= len(outside_tracks[j].appearance_models) + 0.1

            ret[i, j] = sim

    return ret


def output_util(dic1, dic2):
    for each_key in dic2:
        if dic2[each_key] is not None:
            if each_key != "requestOutputArgs":
                dic1[each_key] += dic2[each_key]
            else:
                dic1[each_key].append(dic2[each_key])


win_size = (16, 16)
block_size = (8, 8)
block_stride = (4, 4)
cell_size = (4, 4)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
win_stride = (8, 8)
padding = (0, 0)


def generate_hog(target, img):
    img_patch = cv2.resize(img[min(img.shape[0], max(0, int(target.tlbr[1]))):
                               min(img.shape[0], max(0, int(target.tlbr[3]))),
                           min(img.shape[1], max(0, int(target.tlbr[0]))):
                           min(img.shape[1], max(0, int(target.tlbr[2]))), :], (64, 64))
    return hog.compute(img_patch)


def hog_to_Narsese(online_target, outside_track, outside_track_id, img):
    A = cv2.resize(img[min(img.shape[0], max(0, int(online_target.tlbr[1]))):
                       min(img.shape[0], max(0, int(online_target.tlbr[3]))),
                   min(img.shape[1], max(0, int(online_target.tlbr[0]))):
                   min(img.shape[1], max(0, int(online_target.tlbr[2]))), :], (64, 64))
    B = cv2.resize(img[min(img.shape[0], max(0, int(outside_track.tlbr[1]))):
                       min(img.shape[0], max(0, int(outside_track.tlbr[3]))),
                   min(img.shape[1], max(0, int(outside_track.tlbr[0]))):
                   min(img.shape[1], max(0, int(outside_track.tlbr[2]))), :], (64, 64))

    A = hog.compute(A, win_stride, padding)
    B = hog.compute(B, win_stride, padding)

    similarity = A.dot(B) / ((A.dot(A) ** 0.5) * (B.dot(B) ** 0.5))
    A = "<(*, T" + str(online_target.track_id) + ", " + outside_track_id + ") --> hog_similar>. %" + str(
        similarity) + ";0.8%"
    B = "(--, <(*, T" + str(online_target.track_id) + ", " + outside_track_id + ") --> hog_similar>). %" + str(
        1 - similarity) + ";0.8%"

    return [A, B]


def a_priori_Narsese(r):
    r.AddInput("<(&&, (--, <(*, $x, $y) --> hog_similar>), <(*, $x, $y) --> close) ==> <$y --> occluded>>.", False)
    r.AddInput("<<(*, $x, $y) --> similar) ==> <(*, $x, $y) --> belong>>.", False)
    r.AddInput("<<(*, $x, $y) --> close) ==> <(*, $x, $y) --> belong>>.", False)
    r.AddInput("<<(*, $x, $y) --> hog_similar) ==> <(*, $x, $y) --> belong>>.", False)


def imageflow_demo(predictor, vis_folder, current_time, args):
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
    Method:

    1. DL tracker generate some bboxes.
    2. Outside tracks predict.
    3. IoU distance.
    4. Appearance distance.
    5. Match outside track and online target.
    6. For those unmatched, generate Narsese and do reasoning.
    7. Analyzing the reasoning results and then modify the distance matrix.
    8. Match based on the distance matrix.

    """

    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()

    critical_frame = 45
    frame_id = 0
    results = []

    # a dictionary of tracks {track name: track object}, managed outside the "byte tracker"
    outside_tracks = {}
    outside_tracks_count = 0

    while True:  # each frame

        if frame_id % 20 == 0:  # data logging
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        if frame_id == 104:
            print(1)

        # video length capping
        # if frame_id // 10 == 5:
        #     break

        # tracklet decay
        for each in outside_tracks:
            outside_tracks[each].retire()

        ret_val, frame = cap.read()  # imread
        if ret_val:
            # only high quality (class quality) bboxes remain in outputs
            # the filtering is based on a pre-defined threshold (0.001)
            outputs, img_info = predictor.inference(frame, timer)  # predictor inference, tracks and ids
            if outputs[0] is not None:

                # Step 1:
                # "online targets" are just updated targets
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

                # Step 2:
                # all outside tracks update themselves
                for each in outside_tracks:
                    outside_tracks[each].predict()
                    outside_tracks[each].mean[7] = 0

                # Step 3:
                # synchronize with online targets
                new_faces = []
                for each in online_targets:

                    if track_name(each) in outside_tracks:
                        # just "trust" the deep learning model
                        outside_tracks[track_name(each)].update(each, frame_id)
                    else:
                        new_faces.append(copy.deepcopy(each))

                # test whether this is really a new track, or this is an old one, by reasoning

                # iou is a basic evaluation, if two tracks are too far away
                iou_similarity = 1 - matching.iou_distance(new_faces,
                                                           [outside_tracks[each] for each in outside_tracks])

                updates = {}

                for i, each in enumerate(new_faces):
                    # for all existed outside track
                    belong = False
                    for j, each_track in enumerate(outside_tracks):
                        if belong:
                            break
                        if iou_similarity[i, j] > 0.7:
                            # consistent with an old one
                            outside_tracks[each_track].update(each, frame_id)
                            belong = True
                            break

                    if not belong:
                        updates.update({track_name(each): copy.deepcopy(each)})

                for each in updates:
                    outside_tracks.update({each: updates[each]})

                online_tlwhs = []
                online_ids = []
                online_scores = []

                # ======================================================================================================
                # use the outside tracks
                for each in outside_tracks:
                    tlwh = outside_tracks[each].tlwh
                    tid = int(each[2:])
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(outside_tracks[each].score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},"
                            f"{outside_tracks[each].score:.2f},-1,-1,-1\n "
                        )
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
