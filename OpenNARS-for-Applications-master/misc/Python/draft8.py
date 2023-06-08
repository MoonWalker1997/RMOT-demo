import argparse
import copy
import os
import os.path as osp
import time

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker import matching
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

"""
加入appearance model。
希望这次能够成功。
"""

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
IDE = True
video_name = "street"
np.set_printoptions(2, suppress=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class outside_track:
    """
    outside track的构造是复制由DL模型产生的track。
    """

    def __init__(self, target, appearances = None):
        self.kalman_filter = copy.deepcopy(target.kalman_filter)
        self.mean, self.cov = target.mean, target.covariance
        self.appearances = appearances
        self.active = True
        self.score = 0

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.cov = self.kalman_filter.predict(mean_state, self.cov)

    @property
    def tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def update(self, new_track, appearances = None, similarity = None):
        self.active = True
        new_tlwh = new_track.tlwh
        mean, covariance = self.kalman_filter.update(
            self.mean, self.cov, self.tlwh_to_xyah(new_tlwh))

        self.score = new_track.score

        if appearances is not None:
            self.appearances[0] = self.appearances[0] * 0.9 + appearances[0] * 0.1
            self.appearances[1] = self.appearances[1] * 0.9 + appearances[1] * 0.1
            self.appearances[2] = self.appearances[2] * 0.9 + appearances[2] * 0.1
            mean = self.mean * 0.5 + mean * 0.5
            self.mean = mean
            self.cov = covariance
        else:
            mean = self.mean * (1 - similarity) + mean * similarity
            self.mean = mean
            self.cov = covariance


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
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

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
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


def give_img_patch(target, img_info):
    return img_info["raw_img"][
           min(img_info["raw_img"].shape[0], max(0, int(target.tlbr[1]))):
           min(img_info["raw_img"].shape[0], max(0, int(target.tlbr[3]))),
           min(img_info["raw_img"].shape[1], max(0, int(target.tlbr[0]))):
           min(img_info["raw_img"].shape[1], max(0, int(target.tlbr[2]))), :]


def show_img_patch(target, img_info):
    img_patch = give_img_patch(target, img_info)
    plt.imshow(img_patch)
    plt.show()


def generate_color_hist(img_patch):
    img = cv2.resize(img_patch, (64, 64))
    blue_track = cv2.calcHist([img], [0], None, [256], [0, 256])
    red_track = cv2.calcHist([img], [1], None, [256], [0, 256])
    green_track = cv2.calcHist([img], [2], None, [256], [0, 256])

    return [blue_track, red_track, green_track]


def compare_color_hist(hist1, hist2):
    return cv2.compareHist(hist1[0], hist2[0], 0) * cv2.compareHist(hist1[1], hist2[1], 0) * cv2.compareHist(
        hist1[2], hist2[2], 0)


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
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []

    outside_tracks = {}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    win_size = (16, 16)
    block_size = (8, 8)
    block_stride = (4, 4)
    cell_size = (4, 4)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    win_stride = (8, 8)
    padding = (0, 0)

    ban_list = []

    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

                # ======================================================================================================

                if frame_id == 104:
                    print(1)

                iou_similarity = 1 - matching.iou_distance(online_targets,
                                                           [outside_tracks[each] for each in outside_tracks],
                                                           img_info["raw_img"].shape)

                # DL consistent
                I1 = []
                for i, each in enumerate(online_targets):
                    if each.track_id in outside_tracks:
                        if iou_similarity[i, list(outside_tracks.keys()).index(each.track_id)] > 0.6:

                            if each.track_id == 6:
                                print(1)

                            hist_target = generate_color_hist(give_img_patch(each, img_info))

                            if compare_color_hist(hist_target, outside_tracks[each.track_id].appearances) > 0.8:
                                outside_tracks[each.track_id].update(new_track=each, appearances=hist_target)
                                I1.append(i)
                                continue
                            else:
                                outside_tracks[each.track_id].update(new_track=each, similarity=compare_color_hist(hist_target, outside_tracks[each.track_id].appearances))
                                I1.append(i)
                                continue
                        else:
                            ban_list.append(each.track_id)

                # global consistent
                I2 = []
                for i, each in enumerate(online_targets):
                    if i in I1:
                        continue
                    for j, each_id in enumerate(outside_tracks):
                        if iou_similarity[i, j] == 0:
                            continue
                        if iou_similarity[i, j] > 0.60:

                            hist_target = generate_color_hist(give_img_patch(each, img_info))

                            if compare_color_hist(hist_target, outside_tracks[each_id].appearances) > 0.8:
                                outside_tracks[each_id].update(new_track=each, appearances=hist_target)
                                I2.append(i)
                                break
                            else:
                                outside_tracks[each_id].update(new_track=each, similarity=compare_color_hist(hist_target, outside_tracks[each_id].appearances))
                                I2.append(i)
                                break

                for i, each in enumerate(online_targets):
                    if i in I1 + I2:
                        continue
                    # brand-new track creation
                    if each.score > 0.9 and each.track_id not in ban_list:
                        hist_target = generate_color_hist(give_img_patch(each, img_info))
                        outside_tracks[each.track_id] = outside_track(each, hist_target)

                for each in outside_tracks:
                    if not outside_tracks[each].active:
                        outside_tracks[each].mean[7] = 0
                    outside_tracks[each].predict()
                    outside_tracks[each].active = False

                # ======================================================================================================

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for each in outside_tracks:
                    tlwh = outside_tracks[each].tlwh
                    tid = each
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(outside_tracks[each].score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{outside_tracks[each].score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
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
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
