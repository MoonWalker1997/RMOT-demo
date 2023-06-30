import argparse
import os

import cv2


def make_parser():
    parser = argparse.ArgumentParser("imgs2videos util")
    parser.add_argument(
        "--spec", default=False,
        help="whether to use 'video_spec.txt', if used, only names in that file will be processed."
    )
    return parser


if __name__ == "__main__":

    args = make_parser().parse_args()

    videos_spec = []
    if args.spec:
        videos_spec = open("video_spec.txt").readlines()
        videos_spec = [each.strip("\n") for each in videos_spec]

    test = "./MOT17/test/"
    train = "./MOT17/train/"

    video_configs = {"MOT17-13-SDP": {"FPS": 25, "size": (1920, 1080), "length": 750},
                     "MOT17-11-SDP": {"FPS": 30, "size": (1920, 1080), "length": 900},
                     "MOT17-10-SDP": {"FPS": 30, "size": (1920, 1080), "length": 654},
                     "MOT17-09-SDP": {"FPS": 30, "size": (1920, 1080), "length": 525},
                     "MOT17-05-SDP": {"FPS": 14, "size": (640, 480), "length": 837},
                     "MOT17-04-SDP": {"FPS": 30, "size": (1920, 1080), "length": 1050},
                     "MOT17-02-SDP": {"FPS": 30, "size": (1920, 1080), "length": 600},
                     "MOT17-13-FRCNN": {"FPS": 25, "size": (1920, 1080), "length": 750},
                     "MOT17-11-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 900},
                     "MOT17-10-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 654},
                     "MOT17-09-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 525},
                     "MOT17-05-FRCNN": {"FPS": 14, "size": (640, 480), "length": 837},
                     "MOT17-04-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 1050},
                     "MOT17-02-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 600},
                     "MOT17-13-DPM": {"FPS": 25, "size": (1920, 1080), "length": 750},
                     "MOT17-11-DPM": {"FPS": 30, "size": (1920, 1080), "length": 900},
                     "MOT17-10-DPM": {"FPS": 30, "size": (1920, 1080), "length": 654},
                     "MOT17-09-DPM": {"FPS": 30, "size": (1920, 1080), "length": 525},
                     "MOT17-05-DPM": {"FPS": 14, "size": (640, 480), "length": 837},
                     "MOT17-04-DPM": {"FPS": 30, "size": (1920, 1080), "length": 1050},
                     "MOT17-02-DPM": {"FPS": 30, "size": (1920, 1080), "length": 600},
                     "MOT17-14-SDP": {"FPS": 25, "size": (1920, 1080), "length": 750},
                     "MOT17-12-SDP": {"FPS": 30, "size": (1920, 1080), "length": 900},
                     "MOT17-08-SDP": {"FPS": 30, "size": (1920, 1080), "length": 625},
                     "MOT17-07-SDP": {"FPS": 30, "size": (1920, 1080), "length": 500},
                     "MOT17-06-SDP": {"FPS": 14, "size": (640, 480), "length": 1194},
                     "MOT17-03-SDP": {"FPS": 30, "size": (1920, 1080), "length": 1500},
                     "MOT17-01-SDP": {"FPS": 30, "size": (1920, 1080), "length": 450},
                     "MOT17-14-FRCNN": {"FPS": 25, "size": (1920, 1080), "length": 750},
                     "MOT17-12-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 900},
                     "MOT17-08-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 625},
                     "MOT17-07-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 500},
                     "MOT17-06-FRCNN": {"FPS": 14, "size": (640, 480), "length": 1194},
                     "MOT17-03-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 1500},
                     "MOT17-01-FRCNN": {"FPS": 30, "size": (1920, 1080), "length": 450},
                     "MOT17-14-DPM": {"FPS": 25, "size": (1920, 1080), "length": 750},
                     "MOT17-12-DPM": {"FPS": 30, "size": (1920, 1080), "length": 900},
                     "MOT17-08-DPM": {"FPS": 30, "size": (1920, 1080), "length": 625},
                     "MOT17-07-DPM": {"FPS": 30, "size": (1920, 1080), "length": 500},
                     "MOT17-06-DPM": {"FPS": 14, "size": (640, 480), "length": 1194},
                     "MOT17-03-DPM": {"FPS": 30, "size": (1920, 1080), "length": 1500},
                     "MOT17-01-DPM": {"FPS": 30, "size": (1920, 1080), "length": 450},
                     }

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    print("--processing testing videos--")

    for each_video in os.listdir(test):
        if args.spec and each_video not in videos_spec:
            continue
        video_writer = cv2.VideoWriter(os.path.join(test, each_video, "video.mp4"), fourcc,
                                       video_configs[each_video]["FPS"],
                                       video_configs[each_video]["size"])
        for i in range(1, video_configs[each_video]["length"] + 1):
            frame = cv2.imread(os.path.join(test, each_video, "img1", str(i).zfill(6) + ".jpg"))
            video_writer.write(frame)

        video_writer.release()
        print("|Video:", each_video, "processed.")

    print("--testing videos processing finished--")
    print("--------------------------------------")

    print("--processing training videos--")

    for each_video in os.listdir(train):
        if args.spec and each_video not in videos_spec:
            continue
        video_writer = cv2.VideoWriter(os.path.join(train, each_video, "video.mp4"), fourcc,
                                       video_configs[each_video]["FPS"],
                                       video_configs[each_video]["size"])
        for i in range(1, video_configs[each_video]["length"] + 1):
            frame = cv2.imread(os.path.join(train, each_video, "img1", str(i).zfill(6) + ".jpg"))
            video_writer.write(frame)

        video_writer.release()
        print("|Video:", each_video, "processed.")

    print("--training videos processing finished--")
