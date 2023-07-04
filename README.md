## Intro

This project is a dev (demo) version of hybrid AI multiple object tracking (hAIMOT). It is under developing, may have frequent updates. This project is going to mainly focus on MOT17 challenge.

## Usage

1. Following the installation instructions of ByteTrack (https://github.com/ifzhang/ByteTrack/tree/main). Here is a short summary.

   ```bash
   $ cd RMOT-demo
   $ pip install -r requirements.txt
   $ python setup.py develop
   $ pip install cython
   $ pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   $ pip install cython_bbox
   ```

2. Build ONA (OpenNARS-for-Applications).

   ```bash
   $ cd RMOT-demo/OpenNARS-for-Applications-master
   $ ./build.sh
   ```

   Note that <font color="red">if you are using Windows machines, ONA needs to be built by Cygwin</font>.

3. Since this project is based on ByteTrack, it takes videos as inputs (*though it CAN process images, but videos are more convenient for extending the usage of this project, e.g., we users usually will provide a video instead of a bunch of images*), but this project is tested on MOT17 challenge, so the first step is to make videos of MOT17 dataset.

   * Download the MOT17 dataset (~5Gb), **extract** it, and put it under`RMOT-demo/data/MOT17`. It will be like:

     ```
     RMOT-demo
     |——————data
            |——————MOT17
                   |——————MOT17
                   |      |——————test
                   |      |——————train
                   |------.gitingore
                   |------imgs2video_util.py
                   |------video_spec.txt
     ```

   * Run the utility script to generate videos.

     The argument "True" is for "whether you just want to generate some specific videos instead of using all of them". If this argument (named $spec$) is set True, you may need to modify the file `RMOT-demo/data/MOT17/video_spec.txt`, to include the names of the videos you want (each line for each name).

     ```bash
     $ cd RMOT-demo/data/MOT17/MOT17
     $ python imgs2video_util.py True
     ```

     If you do want to generate all videos, it may take some time.

4. To run the "hard coded" demo.

   Note that the last "XXX" is left for you to fill, if you don't want to specify this argument, you will need to manually move the video you want to process to `RMOT-demo/OpenNARS-for-Applications-master/misc/Python/Demo/videos` and change the code of `test_hard_coded.py` L153, the default value. 

   ```bash
   $ cd RMOT-demo/OpenNARS-for-Applications-master/misc/Python/Demo
   $ python test_hard_coded.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result --path XXX
   ```

   It uses GPU by default, if you don't want to use GPU, please use the additional argument `--device cpu`.

   In some videos of MOT17 challenges, the fps is not 30, if so, please use the additional argument `--fps X`, in which $X$ is the fps you want to use. 

5. To run the "ONA embedded" demo.

   ```bash
   $ cd RMOT-demo/OpenNARS-for-Applications-master/misc/Python
   $ python test_ona_serial.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result --path XXX
   ```

   Descriptions about the arguments are the same as the 4th note above.

6. Test on MOT17.

   After running the above two demos, you will get a video and a txt file under `RMOT-demo/OpenNARS-for-Applications-master/misc/Python/Demo/YOLOX_outputs/yolox_x_mix_det/track_vis` and `RMOT-demo/OpenNARS-for-Applications-master/misc/Python/YOLOX_outputs/yolox_x_mix_det/track_vis` respectively.

   You need to manually move the txt file and rename it with the name of the corresponding video in MOT17 challenge (e.g., rename "x.txt" to "MOT17-09-DPM.txt"), then move it to `RMOT-demo/OpenNARS-for-Applications-master/misc/Python/TrackEval-master/data/trackers/mot_challenge/MOT17-train/data`. (Looks like this repo does not support testing :\\).

   Then run the following code.

   ```bash
   $ cd RMOT-demo/OpenNARS-for-Applications-master/misc/Python/TrackEval-master/scripts
   $ python run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL hAIMOT --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL True --NUM_PARALLEL_CORES 4 --SEQ_INFO X
   ```

   If your device does not support parallel cores, please set the argument `--USE_PARALLEL` to False and delete the argument `--NUM_PARALLEL_CORES`.

   Note that in the end if you don't use the argument `--SEQ_INFO`, it will run the evaluation on ALL MOT challenge videos, otherwise, you will need to spec the names in the end (replace the "X" mark, multiple names are separated by spaces. E.g., `$ python run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL hAIMOT --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL True --NUM_PARALLEL_CORES 4 --SEQ_INFO MOT17-09-DPM MOT17-10-DPM`).

## Reference

1. ByteTrack (https://github.com/ifzhang/ByteTrack/tree/main)
2. TrackEval (https://github.com/JonathonLuiten/TrackEval/tree/master)
3. ONA (https://github.com/opennars/OpenNARS-for-Applications/tree/master)
4. MOT17 (https://motchallenge.net/data/MOT17/)
5. Cygwin (https://www.cygwin.com/)
