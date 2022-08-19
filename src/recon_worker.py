# Created by Dmitriy Shin on 7/11/22 at 12:10 PM



from datetime import datetime
import mariadb
import gc
import torch
import os
from os.path import exists
import time
import shutil
import cv2
import pathlib
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#append TrajectoryRecon folder to system path
parent_folder=pathlib.Path(__file__).parent.resolve()
absolute_path_PoseEst=os.path.join(parent_folder, 'PoseEst')
absolute_path_TrajRecon=os.path.join(parent_folder, 'TrajectoryRecon')
absolute_path_yolov5=os.path.join(parent_folder, 'YoloV5_DeepSort')
absolute_path_AngleEst=os.path.join(parent_folder, 'AngleEst')
absolute_path_Force=os.path.join(parent_folder, 'Force')
sys.path.append(absolute_path_PoseEst)
sys.path.append(absolute_path_TrajRecon)
sys.path.append(absolute_path_yolov5)
sys.path.append(absolute_path_AngleEst)
sys.path.append(absolute_path_Force)

import argparse
import ApplicationClass
import types

class ReconWorker:
    args = types.SimpleNamespace()

    def __init__(self, orginal_video_location, two_d_output, cam_output, world_output, bbox_output, angle_output, bbox_video_out, overlay_out, h264_video_out, two_three_video_out, run_detect, run_track):
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--video_path', help = "absolute path to video input")
        parser.add_argument('-dm', '--detection_model', help = "absolute path to detection model", default='YoloV5_DeepSort/best.pt')
        parser.add_argument('-tm', '--tracking_model', help = "absolute path to tracking model", default='YoloV5_DeepSort/deepsort/deep_128.pb')
        parser.add_argument('-pm', '--pose_model', help = "absolute path to pose estimation model", default="PoseEst/pose_model/poseEstimationModel25.h5")
        parser.add_argument('-rm','--recon_model', help = "absolute path to recon model", default="TrajectoryRecon/checkpoint/recon_model.pt")
        parser.add_argument('-tj', '--two_json', help = "absolute path to two d output json file")
        parser.add_argument('-cj', '--cam_json', help = "absolute path to camera coordinate json file")
        parser.add_argument('-wj', '--world_json', help = "absolute path to world coordinate json file")
        parser.add_argument('-bbj', '--bbox_json', help = "absolute path where to write bbox json")
        parser.add_argument('-aj', '--angle_json',help = "absolute path where to write angles json")

        parser.add_argument('-ds', '--detection_size', help = "detection resolution change wrt to vram limits", default = 384,type = int)
        parser.add_argument('-MT', '--model_type', help = "model type for pose estimation", default="Body_25")
        parser.add_argument('-bbv', '--bbox_video', help = "absolute path to bbox video out")
        parser.add_argument('-ov', '--overlay_video', help = "absolute path to skeletal overlay video")
        parser.add_argument('-c', '--converted_video', help = "absolute path to converted video with h264 codec")
        parser.add_argument('--classes', help="path to text file with object classes", default='yolov5/data/coco.names.txt')

        #default args only change size for detection module if vram constraints require
        parser.add_argument('-s', '--size', type=int, default=384)
        parser.add_argument('-iou', '--iou', default=0.45, type=float)
        parser.add_argument('-sc' ,'--score', default=0.50, type=float)
        parser.add_argument('-m', '--model', default='yolov5')
        parser.add_argument('--write_output', default=True, type=bool)
        '''

        '''
        new how to run it:
        python3 ApplicationMain.py -v ./data/IMG_2330.MOV -dm ./YoloV5_DeepSort/best.pt -tm ./YoloV5_DeepSort/deep_sort/deep_128.pb 
        -pm ./PoseEst/pose_model/poseEstimationModel25.h5 -rm ./TrajectoryRecon/checkpoint/recon_model.pt -tj two_json.json -cj cam_json.json 
        -wj world_json.json -bbj box_json.json -ov overlay_video.mp4  -MT Body_25 -aj angles.json 
        --classes ./YoloV5_DeepSort/yolov5/data/coco.names.txt -bbv box_video.mp4 -c converted_video.mp4
        '''


        self.args.video_path = orginal_video_location
        self.args.pose_model = "./PoseEst/pose_model/poseEstimationModel25.h5"
        self.args.detection_model = "./YoloV5_DeepSort/best.pt"
        self.args.tracking_model = "./YoloV5_DeepSort/deep_sort/deep_128.pb"
        self.args.recon_model = "./TrajectoryRecon/checkpoint/recon_model.pt"
        self.args.model_type = "Body_25"
        self.args.converted_video = h264_video_out
        self.args.overlay_video = overlay_out
        self.args.two_json = two_d_output
        self.args.cam_json = cam_output
        self.args.angle_json = angle_output
        self.args.world_json = world_output
        self.args.bbox_video = bbox_video_out
        self.args.bbox_json = bbox_output
        self.args.classes = 'YoloV4/class_names/coco_classes.txt'
        self.args.detection_size = 512
        self.args.run_detect = run_detect
        self.args.run_track = run_track
        self.args.size = 384
        self.args.iou = .45
        self.args.score = .05
        self.args.model = 'yolov5'
        self.args.write_output = True

    def run_recon_pipeline(self):
        Application = ApplicationClass.Application(self.args)
        Application.pipeline()
        return Application.version_number



def resize_image(image, width, height,COLOUR=[0,0,0]):
    h, w, layers = image.shape
    if h > height:
        ratio = height/h
        image = cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if w > width:
        ratio = width/w
        image = cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if h < height and w < width:
        hless = height/h
        wless = width/w
        if(hless < wless):
            image = cv2.resize(image, (int(image.shape[1] * hless), int(image.shape[0] * hless)))
        else:
            image = cv2.resize(image, (int(image.shape[1] * wless), int(image.shape[0] * wless)))
    h, w, layers = image.shape
    if h < height:
        df = height - h
        df /= 2
        image = cv2.copyMakeBorder(image, int(df), int(df), 0, 0, cv2.BORDER_CONSTANT, value=COLOUR)
    if w < width:
        df = width - w
        df /= 2
        image = cv2.copyMakeBorder(image, 0, 0, int(df), int(df), cv2.BORDER_CONSTANT, value=COLOUR)
    image = cv2.resize(image,(1080,484),interpolation=cv2.INTER_AREA)
    return image


VIDEO_UPLOADS_FOLDER = "/home/dvshin/DEPLOYMENTS/INSEER_DA_WEB_APP/video_uploads"
PROCESSED_VIDEO_UPLOADS_FOLDER = "/home/dvshin/DEPLOYMENTS/INSEER_DA_WEB_APP/processed_video_uploads"
WEB_OUTPUT_VIDEO_FILES_FOLDER = "/var/www/html/inseer_da_videos"
WEB_OUTPUT_DATA_FILES_FOLDER = "/var/www/html/inseer_da_files"
OUTPUT_VIDEO_FILES_FOLDER = "/home/dvshin/DEPLOYMENTS/INSEER_DA_WEB_APP/recon_video_files"
OUTPUT_DATA_FILES_FOLDER = "/home/dvshin/DEPLOYMENTS/INSEER_DA_WEB_APP/recon_data_files"

if __name__ == "__main__":
    # loop here to check if input video file exist
    # if so process it
    conn = None
    cur = None
    try:
        # instantiate DB connection
        conn = mariadb.connect(
            host= 'localhost',
            user= 'da_demo_app_db_user',
            password= 'da_demo_app_db',
            database= "da_demo_app_db",
            port=3306)
        #conn.timezone = "local"
        # instantiate cursor
        cur = conn.cursor()
        while(True):
            try:
                conn.commit()
                # cur.execute("COMMIT;")
                # explicitly initiate DB transaction
                cur.execute("START TRANSACTION;")
                cur.execute(f"SELECT id, input_video_filename FROM recon_job WHERE status=0 LIMIT 1 FOR UPDATE SKIP LOCKED;");
                row = cur.fetchone()
                if row:
                    recon_job_id = row[0]
                    input_video_filename = row[1]
                    if exists(VIDEO_UPLOADS_FOLDER + "/" + input_video_filename):
                        # check if input_video_filename has mp4 extension
                        f_tokens = input_video_filename.split(".")
                        if f_tokens[1] == "mp4":
                            input_video_file_path = VIDEO_UPLOADS_FOLDER + "/" + input_video_filename
                            input_resized_video_file_path = VIDEO_UPLOADS_FOLDER + "/" + f_tokens[0] + "_resized.mp4"
                            converted_video_file_name = f_tokens[0] + "_converted.mp4"
                            converted_video_file_path = OUTPUT_VIDEO_FILES_FOLDER + "/" + converted_video_file_name
                            overlay_video_file_name = f_tokens[0] + "_overlay.mp4"
                            overlay_video_file_path = OUTPUT_VIDEO_FILES_FOLDER + "/" + overlay_video_file_name

                            two_json_file_path = OUTPUT_DATA_FILES_FOLDER + "/" + f_tokens[0] + "_twojson.json"
                            cam_json_file_path = OUTPUT_DATA_FILES_FOLDER + "/" + f_tokens[0] + "_camjson.json"

                            world_json_file_name = f_tokens[0] + "_worldjson.json"
                            world_json_file_path = OUTPUT_DATA_FILES_FOLDER + "/" + world_json_file_name
                            angle_json_file_name = f_tokens[0] + "_anglejson.json"
                            angle_json_file_path = OUTPUT_DATA_FILES_FOLDER + "/" + angle_json_file_name

                            bbox_video_file_path = OUTPUT_VIDEO_FILES_FOLDER + "/" + f_tokens[0] + "_bbox.mp4"
                            bbox_json_file_path = OUTPUT_DATA_FILES_FOLDER + "/" + f_tokens[0] + "_bboxjson.json"
                            two_three_video_file_path = OUTPUT_VIDEO_FILES_FOLDER + "/" + f_tokens[0] + "_twothree.mp4"

                            # resize video to 1080px x 484px with filling padding
                            cap = cv2.VideoCapture(input_video_file_path)
                            fourcc = cv2.VideoWriter_fourcc(*'H264')  # encodes H264 even though gives errors
                            out = cv2.VideoWriter(input_resized_video_file_path, fourcc, 30.0, (1080, 484))
                            cap.set(1, 1)  # set the first frame
                            ret, frame = cap.read()
                            while ret:
                                resized_frame = resize_image(frame, 1080, 484, [46, 46, 46])
                                out.write(resized_frame)
                                ret, frame = cap.read()
                            cap.release()
                            out.release()

                            recon_worker = ReconWorker(input_resized_video_file_path, two_json_file_path, cam_json_file_path, world_json_file_path, bbox_json_file_path,
                                                       angle_json_file_path, bbox_video_file_path, overlay_video_file_path, converted_video_file_path, two_three_video_file_path, True, True)


                            recon_worker.run_recon_pipeline()

                            cur.execute(
                                f"UPDATE recon_job SET status=1, output_2D_overlay_video_filename='{overlay_video_file_name}', output_3D_coords_filename='{world_json_file_name}',  "
                                + f"output_angles_filename='{angle_json_file_name}', completion_datetime=NOW() where id={recon_job_id};")

                            # move processed input video file in to processed folder
                            os.rename(input_video_file_path, PROCESSED_VIDEO_UPLOADS_FOLDER + "/" + input_video_filename)
                            os.rename(input_resized_video_file_path, PROCESSED_VIDEO_UPLOADS_FOLDER + "/" + f_tokens[0] + "_resized.mp4")
                            # move files needed for the web app to web folders
                            shutil.copy(overlay_video_file_path, WEB_OUTPUT_VIDEO_FILES_FOLDER + "/" + overlay_video_file_name)
                            shutil.copy(converted_video_file_path, WEB_OUTPUT_VIDEO_FILES_FOLDER + "/" + converted_video_file_name)
                            shutil.copy(overlay_video_file_path, WEB_OUTPUT_VIDEO_FILES_FOLDER + "/" + overlay_video_file_name)
                            shutil.copy(world_json_file_path, WEB_OUTPUT_DATA_FILES_FOLDER + "/" + world_json_file_name)
                            shutil.copy(angle_json_file_path, WEB_OUTPUT_DATA_FILES_FOLDER + "/" + angle_json_file_name)

                            #cur.execute("COMMIT;")
                            conn.commit()

                now = datetime.now()
                # dd/mm/YY H:M:S
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print("Current date and time: ", dt_string)
                total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                print("RAM memory % used:", round((used_memory / total_memory) * 100, 2))
                time.sleep(5)
            except Exception as error:
                print(error)
            except mariadb.Error as e:
                print(f"MariaDB error: {e}")
            continue
    except Exception as error:
           print(error)
    except mariadb.Error as e:
           print(f"MariaDB error: {e}")
    cur.close()
    conn.close()
    exit(0)






