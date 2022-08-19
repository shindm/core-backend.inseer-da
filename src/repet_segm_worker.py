# Created by Dmitriy Shin on 7/12/22 at 2:52 PM



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
import types

class ReconWorker:

    def run_recon_pipeline(self):
        Application = ApplicationClass.Application(self.args)
        Application.pipeline()
        return Application.version_number


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
                            recon_worker = ReconWorker(input_resized_video_file_path, two_json_file_path, cam_json_file_path, world_json_file_path, bbox_json_file_path,
                                                       angle_json_file_path, bbox_video_file_path, overlay_video_file_path, converted_video_file_path, two_three_video_file_path, True, True)


                            recon_worker.run_recon_pipeline()

                            cur.execute(
                                f"UPDATE recon_job SET status=1, output_2D_overlay_video_filename='{overlay_video_file_name}', output_3D_coords_filename='{world_json_file_name}',  "
                                + f"output_angles_filename='{angle_json_file_name}', completion_datetime=NOW() where id={recon_job_id};")
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







