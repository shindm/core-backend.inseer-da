Inseer Mvp readme
Install anaconda (python version 3.7.1)
Open terminal/cmd and locate move to folder InseerMvpPython

Create an anconda virtual enviroment with the following packages (install method):
python=3.7.1 (conda)
tensorflow-gpu=2.0 (conda)
Python opencv – 4.5.1 (pip)
Cudatoolkit = 10.0.130 (conda)
Cudnn = 7.6.5 (conda)
Numpy = 1.19.X (conda)
Ffmpeg 4.2.2 (conda)
Ffmpeg – python 0.2.0 (pip)
Folder Structure:
Top folder: python_processing
Sub-folders: src (contains subfolders to three modules), pose_model (.h5 weight path to two different pose estimators), checkpoints (tfV1.X  checkpoint data for 3d-pose-estimation) 

PoseEst Use:
Main_Fast.py cmd args:
-i –input absolute path to video of interest
-o1 –output_1 skeletal overlay video file path (absolute path)
-o2 –output_2 absolute path to write out json file of joint locations in pixel coordinates
-m1 -model_1 absolute path to pose estimation model (located in ./InseerPythonMvp/pose_model
-w1 -flag1 boolean if True write output_1 
-w2 -flag2 boolean if True write output_2
-MT -model_type string args (COCO or Body_25) use Body_25 for faster computation 50% time reduction and ~equal mAP)

Threed Reconstruction Use:
ThreedMain.py cmd args:
-i –input absolute path o1 from PoseEst (expecting a .json file containing the 2d joint locations)
-o1 –output_1 absolute path to unNormalized 3d coordinates of joints
-o2 –output_1 absolute path to translated joints json file (input for AngleEstModule)
-m –model model directory (./InseerPythonMvp/checkpoints)
-c –check (0)
-w1 -flag1 boolean if True write output_1 
-w2 -flag2 boolean if True write output_2
-MT -model_type (Again Coco or Body_25 if seconde option convert call will be implemented in main)

AngleEstimation Use:
AnglesMain.py cmd args:
-i inputs absolute path to -o2 from Threed Reconstruction (expecting the normalized 3d joints as a .json file)
-o --output absolute path to angle json file output
-w -flag if True output writes to .json file