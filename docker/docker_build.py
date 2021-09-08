# execute the docker build command
import os

os.system("docker build --build-arg USER_ID=$UID -t detectron2:v0 .")
