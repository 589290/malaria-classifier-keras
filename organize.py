# organize imports
import os
import glob
import datetime
import random

# get the input and output path
input_path  = "/x/data/uninfected/"
output_path = "/x/train/uninfected/"

image_paths = glob.glob(input_path + "*.jpg")
random.shuffle(image_paths)

# loop over the images in the dataset
for image_path in image_paths[0:10000]:
  original_path = image_path
  image_path = image_path.split("/")
  image_path = image_path[len(image_path)-1]
  os.system("mv " + original_path + " " + output_path + image_path)
