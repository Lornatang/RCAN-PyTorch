import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/train --output_dir ../data/DIV2K/RCAN/train --image_size 450 --step 225 --num_workers 10")
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/valid --output_dir ../data/DIV2K/RCAN/valid --image_size 450 --step 225 --num_workers 10")
