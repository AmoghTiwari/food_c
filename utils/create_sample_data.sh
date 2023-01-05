#/bin/bash

cd /ssd_scratch/cvit/amoghtiwari/food_c_challenge/data/

mkdir food_c_data_sample
cd food_c_data_sample
mkdir train_images
touch train.csv

echo "ImageId,ClassName" >> train.csv
echo "f27632d7e5.jpg,water" >> train.csv
echo "efa87919ed.jpg,pizza-margherita-baked" >> train.csv
echo "4f169e8c8d.jpg,broccoli" >> train.csv
echo "a6956654bf.jpg,salad-leaf-salad-green" >> train.csv
echo "d99ce8c3bf.jpg,egg" >> train.csv
echo "0c2b1641a8.jpg,butter" >> train.csv
echo "3f7e5ed3a9.jpg,bread-white" >> train.csv
echo "ffcfba255c.jpg,butter" >> train.csv


cp ../food_c_data/train_images/f27632d7e5.jpg ./train_images/
cp ../food_c_data/train_images/efa87919ed.jpg ./train_images/
cp ../food_c_data/train_images/4f169e8c8d.jpg ./train_images/
cp ../food_c_data/train_images/a6956654bf.jpg ./train_images/
cp ../food_c_data/train_images/d99ce8c3bf.jpg ./train_images/
cp ../food_c_data/train_images/0c2b1641a8.jpg ./train_images/
cp ../food_c_data/train_images/3f7e5ed3a9.jpg ./train_images/
cp ../food_c_data/train_images/ffcfba255c.jpg ./train_images/

cd ~/personal_projects/food_c_challenge

