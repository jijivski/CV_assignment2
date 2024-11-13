conda create --name cv-assignment2 python=3.9.18
conda activate cv-assignment2
pip install imageio==2.9.0 matplotlib==3.3.4 numpy==1.21.2 pillow==9.0.0 pytorch==1.10.1 scikit-image==0.18.1 opencv-python==4.6.0.66
pip install opencv-python==4.6.0.66
python --version
pip list

conda activate cv-assignment2
cd /root/autodl-tmp/MKSC-20-0237-codes-data/data/amazon/CV_assignment2/
cd /home/chenghao/workspace/CV_assignment2/task3
python block_matching.py
CUDA_VISIBLE_DEVICES="0" python task3/train.py
CUDA_VISIBLE_DEVICES="" python task3/train.py

cd /home/chenghao/workspace/CV_assignment2/task3
unzip KITTI_2015_subset.zip
mv KITTI_2015_subset /task3/KITTI_2015_subset


pip install imageio
pip install scikit-image