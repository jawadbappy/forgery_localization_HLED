#!/bin/bash

THISSCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#echo $THISSCRIPTPATH

echo "Installing prerequisites, copied from http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/"
sudo apt-get install -y build-essential cmake pkg-config python-pip
sudo apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev libgtk2.0-dev libatlas-base-dev gfortran python2.7-dev python3.5-dev libpython2.7-dev
sudo pip install numpy scipy

echo "Downloading and unzipping OpenCV 2.4.13 to \"opencv-2.4.13\" folder"
wget -nc https://github.com/Itseez/opencv/archive/2.4.13.zip
unzip -u 2.4.13.zip
cd opencv-2.4.13

echo "Building OpenCV"
#rm -rf build
mkdir -p build
(cd build/ && cmake -D CMAKE_BUILD_TYPE=Release \
	-D WITH_CUDA=OFF -D WITH_OPENMP=ON \
	-D CMAKE_INSTALL_PREFIX=$THISSCRIPTPATH/opencv-2.4.13/build_release .. && make all -j12 && make install)

