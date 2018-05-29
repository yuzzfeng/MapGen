# MapGen

### Introduction

Building simplification is conducted in this work using a fully conv layers with down-conv and up-conv. The original work was provided by a master thesis, which use the car trajectories to reconstruct the road networks.

### Dependencies and Settings

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64 &&
export PATH=$PATH:/usr/local/nvidia/bin:/usr/local/sbin:/usr/sbin:/sbin &&
apt-get update &&
apt-get install cifs-utils -y &&
apt-get install git -y &&
apt-get install python3-pip -y &&
pip3 install keras &&
apt-get install python3-tk -y &&
apt-get install python3-skimage -y &&
apt install gdal-bin python-gdal python3-gdal -y &&
mkdir tmp &&
mount -t cifs -o user=,password= //130.75.51.38/tmp/yu tmp &&
cd tmp 
python3 simply.py
curl -X DELETE http://130.75.51.24/marathon/v2/apps/feng-1gpu
```


### Git Commands

```
git add -A && git commit -m "Your Message"
```