FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

MAINTAINER Milan van Dijck <miscoriadev@gmail.com>

RUN apt update
RUN apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y 
RUN apt update
RUN apt install python3.6 python3.6-dev curl wget git nano unzip -y
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6 - --user

RUN echo 'alias pip="/root/.local/bin/pip3.6"' >> ~root/.bashrc
RUN echo 'alias python="python3.6"' >> ~root/.bashrc

RUN /root/.local/bin/pip3.6 install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN /root/.local/bin/pip3.6 install pillow scipy==1.1.0

RUN mkdir /home/DAIN
COPY . /home/DAIN

RUN cd /home/DAIN/my_package && ./build.sh
RUN cd /home/DAIN/PWCNet/correlation_package_pytorch1_0 && ./build.sh

RUN mkdir /home/DAIN/model_weights
RUN cd /home/DAIN/model_weights && wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth

RUN mkdir /home/DAIN/sharedfs

CMD cd /home/DAIN/ && CUDA_VISIBLE_DEVICES=0 python3.6 infer.py

#------------------------ testing -------------------------

#mkdir MiddleBurySet

#cd ../MiddleBurySet
#wget http://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip
#unzip other-color-allframes.zip
#wget http://vision.middlebury.edu/flow/data/comp/zip/other-gt-interp.zip
#unzip other-gt-interp.zip
#cd ..
#
#

#    '-gencode', 'arch=compute_75,code=sm_75',
#    '-gencode', 'arch=compute_75,code=compute_75'
