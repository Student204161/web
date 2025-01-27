
#before running shell script, make sure to be in h100sh queue for support of updated glibc

#cd $BLACKHOLE/camproj 

#module load python3/3.10.12

#python3.10 -m venv env_works
#source env_works/bin/activate

# conda create -n webv python=3.10
# conda activate webv
base_direc="$PWD"
#install packages #we need torch 2.1.0 since prebuilt linux binaries for pytorch3d is limited.
#module load cuda/11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip3 install open3d
pip install iopath #dependency for pytorch3d
pip install fvcore #dependency for pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
pip3 install plotly rich plyfile jupyterlab nodejs ipywidgets
pip3 install --upgrade PyMCubes
pip install tensorboard
## for MVG code
pip install pyrender
pip install pymeshfix
pip install trimesh
pip install opencv-python


# #cuDNN
# cd ~/Downloads
# tar -xvJf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
# mv cudnn-linux-x86_64-8.7.0.84_cuda11-archive cudnn

# sudo cp cudnn/include/cudnn*.h /usr/local/cuda-11.8/include
# sudo cp -P cudnn/lib/libcudnn* /usr/local/cuda-11.8/lib64
# sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
# sudo apt update

#dependencies for sugar/3DGS
# git clone https://github.com/Anttwo/SuGaR.git --recursive
# cd SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization/
# pip install -e .
# cd ../simple-knn/
# pip install -e .
# cd ../../../../

#XMEM
cd ..
git clone https://github.com/hkchengrex/XMem.git --recursive
cd XMem
pip install -r requirements.txt
mkdir saves
cd saves
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
cd ..
cd ..


# #COLMAP 
# sudo apt update
# sudo apt install colmap

# #GLOMAP (BETTER THAN COLMAP)
# sudo apt-get install \
#     git \
#     cmake \
#     ninja-build \
#     build-essential \
#     libboost-program-options-dev \
#     libboost-graph-dev \
#     libboost-system-dev \
#     libeigen3-dev \
#     libflann-dev \
#     libfreeimage-dev \
#     libmetis-dev \
#     libgoogle-glog-dev \
#     libgtest-dev \
#     libgmock-dev \
#     libsqlite3-dev \
#     libglew-dev \
#     qtbase5-dev \
#     libqt5opengl5-dev \
#     libcgal-dev \
#     libceres-dev

# git clone https://github.com/colmap/glomap.git --recursive

# cd gaussian-splatting/submodules/diff-gaussian-rasterization/
# pip install -e .
# cd ../simple-knn/
# pip install -e .
# cd ../../../

