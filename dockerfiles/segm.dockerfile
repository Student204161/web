FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN export GOOGLE_APPLICATION_CREDENTIALS="webproj-447013-a1db168cc4ae.json"

# Install dependencies

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git cmake ninja-build libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    && \ 
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ src/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN git clone https://github.com/hkchengrex/XMem.git --recursive
RUN cd XMem
RUN mkdir saves
RUN cd saves
RUN wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
RUN cd ..
RUN cd ..

RUN git clone https://github.com/facebookresearch/sam2.git && cd sam2
RUN pip install -e .
RUN pip install -e ".[notebooks]"

RUN cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
RUN export GOOGLE_APPLICATION_CREDENTIALS="webproj-447013-a1db168cc4ae.json"
#RUN wandb login a570414ad58306f2a605a3ec03f3396900256e59
# RUN ls data/
# RUN ls data/processed/

ENTRYPOINT ["python", "-u", "src/app/run1.py"]
