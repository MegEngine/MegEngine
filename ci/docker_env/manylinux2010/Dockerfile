FROM quay.io/pypa/manylinux2010_x86_64:2020-01-31-046f791

ENV UID=1024 \
    PATH=${PATH}:/usr/local/cuda/bin \
    LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/opt/cudnn/lib64:/opt/tensorrt/lib \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/opt/cudnn/lib64:/opt/tensorrt/lib \
    CPATH=${CPATH}:/usr/local/cuda/include:/opt/cudnn/include:/opt/tensorrt/include

ARG platform
COPY setup_mirror.sh .
RUN ./setup_mirror.sh "$platform"

ADD init_image.sh /tmp
RUN /tmp/init_image.sh && rm -f /tmp/init_image.sh

