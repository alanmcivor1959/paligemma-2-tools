FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

RUN pip install --break-system-packages opencv-python-headless
RUN pip install --break-system-packages transformers
RUN pip install --break-system-packages accelerate
