FROM tensorflow/tensorflow:1.13.1-py3-jupyter

# --- Install any needed packages specified in requirements.apt
COPY . /
RUN apt-get update && xargs apt-get install -y --no-install-recommends < /requirements.apt

# --- Install any needed packages specified in requirements.pip
RUN pip install -U pip
#RUN pip install -r /requirements.pip --upgrade
RUN pip install -r /requirements-long.pip

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility
