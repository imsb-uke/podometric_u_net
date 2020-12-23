FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter
# --- Install any needed packages specified in requirements.apt
COPY . /
RUN apt-get update && xargs apt-get install -y --no-install-recommends < /requirements.apt

# --- Install any needed packages specified in requirements.pip
#RUN pip install -r /requirements.pip --upgrade
RUN pip install -r /requirements-long.pip

# activate jupyter extensions
RUN jupyter contrib nbextension install \
  && jupyter nbextension enable codefolding/main \
  && jupyter nbextension enable collapsible_headings/main


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility
