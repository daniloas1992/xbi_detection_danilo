FROM tensorflow/tensorflow:latest-gpu

USER root

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ADD . /app
WORKDIR /app
EXPOSE 3000

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y python3-dev \
				         g++ \
				         libjpeg-dev \
				         libopenblas-dev \
				         liblapack-dev \
				         build-essential \
				         tzdata


RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install np
RUN pip install scipy
RUN pip install Cython
RUN pip install scikit-learn
RUN pip install sklearn
RUN pip install Keras
RUN pip install silence_tensorflow
RUN pip install nltk
RUN pip install pandas

RUN apt-get install -y zlib1g-dev
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install liac-arff
RUN pip install imbalanced-learn
RUN pip install Pillow

CMD ["sh"]
