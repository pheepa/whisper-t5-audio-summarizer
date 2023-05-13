FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip3 install \
     torch==1.13.1+cpu \
     torchaudio==0.13.1+cpu \
     -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get update && apt-get install -y ffmpeg

COPY main.py /code/
COPY whisper_timestamped /code/
