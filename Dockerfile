FROM python:3.11.5

WORKDIR /app

COPY . /app

RUN apt-get update;\
    apt-get -y upgrade;\
    apt-get -y dist-upgrade;\
    apt-get install -y python3-opencv

RUN pip install --upgrade pip

# detectron2 requires torch be installed first. The requirements.txt is sorted alphabetically, hence the pip install torch. Otherwise, it will fail.
RUN pip install torch==2.0.1;\
    pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT  ["python", "-m", "src.cmd.server"]