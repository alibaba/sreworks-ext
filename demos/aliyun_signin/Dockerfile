FROM python:3.11-alpine
COPY . /app
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
RUN apk update \
   && pip install -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -r /app/requirements.txt
WORKDIR /app
ENTRYPOINT ["python", "/app/server.py"]
