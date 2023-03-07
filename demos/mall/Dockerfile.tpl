FROM registry.cn-hangzhou.aliyuncs.com/alisre/mall_base

COPY settings.xml /root/.m2/settings.xml

ARG JAR_FILE=mall.jar
USER root
# 新建应用目录
ARG HOME=/data/mall
RUN mkdir -p $HOME/config;mkdir $HOME/log;mkdir $HOME/bin;mkdir $HOME/h2db;mkdir $HOME/file;mkdir $HOME/code;ls -la $HOME
# build jar
ADD ./ $HOME/code/
WORKDIR $HOME/code
RUN set -eux;ls -la;mvn clean install -DskipTests && cp $HOME/code/target/$JAR_FILE $HOME  && \
    cp $HOME/code/file/* ../file/ && cp $HOME/code/h2db/* ../h2db/ && cp $HOME/code/boot.sh ../bin/

# 启动脚本
WORKDIR $HOME/bin
ENTRYPOINT sh boot.sh start

# 端口
EXPOSE 8081
