# 基于nginx官方镜像构建Dockerfile
FROM nginx

# 将本地的index.html复制到容器中
COPY index.html /usr/share/nginx/html/index.html

# 用本地的nginx.conf文件替换默认的配置文件
COPY nginx.conf /etc/nginx/nginx.conf
