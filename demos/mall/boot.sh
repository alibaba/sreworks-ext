#!/bin/sh
# description: Auto-starts boot
# $API_HOME/bin/boot.sh
#启动命令： boot.sh start
#重启命令： boot.sh restart
#关闭命令： boot.sh stop
#是否运行： boot.sh status

CURRENT_DIR=`dirname $0`
API_HOME=`cd "$CURRENT_DIR/.." >/dev/null; pwd`
# 应用名
Tag="mall"
cd $API_HOME
# 要执行的jar包
Jar="$API_HOME/mall.jar"
RETVAL="0"

# See how we were called.
start()
{
    # SkyWalking ENV 配置
    export SW_AGENT_NAMESPACE=`cat /run/secrets/kubernetes.io/serviceaccount/namespace`
    export SW_AGENT_NAME=${APP_INSTANCE_ID}
    export SW_AGENT_COLLECTOR_BACKEND_SERVICES=skywalking-oap.sreworks-client:11800
    export JAVA_AGENT=-javaagent:/data/mall/code/skywalking-agent/skywalking-agent.jar

    # Log GRPC Export ENV
    export SW_GRPC_LOG_SERVER_HOST=skywalking-oap.sreworks-client
    export SW_GRPC_LOG_SERVER_PORT=11800

    echo $$ > $API_HOME/api.pid
    java \
    -XX:-UseGCOverheadLimit \
    -verbose:gc -Xloggc:jvm-gc.log \
    -Dappliction=$Tag \
    $JAVA_AGENT \
    -jar $Jar --spring.config.location=$API_HOME/config
}


stop()
{
    pid=$(ps -ef | grep -v 'grep' | egrep "$Jar"| awk '{printf $2 " "}')
    if [ "$pid" != "" ]; then
        echo -n "$Tag ( pid: $pid) is running"
        echo
        echo -n "Shutting down $Tag..."
        echo 
        kill -9 "$pid" > /dev/null 2>&1
    fi
        status

}

status()
{
    pid=$(ps -ef | grep -v 'grep' | egrep "$Jar"| awk '{printf $2 " "}')
    #echo "$pid"
    if [ "$pid" != "" ]; then
        echo "$Tag is running,pid is $pid"
    else
        echo "$Tag is stopped"
    fi
}



usage()
{
   echo "Usage: $0 {start|stop|restart|status}"
   RETVAL="2"
}

# See how we were called.
RETVAL="0"
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    reload)
        RETVAL="3"
        ;;
    status)
        status
        ;;
    *)
      usage
      ;;
esac

exit $RETVAL
