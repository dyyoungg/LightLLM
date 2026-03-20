mkdir -p "/mnt/afs/yangdeyu/$HOSTNAME"
nohup bash /mnt/afs/yangdeyu/dependency/lightllm-dev/launch_fp8.sh >/mnt/afs/yangdeyu/$HOSTNAME/llm.log 2>&1 &
apt-get update && apt-get install -y netcat
PORT=18003
HOST="localhost"
MAX_ATTEMPTS=60
SLEEP_TIME=8
echo "开始检查服务是否启动（端口: $PORT）..."

# 使用 POSIX 兼容的 for 循环
i=1
while [ $i -le $MAX_ATTEMPTS ]; do
    echo "尝试 $i/$MAX_ATTEMPTS: 检查端口 $PORT..."
    
    # 使用nc检查端口
    if nc -z $HOST $PORT; then
        echo "✓ 服务已成功启动！端口 $PORT 已开放"
        echo "LLM服务启动完成，可以开始使用"

        /usr/bin/env /opt/conda/envs/llava/bin/python /mnt/afs/yangdeyu/api_test.py # 预热
        
        # LightLLM 启动成功后，再启动 Go 服务
        mkdir -p /opt/user.core.svc/config/
        cp -r /mnt/afs/haoye/build/user.core.svc/manifest/alpha_baidu_config/* /opt/user.core.svc/config/
        cp -r /mnt/afs/share/Qwen2-VL-72B-Instruct /opt/user.core.svc
        cd /mnt/afs/haoye/build/user.core.svc
        /usr/local/go/bin/go build -o /opt/user.core.svc/bin/user.core.svc /mnt/afs/haoye/build/user.core.svc/main.go
        service supervisor start
        echo "Go 服务启动完成"
        sleep 999d
      
    else
        echo "✗ 端口 $PORT 尚未开放，等待 ${SLEEP_TIME}秒后重试..."
        sleep $SLEEP_TIME
    fi
    i=$((i + 1))
done
