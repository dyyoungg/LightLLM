#!/bin/bash

echo "=== GPU内存清理工具 ==="
echo "当前GPU状态:"
nvidia-smi

echo -e "\n正在查找当前用户的GPU进程..."

# 方法1: 查找当前用户的GPU进程
USER_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq)

if [ -z "$USER_PIDS" ]; then
    echo "没有找到GPU进程"
    exit 0
fi

echo "找到以下GPU进程:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

read -p "是否要终止这些进程？(y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for PID in $USER_PIDS; do
        if [ -n "$PID" ] && [ "$PID" -eq "$PID" ] 2>/dev/null; then
            echo "终止进程: $PID"
            kill -9 "$PID" 2>/dev/null || true
        fi
    done
    echo "进程已终止"
    
    # 等待并重新检查
    sleep 2
    echo -e "\n清理后的GPU状态:"
    nvidia-smi
else
    echo "操作已取消"
fi