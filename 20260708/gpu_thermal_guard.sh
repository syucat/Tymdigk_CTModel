#!/bin/bash
# GPU温度が80度以上になったら学習プロセスを一時停止(SIGSTOP)し、
# 75度まで下がったら再開(SIGCONT)する監視スクリプト。
# プロセスをkillするのではなく一時停止するだけなので、学習の進捗は失われない。

PATTERN="python3 main.py"
PAUSE_TEMP=80
RESUME_TEMP=75
CHECK_INTERVAL=10

paused=0
last_seen=1
while true; do
    pid=$(pgrep -f "$PATTERN" | head -1)
    if [ -z "$pid" ]; then
        if [ "$last_seen" -eq 1 ]; then
            echo "$(date +%T) 学習プロセスが見つかりません。再起動されるまで待機します。"
            last_seen=0
        fi
        paused=0
        sleep "$CHECK_INTERVAL"
        continue
    fi
    if [ "$last_seen" -eq 0 ]; then
        echo "$(date +%T) 学習プロセスを検知しました (PID $pid)。監視を再開します。"
        last_seen=1
    fi

    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)

    if [ "$paused" -eq 0 ] && [ "$temp" -ge "$PAUSE_TEMP" ]; then
        kill -STOP "$pid"
        paused=1
        echo "$(date +%T) GPU ${temp}C >= ${PAUSE_TEMP}C: 学習を一時停止 (PID $pid)"
    elif [ "$paused" -eq 1 ] && [ "$temp" -le "$RESUME_TEMP" ]; then
        kill -CONT "$pid"
        paused=0
        echo "$(date +%T) GPU ${temp}C <= ${RESUME_TEMP}C: 学習を再開 (PID $pid)"
    fi

    sleep "$CHECK_INTERVAL"
done
