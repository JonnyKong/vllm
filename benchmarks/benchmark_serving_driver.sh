#!/bin/bash

FREQS=(1740 1590 1440 1275)
export CUDA_VISIBLE_DEVICES=1
PORT=8002
NUM_PROMPTS=20000

wait_for_server() {
    # wait for vllm server to start
    # return 1 if vllm server crashes
    local port=$1
    timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

for freq in ${FREQS[@]}; do
    nvidia-smi -i ${CUDA_VISIBLE_DEVICES} -lgc ${freq}

    # Start in new process group
    setsid vllm serve --port ${PORT} \
        meta-llama/Llama-3.1-8B-Instruct \
        --collect-detailed-traces "worker,power" -tp 1 -pp 1 \
        --max-model-len 65536 --disable-async-output-proc --disable-frontend-multiprocessing \
        --max-num-seqs 1024 --max-num-batched-tokens 1024 \
        --log-dir /export2/kong102/energy_efficient_serving_results/request_timing/2025-04-17_dp/a100_qps9_reqs${NUM_PROMPTS}_fixed${freq} \
        --disable-python-gc &
    VLLM_PID=$!
    wait_for_server ${PORT}

    python benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --dataset-name trace \
        --ignore-eos --max-concurrency 512 \
        --port ${PORT} \
        --dataset-path /export2/kong102/energy_efficient_serving_results/datasets/processed/azure_2024_code_sharegpt-ctx-len_qps9.0_req-cnt20000.csv \
        --num-prompts ${NUM_PROMPTS}

    kill -- -"$VLLM_PID"
    nvidia-smi -i ${CUDA_VISIBLE_DEVICES} -rgc
done
