#!/bin/bash

PORT=8002
NUM_PROMPTS=2000
MODEL_NAME_HF=microsoft/phi-2

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | awk '{print $NF}')
MODEL_NAME_SHORT="${MODEL_NAME_HF#*/}" # Strip the org or creator

ADDITIONAL_VLLM_ARGS=""
if [[ ${GPU} == "A40" && ${MODEL_NAME_HF} == "meta-llama/Llama-3.1-8B-Instruct" ]]; then
    ADDITIONAL_VLLM_ARGS+=" --max-model-len 65536 --max-num-seqs 1024 --max-num-batched-tokens 1024"
elif [[ ${GPU} == "T4" && ${MODEL_NAME_HF} == "microsoft/phi-2" ]]; then
    ADDITIONAL_VLLM_ARGS+=" --max-model-len 2048 --dtype=half"
else
    echo "GPU-model combo not found"
    exit 1
fi

wait_for_server() {
    # wait for vllm server to start
    # return 1 if vllm server crashes
    local port=$1
    timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

for qps in 2; do
    for freq in 1590; do
        nvidia-smi -i ${CUDA_VISIBLE_DEVICES} -lgc ${freq}

        # Start in new process group
        VLLM_CMD=(setsid vllm serve ${MODEL_NAME_HF} --port ${PORT}
            --collect-detailed-traces "worker,power" -tp 1 -pp 1
            --disable-async-output-proc --disable-frontend-multiprocessing
            --log-dir ~/energy_efficient_serving_results/request_timing/2025-05-03_profile-borderline-qps/${GPU}_${MODEL_NAME_SHORT}_qps${qps}_reqs${NUM_PROMPTS}_fixed${freq}
            --disable-python-gc ${ADDITIONAL_VLLM_ARGS})
        # Print the command
        echo "${VLLM_CMD[@]}"

        # Run the command
        "${VLLM_CMD[@]}" &
        VLLM_PID=$!
        wait_for_server "${PORT}"

        python benchmark_serving.py --model ${MODEL_NAME_HF} --dataset-name trace \
            --ignore-eos --max-concurrency 512 --port ${PORT} \
            --dataset-path ~/energy_efficient_serving_results/datasets/processed/azure_2024_code_sharegpt-ctx-len_qps${qps}.0_req-cnt20000.csv \
            --num-prompts ${NUM_PROMPTS}

        kill -- -"$VLLM_PID"
        nvidia-smi -i ${CUDA_VISIBLE_DEVICES} -rgc
    done
done
