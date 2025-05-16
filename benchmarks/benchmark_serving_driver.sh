#!/bin/bash

PORT=8002
# RESULT_ROOT=/export2/kong102/energy_efficient_serving_results
RESULT_ROOT=${HOME}/energy_efficient_serving_results

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | sed 's/^NVIDIA //' | sed 's/^TESLA//' | sed 's/ /-/g')

wait_for_server() {
    # wait for vllm server to start
    # return 1 if vllm server crashes
    local port=$1
    timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

get_additional_vllm_args() {
    MODEL_NAME_HF=${1}

    # Keep it same with `benchmark_batch_driver.py`
    ADDITIONAL_VLLM_ARGS=""
    if [[ ${GPU} == "A40" && ${MODEL_NAME_HF} == "meta-llama/Llama-3.1-8B-Instruct" ]]; then
        ADDITIONAL_VLLM_ARGS+=" --max-model-len 65536 --max-num-seqs 1024 --max-num-batched-tokens 1024"
    elif [[ ${GPU} == "T4" && ${MODEL_NAME_HF} == "microsoft/phi-2" ]]; then
        # Model max len is 2048
        # T4 does not have bf16, need to use fp16
        ADDITIONAL_VLLM_ARGS+=" --max-model-len 2048 --dtype=half"
    elif [[ ${GPU} == "A100-SXM4-80GB" && ${MODEL_NAME_HF} == "google/gemma-2-27b-it" ]]; then
        # Model max len is 8192
        ADDITIONAL_VLLM_ARGS+=" --max-model-len 8192 --max-num-seqs 1024 --max-num-batched-tokens 1024"
    elif [[ ${GPU} == "H100-80GB-HBM3" && ${MODEL_NAME_HF} == "google/gemma-2-27b-it" ]]; then
        ADDITIONAL_VLLM_ARGS+=" --max-model-len 8192 --max-num-seqs 1024 --max-num-batched-tokens 1024"
    elif [[ ${GPU} == "A100-SXM4-80GB" && ${MODEL_NAME_HF} == "meta-llama/Llama-3.1-70B-Instruct" ]]; then
        ADDITIONAL_VLLM_ARGS+=" -tp 4 --max-model-len 8192 --max-num-seqs 1024 --max-num-batched-tokens 1024"
    else
        echo "GPU-model combo not found"
        exit 1
    fi
    echo ${ADDITIONAL_VLLM_ARGS}
}

run() {
    MODEL_NAME_HF=${1}
    QPS=${2}
    FREQ=${3}
    LOG_DIR=${4}
    DATASET_PATH=${5}
    NUM_PROMPTS=${6}

    nvidia-smi -i ${CUDA_VISIBLE_DEVICES} -lgc ${FREQ}

    ADDITIONAL_VLLM_ARGS=$(get_additional_vllm_args ${MODEL_NAME_HF})

    # Start in new process group
    VLLM_CMD=(setsid vllm serve ${MODEL_NAME_HF} --port ${PORT}
        --collect-detailed-traces "worker,power" -tp 1 -pp 1
        --disable-async-output-proc --disable-frontend-multiprocessing
        --log-dir ${LOG_DIR}
        --disable-python-gc --enable-chunked-prefill ${ADDITIONAL_VLLM_ARGS})
    # Print the command
    echo "${VLLM_CMD[@]}"

    # Run the command
    "${VLLM_CMD[@]}" &
    VLLM_PID=$!
    wait_for_server "${PORT}"

    python benchmark_serving.py --model ${MODEL_NAME_HF} --dataset-name trace \
        --ignore-eos --max-concurrency 512 --port ${PORT} \
        --dataset-path ${DATASET_PATH} \
        --num-prompts ${NUM_PROMPTS}

    kill -- -"$VLLM_PID"
    nvidia-smi -i ${CUDA_VISIBLE_DEVICES} -rgc
}

profile_batch_shapes() {
    MODEL_NAME_HF=meta-llama/Llama-3.1-8B-Instruct
    FREQ=1740
    NUM_PROMPTS=20000

    MODEL_NAME_SHORT="${MODEL_NAME_HF#*/}" # Strip the org or creator
    for qps in 9.0 5.0; do
        DATASET_PATH=${RESULT_ROOT}/datasets/processed/azure_2024_code_sharegpt-ctx-len_qps${qps}_req-cnt20000.csv
        LOG_DIR=${RESULT_ROOT}/request_timing/2025-05-05_batch-shape-profiling/${GPU}_${MODEL_NAME_SHORT}_qps${qps}_reqs${NUM_PROMPTS}_fixed${FREQ}
        run ${MODEL_NAME_HF} ${qps} ${FREQ} ${LOG_DIR} ${DATASET_PATH} ${NUM_PROMPTS}
    done
}

profile_borderline_qps() {
    MODEL_NAME_HF=google/gemma-2-27b-it
    FREQ=1980
    NUM_PROMPTS=2000

    MODEL_NAME_SHORT="${MODEL_NAME_HF#*/}" # Strip the org or creator
    for qps in 6 8; do
        DATASET_PATH=${RESULT_ROOT}/datasets/processed/azure_2024_code_sharegpt-ctx-len_qps${qps}.0_req-cnt20000.csv
        LOG_DIR=${RESULT_ROOT}/request_timing/2025-05-05_borderline-qps-profiling/${GPU}_${MODEL_NAME_SHORT}_qps${qps}_reqs${NUM_PROMPTS}_fixed${FREQ}
        run ${MODEL_NAME_HF} ${qps} ${FREQ} ${LOG_DIR} ${DATASET_PATH} ${NUM_PROMPTS}
    done
}

profile_simuluate_autoscaling() {
    MODEL_NAME_HF=meta-llama/Llama-3.1-8B-Instruct
    FREQ=1740
    NUM_PROMPTS=6000

    MODEL_NAME_SHORT="${MODEL_NAME_HF#*/}" # Strip the org or creator
    for qps in 9.0 9.7; do
        DATASET_PATH=${RESULT_ROOT}/datasets/processed/azure_2024_code_sharegpt-ctx-len_qps${qps}_req-cnt20000_prob-subsampling.csv
        LOG_DIR=${RESULT_ROOT}/request_timing/2025-05-13_simulate-autoscaling/${GPU}_${MODEL_NAME_SHORT}_qps${qps}_reqs${NUM_PROMPTS}_fixed${FREQ}
        run ${MODEL_NAME_HF} ${qps} ${FREQ} ${LOG_DIR} ${DATASET_PATH} ${NUM_PROMPTS}
    done
}

# profile_batch_shapes
# profile_borderline_qps
profile_simuluate_autoscaling
