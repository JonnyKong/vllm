import os
import signal
import subprocess
import time
from pathlib import Path

from benchmark_utils import (get_gpu_name, get_result_root,
                             uniform_sample_sorted)

from vllm.logger import init_logger
from vllm.platforms.nvml_utils import nvml_get_available_freq, nvml_lock_freq

logger = init_logger(__name__)

SERVER_PORT = 8000
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'


def start_vllm_server(log_dir: Path):
    """Starts the vLLM server in a subprocess."""
    command = (f'vllm serve {MODEL} --port {SERVER_PORT} '
               f'--log-dir {log_dir} '
               '--collect-detailed-traces "worker,power" '
               '-tp 1 -pp 1 --max-model-len 65536')

    logger.info('Starting vLLM server...')
    vllm_process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    return vllm_process


def wait_for_server(port):
    """Waits for the vLLM server to be ready, returns False if it fails."""
    start_time = time.time()
    while time.time() - start_time < 120:
        try:
            result = subprocess.run(
                f'curl -s http://localhost:{port}/v1/completions', shell=True)
            if result.returncode == 0:
                logger.info('vLLM server is ready')
                return True
        except Exception as e:
            logger.warning('Waiting for server... %s', e)

        time.sleep(1)

    logger.error('Timeout: vLLM server did not start in time')
    return False


def run_benchmark_script(req_rate: float):
    logger.info('Running benchmark script...')
    command = ('python3 benchmark_serving.py '
               f'--model {MODEL} '
               f'--port {SERVER_PORT} '
               '--dataset-name sharegpt '
               '--dataset-path ShareGPT_V3_unfiltered_cleaned_split.json '
               '--num-prompts 3000 '
               f'--request-rate {req_rate} --burstiness 0.5')
    subprocess.run(command, shell=True)


def kill_server(process):
    """Kills the vLLM server process."""
    logger.info('Stopping vLLM server...')
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def run_once(log_dir: Path, req_rate: float):
    vllm_process = None

    try:
        vllm_process = start_vllm_server(log_dir)

        if not wait_for_server(SERVER_PORT):
            logger.error('Exiting due to server startup failure')
            return
        run_benchmark_script(req_rate)

    finally:
        if vllm_process:
            kill_server(vllm_process)
            logger.info('vLLM server stopped')


def main():
    expr_root = (get_result_root() / 'request_timing' /
                 '2025-02-08_benchmark-serving-varying-freq_test')
    freq_arr = uniform_sample_sorted(nvml_get_available_freq(), 6)
    for req_rate in [5, 8]:
        for freq in freq_arr:
            log_dir = (expr_root /
                       f'{get_gpu_name()}_req-rate{req_rate}_freq{freq}')
            with nvml_lock_freq(freq):
                run_once(log_dir, req_rate)


if __name__ == '__main__':
    main()
