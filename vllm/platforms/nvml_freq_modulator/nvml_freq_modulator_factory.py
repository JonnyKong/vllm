# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from vllm.config import VllmConfig
from vllm.platforms.nvml_utils import get_gpu_name, get_preselected_freq

from .dqn_nvml_freq_modulator import DQNNvmlFreqModulator
from .mp_freq_modulator import MPNvmlFreqModulatorClient
from .nvml_freq_modulator import NvmlFreqModulatorInterface
from .q_table_nvml_freq_modulator import QTableFreqModulator
from .rule_based_freq_modulator import RuleBasedNvmlFreqModulator
from .value_iter_freq_modulator import ValueIterationNvmlFreqModulator


def nvml_freq_modulator_factory(config: VllmConfig,
                                llm_engine) -> NvmlFreqModulatorInterface:
    '''
    Factory method to create an NvmlFreqModulator instance from a
    VllmConfig. Currently, always returns a RuleBasedNvmlFreqModulator.
    '''
    interval_s = 1.0
    freq_choices = get_preselected_freq(get_gpu_name())

    if config.freq_mod_mode == 'rule':
        return RuleBasedNvmlFreqModulator(llm_engine, interval_s=interval_s)
    elif config.freq_mod_mode == 'value-iter':
        assert config.log_dir
        log_file = str(Path(config.log_dir) / 'value_iter.csv')
        return ValueIterationNvmlFreqModulator(llm_engine,
                                               interval_s=interval_s,
                                               freq_choices=freq_choices,
                                               log_file=log_file)
    elif config.freq_mod_mode == 'q-learn':
        assert config.log_dir
        return QTableFreqModulator(
            llm_engine,
            interval_s=interval_s,
            freq_choices=freq_choices,
            power_usage_queue=llm_engine.power_usage_queue,
            log_dir=config.log_dir,
            save_rl_history=True,
            pretrained_rl_model_path=config.pretrained_rl_model_path,
            gpu_tdp=300,
            epsilon=0.01,
            tbt_slo=0.250)
    elif config.freq_mod_mode == 'dqn':
        assert config.log_dir
        return DQNNvmlFreqModulator(
            llm_engine,
            interval_s=interval_s,
            freq_choices=freq_choices,
            power_usage_queue=llm_engine.power_usage_queue,
            log_dir=config.log_dir,
            save_rl_history=True,
            pretrained_rl_model_path=config.pretrained_rl_model_path,
            gpu_tdp=300,
            tbt_slo=0.250)
    elif config.freq_mod_mode == 'mp':
        return MPNvmlFreqModulatorClient(
            llm_engine,
            config,
            freq_choices=freq_choices,
            log_dir=Path(config.log_dir),
            tbt_sla=0.22,
            optim_target='power',
            mod_interval=1,
        )
    else:
        raise NotImplementedError(
            f'Unrecognized freq_mod_mode: {llm_engine.freq_mod_mode}')
