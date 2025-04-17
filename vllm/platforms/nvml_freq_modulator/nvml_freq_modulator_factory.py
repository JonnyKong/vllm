# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from vllm.config import VllmConfig

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
    if config.freq_mod_mode == 'rule':
        return RuleBasedNvmlFreqModulator(llm_engine, interval_s=interval_s)
    elif config.freq_mod_mode == 'value-iter':
        assert config.log_dir
        a40_freq_choices = [210, 510, 825, 1125, 1440, 1740]
        log_file = str(Path(config.log_dir) / 'value_iter.csv')
        return ValueIterationNvmlFreqModulator(llm_engine,
                                               interval_s=interval_s,
                                               freq_choices=a40_freq_choices,
                                               log_file=log_file)
    elif config.freq_mod_mode == 'q-learn':
        assert config.log_dir
        a40_freq_choices = [
            210,
            360,
            510,
            675,
            825,
            975,
            1125,
            1275,
            1440,
            1590,
            1740,
        ]
        return QTableFreqModulator(
            llm_engine,
            interval_s=interval_s,
            freq_choices=a40_freq_choices,
            power_usage_queue=llm_engine.power_usage_queue,
            log_dir=config.log_dir,
            save_rl_history=True,
            pretrained_rl_model_path=config.pretrained_rl_model_path,
            gpu_tdp=300,
            epsilon=0.01,
            tbt_slo=0.250)
    elif config.freq_mod_mode == 'dqn':
        assert config.log_dir
        a40_freq_choices = [
            210,
            360,
            510,
            675,
            825,
            975,
            1125,
            1275,
            1440,
            1590,
            1740,
        ]
        return DQNNvmlFreqModulator(
            llm_engine,
            interval_s=interval_s,
            freq_choices=a40_freq_choices,
            power_usage_queue=llm_engine.power_usage_queue,
            log_dir=config.log_dir,
            save_rl_history=True,
            pretrained_rl_model_path=config.pretrained_rl_model_path,
            gpu_tdp=300,
            tbt_slo=0.250)
    elif config.freq_mod_mode == 'mp':
        a40_freq_choices = [
            210,
            360,
            510,
            675,
            825,
            975,
            1125,
            1275,
            1440,
            1590,
            1740,
        ]
        return MPNvmlFreqModulatorClient(
            llm_engine,
            freq_choices=a40_freq_choices,
        )
    else:
        raise NotImplementedError(
            f'Unrecognized freq_mod_mode: {llm_engine.freq_mod_mode}')
