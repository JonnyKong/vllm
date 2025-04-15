# SPDX-License-Identifier: Apache-2.0
from multiprocessing import Process, Queue
from typing import Optional

import msgspec

from vllm.engine.metrics_types import Stats
from vllm.logger import init_logger

from .nvml_freq_modulator import NvmlFreqModulatorInterface

logger = init_logger(__name__)


class FreqModMsg(msgspec.Struct):
    """
    Msg from client to server.
    """
    num_prefills: int


class MPNvmlFreqModulatorClient(NvmlFreqModulatorInterface):
    """
    Adjusts frequency in a separate process. Useful if the procedure of
    determining the frequency is computation heavy.
    """

    def __init__(
        self,
        llm_engine,
        freq_choices: list[int],
    ):
        self.llm_engine = llm_engine

        self.encoder = msgspec.msgpack.Encoder()

        self.q: Queue = Queue()
        self.server = _MPNvmlFreqModulatorServer(freq_choices, self.q)
        self.server_process = Process(target=self.server.run, daemon=True)
        self.server_process.start()

    def step(self, stats: Optional[Stats]) -> None:
        msg = self.build_msg()
        msg_encoded = self.encoder.encode(msg)
        self.q.put(msg_encoded)

    @staticmethod
    def build_msg() -> FreqModMsg:
        return FreqModMsg(num_prefills=0, )


class _MPNvmlFreqModulatorServer:

    def __init__(
        self,
        freq_choices: list[int],
        q: Queue,
    ):
        self.freq_choices = freq_choices
        self.q = q

        self.decoder = msgspec.msgpack.Decoder()

    def run(self):
        while True:
            msg: FreqModMsg = self.decoder.decode(self.q.get())
            print('msg: ', msg)
