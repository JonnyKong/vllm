# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any


class IdleTimeInjector(ABC):
    '''
    Base class for an idle time injector. It injects 
    idle time in between batches to reduce 
    energy consumption.
    '''

    def __init__(self, llm_engine) -> None:
        self.llm_engine = llm_engine

    @abstractmethod
    def get_idle_time(self) -> float:
        raise NotImplementedError

    @staticmethod
    def create_from_config(config: Any, llm_engine: Any) -> 'IdleTimeInjector':
        '''
        Factory method to create an IdleTimeInjector instance from a
        VllmConfig. Currently, always returns a ConstIdleTimeInjector.
        '''
        return ConstIdleTimeInjector(llm_engine)


class ConstIdleTimeInjector(IdleTimeInjector):
    '''
    An implementation of IdleTimeInjector that 
    injects a constant amount of time
    in between each batch.
    '''

    def __init__(self, llm_engine: Any) -> None:
        super().__init__(llm_engine)

    def get_idle_time(self) -> float:
        return 0.1
