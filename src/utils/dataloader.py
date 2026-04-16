from src.tasks.humaneval import HumanEvalTask
from src.tasks.mbpp import MBPPTask
from src.tasks.bigcode import BigCodeTask

from src.adapters.humaneval import HumanEvalAdapter
from src.adapters.mbpp import MBPPAdapter
from src.adapters.bigcode import BigCodeAdapter


def load_task_and_adapter(dataset_name: str):
    """
    dataset 이름에 따라 task loader와 adapter를 함께 반환
    """
    if dataset_name == "humaneval":
        return HumanEvalTask(), HumanEvalAdapter()

    if dataset_name == "mbpp":
        return MBPPTask(), MBPPAdapter()
    
    if dataset_name == "bigcode":
        return BigCodeTask(), BigCodeAdapter()

    raise ValueError(f"Unsupported dataset: {dataset_name}")