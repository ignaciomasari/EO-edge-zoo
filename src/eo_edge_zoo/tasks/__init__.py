"""Task = dataset (load + bands) + splits + head size + metrics. One registry entry per task."""
from .base import ClassificationTask, Task
from .eurosat import EuroSAT

TASKS: dict[str, Task] = {task.name: task for task in (EuroSAT(),)}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"task={name!r} not in {sorted(TASKS)}")
    return TASKS[name]


__all__ = ["TASKS", "ClassificationTask", "Task", "get_task"]
