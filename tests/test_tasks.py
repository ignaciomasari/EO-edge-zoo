import pytest

from eo_edge_zoo.tasks import get_task


# --- registry
def test_eurosat_task_metadata():
    task = get_task("eurosat")
    assert (task.name, task.num_classes, task.splits) == ("eurosat", 10, ("train", "val", "test"))
    # cached x in SSL4EO-S12 band order: B8A right after B08
    assert task.bands == ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


def test_unknown_task_raises_with_known_names():
    with pytest.raises(ValueError, match="eurosa.*eurosat"):
        get_task("eurosa")


# --- input bands per init
FILE_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A")  # B8A last


def other_order_task():
    from dataclasses import dataclass

    from eo_edge_zoo.tasks import ClassificationTask

    @dataclass(frozen=True)
    class FileOrderTask(ClassificationTask):
        def load(self, root, num_workers=4):
            raise NotImplementedError
    return FileOrderTask(name="file_order", num_classes=3, bands=FILE_ORDER)


def test_eurosat_input_bands_per_init():
    task = get_task("eurosat")
    assert task.input_idx("imagenet_rgb") == [3, 2, 1]  # B04, B03, B02
    assert task.input_idx("imagenet_13ch") == list(range(13))
    assert task.input_idx("s2_ssl") == list(range(13))  # already SSL4EO order
    assert [task.in_chans(init) for init in ("imagenet_rgb", "imagenet_13ch", "s2_ssl")] == [3, 13, 13]


def test_s2_ssl_reorders_to_ssl4eo_band_order():
    task = other_order_task()
    assert task.input_idx("s2_ssl") == [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11]  # B8A moved after B08
    assert task.input_idx("imagenet_13ch") == list(range(13))  # own order: no pretrained band semantics


def test_s2_ssl_needs_all_13_bands():
    task = other_order_task()
    from dataclasses import replace
    with pytest.raises(ValueError, match="B8A"):
        replace(task, bands=FILE_ORDER[:-1]).input_idx("s2_ssl")


# --- metrics
def test_classification_metrics_top1():
    import torch
    preds, targets = torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 1, 2])  # 3 of 4 right
    assert get_task("eurosat").metrics(preds, targets) == {"top1": 0.75}


def test_classification_primary_metric_is_top1():
    import torch
    task = get_task("eurosat")
    assert task.primary_metric == "top1"
    assert task.score(torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 1, 2])) == 0.75


def test_metrics_empty_split_is_zero_not_nan():
    import torch
    assert get_task("eurosat").metrics(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) == {"top1": 0.0}
