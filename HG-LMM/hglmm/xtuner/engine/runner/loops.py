# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.runner import IterBasedTrainLoop
from torch.utils.data import DataLoader
from mmengine.runner import EpochBasedTrainLoop as BaseEpochBasedTrainLoop


class TrainLoop(IterBasedTrainLoop):

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: Optional[int] = None,
                 max_epochs: Union[int, float] = None,
                 **kwargs) -> None:

        if max_iters is None and max_epochs is None:
            raise RuntimeError('Please specify the `max_iters` or '
                               '`max_epochs` in `train_cfg`.')
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError('Only one of `max_iters` or `max_epochs` can '
                               'exist in `train_cfg`.')
        else:
            if max_iters is not None:
                iters = int(max_iters)
                assert iters == max_iters, ('`max_iters` should be a integer '
                                            f'number, but get {max_iters}')
            elif max_epochs is not None:
                if isinstance(dataloader, dict):
                    diff_rank_seed = runner._randomness_cfg.get(
                        'diff_rank_seed', False)
                    dataloader = runner.build_dataloader(
                        dataloader,
                        seed=runner.seed,
                        diff_rank_seed=diff_rank_seed)
                iters = max_epochs * len(dataloader)
            else:
                raise NotImplementedError
        super().__init__(
            runner=runner, dataloader=dataloader, max_iters=iters, **kwargs)


# Copyright (c) OpenMMLab. All rights reserved.


class EpochBasedTrainLoop(BaseEpochBasedTrainLoop):
    """简化版 Epoch 训练循环，仅支持通过 max_epochs 指定训练轮次"""

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_epochs: int,  # 强制要求必须指定 max_epochs
                 **kwargs) -> None:
        # 构建 dataloader（如果传入的是配置字典）
        if isinstance(dataloader, dict):
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            dataloader = runner.build_dataloader(
                dataloader,
                seed=runner.seed,
                diff_rank_seed=diff_rank_seed)

        # 直接调用父类初始化，无需复杂转换
        super().__init__(
            runner=runner,
            dataloader=dataloader,
            max_epochs=max_epochs,
            **kwargs)


