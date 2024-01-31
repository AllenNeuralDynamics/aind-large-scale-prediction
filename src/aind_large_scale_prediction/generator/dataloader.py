"""
Custom zarr data loader to make sure
shuffling is deactivated.
"""

from typing import Optional

from torch.utils.data import DataLoader, Dataset


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """
        Constructor
        """
        self.sampler = sampler

    def __iter__(self):
        """
        Iteration method
        """
        while True:
            yield from iter(self.sampler)


class ZarrDataLoader(DataLoader):
    """
    Zarr custom data loader
    """

    def __init__(
        self, dataset: Dataset, shuffle: Optional[bool] = False, **kwargs
    ):
        """
        Init method

        Parameters
        ----------
        dataset: Dataset
            Dataset to load

        shuffle: Optional[bool]
            Should be always False to manage
            each super chunk at a time.
        """
        if shuffle:
            raise ValueError("Zarr data loader only works without shuffling")

        super(ZarrDataLoader, self).__init__(
            dataset=dataset, shuffle=False, **kwargs
        )
        # object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.iterator = super().__iter__()

    # def __len__(self):
    #     return len(self.batch_sampler.sampler)

    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield next(self.iterator)
