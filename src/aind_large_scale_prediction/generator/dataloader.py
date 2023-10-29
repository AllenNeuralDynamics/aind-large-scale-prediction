"""
Custom zarr data loader to make sure
shuffling is deactivated.
"""

from typing import Optional

from torch.utils.data import DataLoader, Dataset


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
