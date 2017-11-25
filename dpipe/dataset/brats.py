import numpy as np

from .from_csv import FromCSVInt


# We need this class because in the original data segm values are [0, 1, 2, 4]
class Brats2017(FromCSVInt):
    def __init__(self, data_path, metadata_rpath='metadata.csv', modalities=('t1', 't1ce', 't2', 'flair')):
        super().__init__(
            data_path=data_path,
            metadata_rpath=metadata_rpath,
            modalities=list(modalities),
            target='segm',
            segm2msegm_matrix=np.array([
                [0, 0, 0],
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 1]
            ], dtype=bool)
        )

    def load_segm(self, identifier):
        segm = super().load_segm(identifier)
        segm[segm == 4] = 3
        return segm

# For Brats 2015
# segm2msegm = np.array([
#     [0, 0, 0],
#     [1, 1, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [1, 1, 1]
# ], dtype=bool)
