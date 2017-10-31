import numpy as np
from .from_csv import FromCSVInt
from dpipe.config import register


@register('oasis')
class Oasis(FromCSVInt):
    def __init__(self, data_path, metadata_rpath='metadata.csv'):
        super().__init__(
            data_path=data_path,
            metadata_rpath=metadata_rpath,
            modalities=['S'],
            target='T',
            segm2msegm_matrix=np.array([[1], [1], [1], [1]], dtype=bool)
        )

    def load_segm(self, patient_id):
        return super().load_segm(patient_id).astype(int)

