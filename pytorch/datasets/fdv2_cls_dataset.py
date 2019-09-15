"""
    FDV2 classification dataset.
"""

from .fdv1_cls_dataset import FDV1MetaInfo


class FDV2MetaInfo(FDV1MetaInfo):
    """
    Descriptor of FDV2 dataset.
    """
    def __init__(self):
        super(FDV2MetaInfo, self).__init__()
        self.label = "FDV2"
        self.short_label = "fdv2"
        self.root_dir_name = "fdv2"
