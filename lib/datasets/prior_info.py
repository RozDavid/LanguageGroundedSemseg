from lib.datasets.scannet import *

class Scannet200Textual2cmDataset(Scannet200Voxelization2cmDataset):

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 phase=DatasetPhase.Train):

        super().__init__(
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config,
            cache=cache,
            phase=phase)

        language_features_path = config.scannet_path + '/' + config.language_features_path
        if os.path.isfile(language_features_path):
            with open(language_features_path, 'rb') as f:
                self.loaded_text_features = pickle.load(f)

            logging.info(f"Loaded file {config.language_features_path}")
        else:
            logging.info(f"Cant find file {config.language_features_path}")


class ScannetTextual2cmDataset(ScannetVoxelization2cmDataset):

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 phase=DatasetPhase.Train):

        super().__init__(
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config,
            cache=cache,
            phase=phase)

        language_features_path = config.scannet_path + '/' + config.language_features_path
        if os.path.isfile(language_features_path):
            with open(language_features_path, 'rb') as f:
                self.loaded_text_features = pickle.load(f)

            logging.info(f"Loaded file {config.language_features_path}")
        else:
            logging.info(f"Can't find file {config.language_features_path}")


class Scannet200TextualDataset(Scannet200Textual2cmDataset):
    VOXEL_SIZE = 0.05

