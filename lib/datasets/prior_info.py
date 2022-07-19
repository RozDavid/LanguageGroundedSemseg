from lib.datasets.scannet import *

class GraphPrior2cmDataset(Scannet200Voxelization2cmDataset):

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

        mean_features_path = config.scannet_path + '/' + config.mean_features_path
        if os.path.isfile(mean_features_path):
            with open(mean_features_path, 'rb') as f:
                loaded_mean_features = pickle.load(f)

                self.feature_means = torch.Tensor(loaded_mean_features['feature_means'])
                if config.representation_distance_type == 'cos':
                    self.loaded_category_means = torch.Tensor(loaded_mean_features['normed_category_means'])
                else:
                    self.loaded_category_means = torch.Tensor(loaded_mean_features['l2_category_means'])


        spatial_distances_path = config.scannet_path + '/' + config.spatial_distances_path
        if os.path.isfile(spatial_distances_path):
            with open(spatial_distances_path, 'rb') as f:
                spatial_distances = np.load(f)

            # Normalize spatial distances mat and take 1 - sigmoids for dot similarities
            max_dist = np.nanmax(spatial_distances)
            spatial_distances[np.isnan(spatial_distances)] = max_dist

            sigmoid_distances = 1 - (1 / (1 + np.exp(-spatial_distances)))  # 1-sigmoid(distances)
            sigmoid_distances /= sigmoid_distances.max()

            # Make objects self distance minimal
            sigmoid_distances[np.diag_indices(sigmoid_distances.shape[0])] = sigmoid_distances.max()

            # Save graph as global variable
            self.spatial_distances = torch.Tensor(sigmoid_distances)

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
            logging.info(f"Cant find file {config.language_features_path}")


class Scannet200TextualDataset(Scannet200Textual2cmDataset):
    VOXEL_SIZE = 0.05

