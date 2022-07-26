import argparse

def str2opt(arg):
    assert arg in ['SGD', 'Adam']
    return arg

def str2scheduler(arg):
    assert arg in ['StepLR', 'MultiStepLR', 'PolyLR', 'ExpLR', 'SquaredLR', 'ReduceLROnPlateau']
    return arg

def str2evaluate(arg):
    assert arg in ['target_down', 'pred_up']
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

def str2intlist(l):
    return [int(i) for i in l.split(',')]

def str2stringlist(l):
    split = l.split(',')
    if len(l) == 0:
        return []
    else:
        return [str(s) for s in split]

def str2floatlist(l):
    return [float(i) for i in l.split(',')]

def str2graphpriordist(arg):
    assert arg in ['cos', 'l2', 'l1']
    return arg

def str2embloss(arg):
    assert arg in ['contrastive', 'l2']
    return arg

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def instance_augmentation(arg):
    assert arg in ['latent', 'raw']
    return str(arg)

arg_lists = []
parser = argparse.ArgumentParser()

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNet14', help='Model name')
net_arg.add_argument('--conv1_kernel_size', type=int, default=3, help='First layer conv kernel size')
net_arg.add_argument('--weights', type=str, default='None', help='Saved weights to load')
net_arg.add_argument('--weights_for_inner_model',  type=str2bool, default=False, help='Weights for model inside a wrapper')
net_arg.add_argument('--dilations', type=str2intlist, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')
net_arg.add_argument('--resolution_evaluation', type=str2evaluate, default='pred_up',
                     help='If output is not from final layer,'
                          ' defines if pred or target shold be aligned to find correspondences')
net_arg.add_argument('--child_classifier_dimension', type=int, default=30,
                     help='How many childs a base parent class can predict')

# Wrappers
net_arg.add_argument('--wrapper_type', default='None', type=str, help='Wrapper on the network')
net_arg.add_argument('--wrapper_region_type', default=1, type=int,  help='Wrapper connection types 0: hypercube, 1: hypercross, (default: 1)')
net_arg.add_argument('--wrapper_kernel_size', default=3, type=int, help='Wrapper kernel size')
net_arg.add_argument('--wrapper_lr', default=1e-1,  type=float, help='Used for freezing or using small lr for the base model, freeze if negative')

# Meanfield arguments
net_arg.add_argument('--meanfield_iterations', type=int, default=10, help='Number of meanfield iterations')
net_arg.add_argument('--crf_spatial_sigma', default=1, type=int, help='Trilateral spatial sigma')
net_arg.add_argument('--crf_chromatic_sigma', default=12, type=int, help='Trilateral chromatic sigma')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--lr', type=float, default=0.05)
opt_arg.add_argument('--separate_lrs', type=str2floatlist, default='0.05,0.05,0.05,0.05',
                     help='The learning rates for the base model, parent head, refinement head and children head respectively [0.05,0.05,0.05,0.05]')
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.02)
opt_arg.add_argument('--parent_bce_weights', type=str2floatlist, default='1,1')
opt_arg.add_argument('--supercat_bce_weights', type=str2floatlist, default='1,1')
opt_arg.add_argument('--classifier_only', type=str2bool, default=False,
                     help='For freezing the whole model and train a classifier (final layer) only')

# Loss params
opt_arg.add_argument('--loss_type', type=str, default='cross_entropy')  # cross_entropy, focal, weighted_ce
opt_arg.add_argument('--focal_alpha', type=float, default=1.)  # cross_entropy, focal, weighted_ce


# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='MultiStepLR')
opt_arg.add_argument('--max_iter', type=int, default=10e6)
opt_arg.add_argument('--max_epoch', type=int, default=400)
opt_arg.add_argument('--step_size', type=int, default=2e4)
opt_arg.add_argument('--multi_step_milestones', type=str2intlist, default=[120, 150])
opt_arg.add_argument('--step_gamma', type=float, default=0.3)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)
opt_arg.add_argument('--exp_step_size', type=float, default=445)
opt_arg.add_argument('--scheadule_monitor', type=str, default='val_miou')
opt_arg.add_argument('--scheduler_min_lr', type=float, default=10e-4)
opt_arg.add_argument('--reduce_patience', type=float, default=20)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='outputs/default')
dir_arg.add_argument('--data_dir', type=str, default='data')
dir_arg.add_argument('--child_label_mapper', type=str, default='child_classifier_mapping_30_ids.pkl',
                     help='A dictionary containing possible child category ids for all parent category ids')

# Weighting categories
dir_arg.add_argument('--category_weights', type=str, default='feature_data/scannet200_category_weights.pkl',
                     help='A dictionary containing normalized weights based on the validation point frequencies/category')
dir_arg.add_argument('--category_frequencies_path', type=str, default='feature_data/dataset_frequencies.pkl',
                     help='Loading teh weights (log histogram of cat frequencies) for all categories')
dir_arg.add_argument('--weighted_cross_entropy', type=str2bool, default=False,
                     help='Whether to apply some weights to the CE losses')
dir_arg.add_argument('--instance_sampling_weights', type=str, default='feature_data/tail_split_inst_sampling_weights.pkl',
                     help='A dictionary containing probability weights to pick a category for additional sampling')
dir_arg.add_argument('--sample_tail_instances', type=str2bool, default=False,
                     help='Adding new instances on the fly for more balanced sampling')
dir_arg.add_argument('--bounding_boxes_path', type=str, default='feature_data/full_train_bbs_with_rels.pkl',
                     help='A precomputed dictionary containing bounding boxes of al instances')
dir_arg.add_argument('--correct_samples_prop_path', type=str, default='feature_data/prop_of_points_to_sample.pkl',
                     help='For saving a balanced set of feature maps')
dir_arg.add_argument('--max_instance_placing_iterations', type=int, default=50,
                     help='If we cant find a place for the new instance we skip the placement')
dir_arg.add_argument('--num_instances_to_add', type=int, default=5,
                     help='How many new instances we want to sample to every train scene')
dir_arg.add_argument('--sampled_features', type=str2bool, default=False,
                     help='If we want to save sample intermediate features for evaluation')

# Graph prior
graph_arg = add_argument_group('Graph')
graph_arg.add_argument('--mean_features_path', type=str, default='feature_data/Res16UNet34C200_mean_features.pkl',
                     help='Loading the mean features for the graph prior multiplication')
graph_arg.add_argument('--spatial_distances_path', type=str, default='spatial_graph_distances.npy',
                     help='Loading the nxn matrix containing the instance distances')
graph_arg.add_argument('--language_features_path', type=str, default='feature_data/clip_feats_scannet_200.pkl',
                     help='Language category representation clusters for the 200 cats')


# Metric learning
metric_arg = add_argument_group('Metric')
metric_arg.add_argument('--use_embedding_loss', type=str, default=None)
metric_arg.add_argument('--embedding_loss_type', type=str, default='contrast')
metric_arg.add_argument('--num_pos_samples', type=int, default=1)
metric_arg.add_argument('--num_negative_samples', type=int, default=3)
metric_arg.add_argument('--clip_uniform_sampling', type=str2bool, default=True, help='If we want to sample negatives from the scene or any other category')
metric_arg.add_argument('--contrast_pos_thresh', type=float, default=0.0, help='If features are closer than this, no loss should be applied')
metric_arg.add_argument('--contrast_neg_thresh', type=float, default=0.6, help='After the distance is smaller than this, loss shouldnt be applied')
metric_arg.add_argument('--contrast_neg_weight', type=float, default=1.0, help='weighting factor for negative distancing loss')
metric_arg.add_argument('--embedding_loss_lambda', type=float, default=1.0,
                        help='Multiplier for embedding loss with representations and target clusters (e.g train mean features or GloVe features')
metric_arg.add_argument('--representation_distance_type', type=str2graphpriordist, default='cos',
                        help='Either \'cos\' or \'l2\' for finding most similar features')
metric_arg.add_argument('--normalize_features', type=str2bool, default=False,
                        help='Unit sphere projection penalization')
metric_arg.add_argument('--feat_norm_loss_max', type=float, default=0.2,
                        help='Clamping the penalty for feature norm')
metric_arg.add_argument('--learned_projection', type=str2bool, default=False, help='Learn a projection function from a higher order representation space to a smaller one')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='Scannet200Voxelization2cmDataset')
data_arg.add_argument('--temporal_dilation', type=int, default=30)
data_arg.add_argument('--temporal_numseq', type=int, default=3)
data_arg.add_argument('--point_lim', type=int, default=-1)
data_arg.add_argument('--pre_point_lim', type=int, default=-1)
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--val_batch_size', type=int, default=1)
data_arg.add_argument('--test_batch_size', type=int, default=1)
data_arg.add_argument('--effective_batch_size', type=int, default=8)
data_arg.add_argument('--cache_data', type=str2bool, default=False)
data_arg.add_argument('--num_workers', type=int, default=4, help='num workers for train dataloader')
data_arg.add_argument('--num_val_workers', type=int, default=4, help='num workers for val/test dataloader')
data_arg.add_argument('--ignore_label', type=int, default=-1)
data_arg.add_argument('--return_transformation', type=str2bool, default=False)
data_arg.add_argument('--ignore_duplicate_class', type=str2bool, default=False)
data_arg.add_argument('--partial_crop', type=float, default=0.)
data_arg.add_argument('--train_limit_numpoints', type=int, default=1800000)

# Point Cloud Dataset
data_arg.add_argument('--scannet_path', type=str, default='', help='Scannet online voxelization dataset root dir')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--stat_freq', type=int, default=40, help='print frequency')
train_arg.add_argument('--test_stat_freq', type=int, default=100, help='print frequency')
train_arg.add_argument('--visualize_freq', type=int, default=0, help='Save colored pointclouds')
train_arg.add_argument('--save_freq', type=int, default=1000, help='save frequency')
train_arg.add_argument('--val_freq', type=int, default=400, help='validation frequency')
train_arg.add_argument('--empty_cache_freq', type=int, default=4, help='Clear pytorch cache frequency')
train_arg.add_argument('--train_phase', type=str, default='train', help='Dataset for training')
train_arg.add_argument('--val_phase', type=str, default='val', help='Dataset for validation')
train_arg.add_argument('--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
train_arg.add_argument('--resume_optimizer',default=True, type=str2bool, help='Use checkpoint optimizer states when resume training')
train_arg.add_argument('--eval_upsample', type=str2bool, default=False)
train_arg.add_argument('--lenient_weight_loading', type=str2bool, default=True, help='Weights with the same size will be loaded')

# Data augmentation
data_aug_arg = add_argument_group('DataAugmentation')
data_aug_arg.add_argument(
    '--train_augmentation', type=str2bool, default=True, help='Turn off augmentation altogether')
data_aug_arg.add_argument(
    '--elastic_distortion', type=str2bool, default=True, help='Turn off geometry distortion')
data_aug_arg.add_argument(
    '--use_feat_aug', type=str2bool, default=True, help='Simple feat augmentation')
data_aug_arg.add_argument(
    '--data_aug_color_trans_ratio', type=float, default=0.10, help='Color translation range')
data_aug_arg.add_argument(
    '--data_aug_color_jitter_std', type=float, default=0.05, help='STD of color jitter')
data_aug_arg.add_argument(
    '--data_aug_color_scaling_factor', type=float, default=1.0, help='To linearly scale the color features')
data_aug_arg.add_argument('--normalize_color', type=str2bool, default=True)
data_aug_arg.add_argument('--data_aug_scale_min', type=float, default=0.9)
data_aug_arg.add_argument('--data_aug_scale_max', type=float, default=1.1)
data_aug_arg.add_argument('--data_aug_hue_max', type=float, default=0.5, help='Hue translation range. [0, 1]')
data_aug_arg.add_argument('--data_aug_saturation_max', type=float, default=0.20, help='Saturation translation range, [0, 1]')
data_aug_arg.add_argument('--data_aug_patch_dropout_ratio', type=float, default=0.35, help='For SimSiam pair dataloader, drop percentage of points of this ratio')
data_aug_arg.add_argument('--instance_augmentation', type=str, default=None, help='For applying targeted augmentation of less frequent categories. [raw, latent] for [pointcloud, latent space] augmentation')
data_aug_arg.add_argument('--instance_augmentation_color_aug_prob', type=float, default=0.5, help='The probability to apply targeted hue shift on the instances')
data_aug_arg.add_argument('--instance_augmentation_scale_aug_prob', type=float, default=0.2, help='The probability to apply targeted scaling on the instances')
data_aug_arg.add_argument('--projection_model_path', type=str, default='feature_data/scannet200_attribute_projection_model.ckpt', help='The pretrained model that learned the attributes in an offline optimization')


# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--visualize', type=str2bool, default=False)
test_arg.add_argument('--test_temporal_average', type=str2bool, default=False)
test_arg.add_argument('--visualize_path', type=str, default='outputs/visualize')
test_arg.add_argument('--save_prediction', type=str2bool, default=False)
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred')
test_arg.add_argument('--test_phase', type=str, default='test', help='Dataset for test')
test_arg.add_argument(
    '--evaluate_original_pointcloud',
    type=str2bool,
    default=False,
    help='Test on the original pointcloud space during network evaluation using voxel projection.')
test_arg.add_argument(
    '--test_original_pointcloud',
    type=str2bool,
    default=False,
    help='Test on the original pointcloud space as given by the dataset using kd-tree.')

#Debugging
debug_arg = add_argument_group('Debug')
debug_arg.add_argument('--gt_type', type=str, default='none')
debug_arg.add_argument('--gt_types', type=str2stringlist, default='',
                       help='GT types passed in as a list of comma separated strings')
debug_arg.add_argument('--eval_only_on_parents', type=str2bool, default=True,
                       help='To see if hierarchical works within parent BBs')
debug_arg.add_argument('--overfit_batches', type=float, default=0.0,
                       help='Uses this much data of the training set. If nonzero, will use the same training set for validation and testing.')


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--seed', type=int, default=42)

# Data balancing
balance_arg = add_argument_group('Balancing')
balance_arg.add_argument('--balanced_category_sampling', type=str2bool, default=True)
balance_arg.add_argument('--balanced_sample_head_ratio', type=float, default=-1)
balance_arg.add_argument('--balanced_sample_common_ratio', type=float, default=-1)


def get_config():
    config = parser.parse_args()
    return config  # Training settings
