import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import torch
from concurrent.futures import ProcessPoolExecutor

from lib.constants.scannet_constants import *
from utils import *
from lib.ext.pointnet2.pointnet2_utils import furthest_point_sample

# Modify path to point where ScanNet data lives
SCANNET_RAW_PATH = Path('/mnt/data/ScanNet')
SCANNET_OUT_PATH = Path('/mnt/data/Datasets/limited_annotation/scannet_200')
COMBINED_LABEL_NAMES_FILE = 'scannetv2-labels.combined.tsv'


in_path = 'scans'
POINTCLOUD_FILE = '_vh_clean_2.ply'

TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}

CLASS_IDs = VALID_CLASS_IDS_LONG
num_threads = 8

# Limited annotations
min_points_in_instance = 5
ratio_of_annotated = -1

# Save instances independently
save_instances = False

# Load categories
print('Load Label map')
labels_pd = pd.read_csv(str(SCANNET_RAW_PATH) + '/' + COMBINED_LABEL_NAMES_FILE, sep='\t', header=0)
labels_pd.loc[labels_pd.raw_category == 'stick', ['category']] = 'object'
labels_pd.loc[labels_pd.category == 'wardrobe ', ['category']] = 'wardrobe'
category_label_names = labels_pd['category'].unique()

# Preprocess data.
print('Start Preprocess')
def handle_process(path):

    cloud_file = Path(path.split(',')[0])
    segments_file = cloud_file.parent / (cloud_file.stem + '.0.010000.segs.json')
    aggregations_file = cloud_file.parent / (cloud_file.stem[:-len('_vh_clean_2')] + '.aggregation.json')
    info_file = cloud_file.parent / (cloud_file.stem[:-len('_vh_clean_2')] + '.txt')
    phase_out_path = Path(path.split(',')[1])

    scene_id = cloud_file.stem[:-(len(POINTCLOUD_FILE) - len(cloud_file.suffix))]
    print('Processing: ', scene_id, 'in', phase_out_path.name)

    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=' ')

    if 'axisAlignment' not in info_dict:
        rot_matrix = np.identity(4)
    else:
        rot_matrix = info_dict['axisAlignment'].reshape(4, 4)

    mesh = o3d.io.read_triangle_mesh(str(cloud_file))
    points = np.array(mesh.vertices)
    colors = np.round(np.array(mesh.vertex_colors) * 255.)
    alphas = (np.ones(points.shape[0]) * 255).reshape(-1, 1)
    pointcloud = np.hstack((points, colors, alphas))
    faces_array = np.array(mesh.triangles)

    # Rotate PC to axis aligned
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    r_points = np.dot(rot_matrix, r_points)
    pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1)

    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments['segIndices'])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation['segGroups'])

    # Make sure alpha value is meaningless.
    assert np.unique(pointcloud[:, -1]).size == 1

    # Generate new labels
    labelled_pc = np.zeros((pointcloud.shape[0], 1))
    instance_ids = np.zeros((pointcloud.shape[0], 1))
    for group in seg_groups:
        segment_points, p_inds, label_id = point_indices_from_group(pointcloud, seg_indices, group, labels_pd, CLASS_IDs)

        # Apply limited annotation if necessary
        if ratio_of_annotated != -1 and scene_id in TRAIN_SCENES:
            coords = pointcloud[p_inds, :3]
            t_coords = torch.Tensor(coords).cuda().unsqueeze(0)
            points_to_sample = max(min_points_in_instance, round(ratio_of_annotated * p_inds.shape[0]))
            sampled_inds = furthest_point_sample(t_coords,  points_to_sample).squeeze(0).long().cpu().numpy()
            p_inds = np.vectorize(lambda p: p_inds[p])(sampled_inds)
        else:
            sampled_inds = None

        labelled_pc[p_inds] = label_id
        instance_ids[p_inds] = group['id']

        cat_name = labels_pd[labels_pd['id'] == label_id]['category'].iloc[0] if label_id > 0 else 'invalid'
        if save_instances and cat_name in TAIL_CATS_SCANNET_200:
            # Save segment points as instance
            # calculate segment faces instances
            # uncomment if saving meshes instead of point clouds
            #face_node_mapper = lambda n: np.where(p_inds == n)[0][0]
            #inst_face_mask = np.all(np.isin(faces_array, p_inds), axis=1)
            #inst_faces = faces_array[inst_face_mask]
            #inst_faces = np.vectorize(face_node_mapper)(inst_faces)

            save_instance(segment_points, label_id, cat_name, scene_id, str(SCANNET_OUT_PATH), limited_annotation_points=sampled_inds)  # , segment_faces=inst_faces

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    # Concatenate with original cloud
    processed = np.hstack((pointcloud[:, :6], labelled_pc, instance_ids))

    if (np.any(np.isnan(processed)) or not np.all(np.isfinite(processed))):
        raise ValueError('nan')

    # Save processed cloud
    out_file = phase_out_path / (cloud_file.name[:-len(POINTCLOUD_FILE)] + cloud_file.suffix)
    save_point_cloud(processed, out_file, with_label=True, verbose=False)


train_scenes_file = open(SCANNET_RAW_PATH / "scans.txt", 'r')
train_scenes = train_scenes_file.readlines()

# Strips the newline character
for i, line in enumerate(train_scenes):
    train_scenes[i] = line.rstrip()

test_scenes_file = open(SCANNET_RAW_PATH / 'scans_test.txt', 'r')
test_scenes = test_scenes_file.readlines()
for i, line in enumerate(test_scenes):
    test_scenes[i] = line.rstrip()

path_list = []
train_pc_files = list((SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE))
for f in train_pc_files:

    scene_id = f.name[:-len(POINTCLOUD_FILE)]

    if scene_id in train_scenes:
        out_path = TRAIN_DEST
    elif scene_id in test_scenes:
        out_path = TEST_DEST
    else:
        out_path = ''
        print('ERROR: no matching scene id')

    phase_out_path = SCANNET_OUT_PATH / out_path
    phase_out_path.mkdir(parents=True, exist_ok=True)

    path_list.append(str(f) + ',' + str(phase_out_path))

pool = ProcessPoolExecutor(max_workers=num_threads)
result = list(pool.map(handle_process, path_list))
