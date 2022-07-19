conda activate 3dsemseg

# Add project root to pythonpath
insseg_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
downstream_dir="$(dirname "$insseg_dir")"
project_dir="$(dirname "$downstream_dir")"
export PYTHONPATH="${PYTHONPATH}:${project_dir}"

echo "${project_dir}"


outputs_base=/home/drozenberszki/dev/LongTailSemseg/output/instseg
for d in "$outputs_base"/*/ ; do
    echo "$d"

    if [[ "$d" == *"34D"* ]]
    then
      model=Res16UNet34D
    else
      model=Res16UNet34C
    fi

    python ddp_main.py train.is_train=False \
          train.lenient_weight_loading=True \
          net.model="$model" \
          data.dataset=Scannet200Voxelization2cmDataset \
          data.scannet_path=/mnt/data/Datasets/scannet_200_insseg \
          data.return_transformation=True \
          misc.log_dir=./outputs \
          train.resume="$d" \
          test.visualize=True \
          test.visualize_path="$d"visualize

done