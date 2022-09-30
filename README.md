## Neus_tx

#### colmap 
DATASET_PATH=/mnt/data2/lzc/tx/data/scene06fo
colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images \
    --single_camera 1 \
    --sparse 1 \
    --gpu_index 2 \
    --dense 1 

处理自己的数据:
