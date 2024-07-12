container_name=$1

CUDA_VISIBLE_DEVICES='7'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --name $container_name \
    --mount src=$(pwd),dst=/visil,type=bind \
    --mount src=/media/data2/,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -p 2323:8888 \
    -w /visil \
    cvpr24_video_retrieval:latest \
    bash -c "bash" \
