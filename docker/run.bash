container_name=$1

cd ..
CUDA_VISIBLE_DEVICES='6'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --name $container_name \
    --mount src=$(pwd),dst=/visil,type=bind \
    --mount src=/media/data2/,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /visil \
    litcoderr:visil \
    bash -c "bash" \
