#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -it \
    --gpus=all \
    --ipc=host \
    --name=t3_docker \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$PWD/../data/fonts:/data/fonts \
    t3_docker:latest \
    fish
