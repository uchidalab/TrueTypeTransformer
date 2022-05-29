#!/bin/bash
docker exec -itd my_docker tensorboard --logdir=logs --host=0.0.0.0 --port=6006
