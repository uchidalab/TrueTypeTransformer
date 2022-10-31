#!/bin/bash
docker exec -itd T3_docker tensorboard --logdir=logs --host=0.0.0.0 --port=6006
