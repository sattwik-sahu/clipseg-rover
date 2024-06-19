#!/bin/bash

# Start the ROS 2 development container with GUI support and X11 forwarding
docker run -d -it --rm --privileged \
  --net=host \
  --ipc=host \
  --gpus="all" \
  --runtime="nvidia" \
  --env="DISPLAY" \
  -v $PWD:/home/sattwik/vislang-map:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume="${XAUTHORITY}:/root/.Xauthority:rw" \
  -e "TERM=xterm-256color" \
  --device /dev/dri:/dev/dri:rw \
  --device /dev:/dev:rw \
  --user="sattwik" \
  --name clipseg-rover \
  clipseg-rover
