#!/bin/bash

step=5

echo "Step:$step" && sleep 2;

dataset=/mipirepo/data/acoustic_scene_data/TAU-urban-acoustic-scenes-2020-mobile-development
wavefile=$dataset/audio
task=2020task1a

# CUDA_VISIBLE_DEVICES="3"

if [[ $step -eq 1 ]]; then
  echo "## Training:"
  thefeature=data/$task/mono256dim/norm
  data=data/$task/evaluation_setup
  CUDA_VISIBLE_DEVICES="3" python train.py --preset=conf/${task}_256dim.json $data $thefeature exp/$task/mono256dim || exit 1;
fi

if [[ $step -eq 2 ]]; then
  echo "## Training:"
  thefeature=data/$task/mono256dim/norm
  data=data/$task/evaluation_setup_A
  CUDA_VISIBLE_DEVICES="3" python train.py --preset=conf/${task}_logmel256dim.json $data $thefeature exp/$task/mono256dimA || exit 1;
fi

if [[ $step -eq 3 ]]; then
  echo "## Training:"
  thefeature=data/$task/mono256dim/norm
  data=data/$task/evaluation_setup_all
  CUDA_VISIBLE_DEVICES="3" python train.py --preset=conf/${task}_logmel256dim.json $data $thefeature exp/$task/mono256dimall || exit 1;
fi

if [[ $step -eq 4 ]]; then
  echo "## Extract Feature"
  echo "## Wavefile:$wavefile"
  featpath=data/$task/hpss256dim/org
  python preprocess.py --num_workers=30 --preset=conf/2020task1a_hpss256dim.json $wavefile $featpath || exit 1;
fi

if [[ $step -eq 5 ]]; then
  echo "## Training:"
  thefeature=data/$task/hpss256dim/org
  data=data/$task/evaluation_setup
  CUDA_VISIBLE_DEVICES="2" python train.py --preset=conf/${task}_hpss256dim.json $data $thefeature exp/$task/mono256dimhpss || exit 1;
fi