#!/bin/bash
# All renders used in the report. Run inside the container (bash isaac/run_container.sh).
# About 12 s per scene on an RTX GPU; the main dataset takes ~45 min.
cd /workspace
mkdir -p logs
gen() { PYTHONUNBUFFERED=1 /isaac-sim/python.sh isaac/generate_scenes.py "$@"; }

# planes for the noise check (fronto-parallel and tilted by 25 deg)
gen --mode plane --hdr 0 --out data/plane_fronto > logs/plane_fronto.txt 2>&1
gen --mode plane --hdr 0 --tilt-deg 25 --out data/plane_tilt25 > logs/plane_tilt25.txt 2>&1

# main dataset: 200 scenes
gen --mode scenes --num-scenes 200 --seed 1 --out data/scenes_main > logs/scenes_main.txt 2>&1

# sensor-physics ablations (HDR radiance kept): 3 scenes
gen --mode scenes --num-scenes 3 --seed 1 --hdr 1 --out data/hdr_test > logs/hdr_test.txt 2>&1

# anodized material variants: metallic / roughness, same geometry as the first 30 main scenes
for v in "1.0 0.30 r030" "1.0 0.20 r020" "0.7 0.45 m070"; do
  set -- $v
  gen --mode scenes --num-scenes 30 --seed 1 --hdr 0 --metallic-anodized $1 --rough-anodized $2 \
    --out data/anodized_$3 > logs/anodized_$3.txt 2>&1
done

# roughness of the polished finish: same seed, so same geometry as the first 30 main scenes
for r in 0.05 0.20 0.30; do
  gen --mode scenes --num-scenes 30 --seed 1 --hdr 0 --rough-polished $r --out data/polished_r${r/./} \
    > logs/polished_r${r/./}.txt 2>&1
done
