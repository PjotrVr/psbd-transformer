#!/bin/bash
# The two controls that decide whether the occlusion probe is a detector or an artefact.
#
#   benign  - a model with NO backdoor, probed with the a2a trigger. If this fires, the
#             probe is reacting to trigger-shaped pixels rather than to a backdoor, and
#             the whole thing is worthless.
#   a2o     - all-to-one. Every poisoned sample already predicts the SAME class, so
#             occlusion sends them back to many different classes: a one-to-many map,
#             not a permutation. The probe should NOT fire here, which is what makes it
#             complementary to PSU rather than a replacement.
set -u
BASE=/lustre/home/pstika/projects/PSBD-ViT
cd $BASE; source .venv/bin/activate; export PYTHONPATH=$BASE

until grep -q "AUROC (occlusion" /tmp/claude-5225/-lustre-home-pstika-projects-PSBD-ViT/e54a8c64-6982-4399-8268-78a75b86bf49/tasks/bs79f5x88.output 2>/dev/null; do sleep 20; done

echo "############ 2. BENIGN MODEL, a2a trigger (negative control) ############"
python experiments/all_to_all_occlusion/occlusion_probe.py \
    --checkpoint checkpoints/vit_cifar10_benign/attack_result.pt \
    --probe-attack badnet_a2a --probe-target-label 0 --limit 1500 2>&1 \
    | grep -viE "warning|warn|Seed set" | tail -14

echo
echo "############ 3. ALL-TO-ONE (should NOT fire) ############"
python experiments/all_to_all_occlusion/occlusion_probe.py \
    --checkpoint checkpoints/vit_cifar10_badnet_a2o_0_1/attack_result.pt --limit 1500 2>&1 \
    | grep -viE "warning|warn|Seed set" | tail -14
echo "############ CONTROLS COMPLETE ############"
