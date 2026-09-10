#!/bin/bash
# Generate and submit one PBS job per (arch, dataset, attack, rate) combination.
# 120 jobs total: 2 arch x 2 datasets x 10 attacks x 3 rates.
# v2: auto-calibrated probe rate, fractional PSU, k=3 passes, increased walltime.

BASE=/lustre/home/pstika/projects/PSBD-ViT
LOGDIR=$BASE/logs/psbd_evade
PBSDIR=$BASE/pbs/psbd_evade
mkdir -p "$LOGDIR"

ATTACKS="badnet_a2o blend wanet lc adaptive_blend sig lf bpp tact badnet_a2a"

# Rate tag for checkpoint naming: 0.10 -> 0_1, 0.05 -> 0_05, 0.01 -> 0_01
rate_tag() {
    case "$1" in
        0.10) echo "0_1" ;;
        0.05) echo "0_05" ;;
        0.01) echo "0_01" ;;
    esac
}

# Walltime depends on dataset (Tiny is larger). Increased from v1 (4h/6h) to
# account for auto-calibration overhead and to avoid walltime kills that lose
# the entire run (checkpoint is written only after the last epoch).
walltime() {
    case "$1" in
        tiny) echo "12:00:00" ;;
        *)    echo "06:00:00" ;;
    esac
}

# Best evasion operator per architecture
evade_op() {
    case "$1" in
        vit)  echo "token_mask" ;;
        swin) echo "dropout" ;;
    esac
}

# Short dataset name for job/log naming
ds_short() {
    case "$1" in
        cifar100) echo "c100" ;;
        tiny)     echo "tiny" ;;
    esac
}

# Sweep commands differ between ViT (4 transfer ops) and Swin (4 transfer ops)
vit_sweep() {
    local CKPT=$1
    cat <<'SWEEP'
    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config before_attention_norm \
        --perturbation token_mask \
        --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --forward-passes 3 --skip-existing

    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config before_attention_norm \
        --rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --forward-passes 3 --skip-existing

    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config mlp_norm_out \
        --perturbation gain_scale \
        --rates 0.25 0.5 1 2 3 4 6 9 \
        --forward-passes 3 --skip-existing

    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config before_mlp \
        --perturbation gaussian \
        --rates 0.05 0.1 0.2 0.3 0.5 0.75 1 1.5 2 3 \
        --forward-passes 3 --skip-existing
SWEEP
}

swin_sweep() {
    cat <<'SWEEP'
    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config before_attention_norm \
        --rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --forward-passes 3 --skip-existing

    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config before_attention_norm \
        --perturbation token_mask \
        --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --forward-passes 3 --skip-existing

    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config mlp_norm_out \
        --perturbation gain_scale \
        --rates 0.25 0.5 1 2 3 4 6 9 \
        --forward-passes 3 --skip-existing

    python -m cli.sweep \
        --checkpoint-folder CKPT_PLACEHOLDER \
        --position-config pre_residual \
        --rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --forward-passes 3 --skip-existing
SWEEP
}

COUNT=0

for ARCH in vit swin; do
    OP=$(evade_op $ARCH)
    for DATASET in cifar100 tiny; do
        DS=$(ds_short $DATASET)
        WT=$(walltime $DATASET)
        for RATE in 0.10 0.05 0.01; do
            RT=$(rate_tag $RATE)
            for ATTACK in $ATTACKS; do
                CKPT="${ARCH}_${DATASET}_${ATTACK}_${RT}_evade_l1"
                JOBNAME="ev_${ARCH:0:1}_${DS}_${ATTACK}_${RT}"
                PBSFILE="$PBSDIR/${JOBNAME}.pbs"

                # Build sweep block
                if [ "$ARCH" = "vit" ]; then
                    SWEEP_BLOCK=$(vit_sweep | sed "s/CKPT_PLACEHOLDER/$CKPT/g")
                else
                    SWEEP_BLOCK=$(swin_sweep | sed "s/CKPT_PLACEHOLDER/$CKPT/g")
                fi

                cat > "$PBSFILE" <<EOF
#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime=${WT}
#PBS -N ${JOBNAME}
#PBS -o ${LOGDIR}/${JOBNAME}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  \$PBS_JOBID"
echo "Node:    \$(hostname)"
echo "Started: \$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

cd ${BASE}
source .venv/bin/activate

echo "=== train ${CKPT} ==="

python -m cli.train_backdoor \\
    --dataset ${DATASET} --attack ${ATTACK} --poison-rate ${RATE} \\
    --target-label 0 --architecture ${ARCH} --epochs 15 --seed 0 \\
    --batch-size 48 --num-workers 8 \\
    --evade-psbd --evade-weight 1.0 \\
    --evade-operator ${OP} --evade-position before_attention_norm \\
    --evade-rate 0 --evade-passes 3 \\
    --output checkpoints/${CKPT}/attack_result.pt

echo "=== sweep ${CKPT} ==="

${SWEEP_BLOCK}

echo "Finished: \$(date)"
exit 0
EOF
                COUNT=$((COUNT + 1))
                if [ "${DRY_RUN:-0}" = "1" ]; then
                    echo "[dry] $PBSFILE"
                else
                    JOBID=$(qsub "$PBSFILE")
                    echo "$JOBID  $JOBNAME"
                fi
            done
        done
    done
done

echo ""
echo "Submitted $COUNT jobs total."
