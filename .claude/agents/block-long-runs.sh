#!/bin/bash
# Backstop for ml-builder. The system prompt already forbids launching real
# training, but a prompt is a request and a hook is a rule. Blocks anything
# that would burn cluster time or write to a real experiment tracker.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  exit 0
fi

# Cluster submission and long-running launchers
if echo "$COMMAND" | grep -qE '\b(sbatch|srun|salloc|squeue|torchrun|deepspeed|accelerate launch|nohup)\b'; then
  echo "Blocked: ml-builder may not launch training jobs. Hand off to ml-test-automator." >&2
  exit 2
fi

# Online experiment tracking would pollute the real project with smoke runs
if echo "$COMMAND" | grep -qE 'WANDB_MODE=online|wandb (login|sync)'; then
  echo "Blocked: use WANDB_MODE=offline for smoke tests." >&2
  exit 2
fi

# Results and checkpoints are produced by real runs and must never be touched here
if echo "$COMMAND" | grep -qE '\brm\b.*(checkpoints|results|artifacts|outputs)'; then
  echo "Blocked: ml-builder may not delete run artifacts." >&2
  exit 2
fi

exit 0
