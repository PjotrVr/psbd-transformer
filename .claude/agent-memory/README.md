# agent-memory

Durable findings from `sweep-monitor`, `sweep-analyst` and `leak-auditor`, carried across sessions
because a later session cannot rediscover them from the artifacts alone.

1 file per finding. Name it after the finding, not after the run or the date. The first line says
what the finding is in 1 sentence, and the body says how it was established well enough that a
future session can decide whether it still holds.

## What belongs here

- A structural fact about the sweep that survives the run that revealed it, such as an axis whose
  effect is entirely inside the seed noise floor.
- A leak class that was found and fixed, and what the check that catches it looks like now.
- A failure mode that recurs across stages and its cause, such as a config that reliably runs out of
  memory at a given environment count.
- A measurement that changes how later numbers are read, such as the exploitability ceiling Stage 0
  established, or the fitted calibration constant.
- A correction to something a previous session wrote down here.

## What does not belong here

- Anything derivable from `logs/`, `runs/` or `checkpoints/` by reading them. Re-read the artifact.
- This run's numbers. A win rate belongs in the run's own trajectory and in the evaluator JSON,
  which name the commits that produced it. A copy here goes stale and gets quoted anyway.
- Rankings and stage reports. Those are output, they go to the human who asked for them.
- Anything already written in `docs/`, `README.md` or the plan. If it belongs there, say so and let
  the owning agent put it there.
