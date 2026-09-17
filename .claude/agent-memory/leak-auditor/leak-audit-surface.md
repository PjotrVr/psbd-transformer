---
name: leak-audit-surface
description: Where the privileged-information boundary lives in ek_solver, plus the three places privileged data is deliberately allowed and why each is contained
metadata:
  type: project
---

The leak boundary in ek_solver is enforced by types and by crate layout, not by convention:
`ek-obs` depends only on `ek-core` and every public function in it takes `&View`. The only route from
a `View` to a `State` is `determinize`, which invents the hidden half.

**Why:** the game project this engine is vendored from shipped a cheating bot twice and had to retire
every rating taken before each fix. The failure is silent — self-play looks excellent because both
seats cheat identically.

**How to apply:** when auditing, the fast path is to confirm the crate boundary still holds
(`grep -n 'State\|card_at(' engine/crates/ek-obs/src/*.rs` should hit only comments and `#[cfg(test)]`),
then check the three sanctioned exceptions below rather than re-reading every encoder.

Three places privileged information is deliberately allowed, each contained differently:

1. **`ek-env/src/targets.rs`** — ground truth for the two belief heads. Contained by living outside
   `ek-obs` and by riding in its own buffer (`vec.rs`, separate width, separate rayon leg), never
   concatenated into `observations`.
2. **The oracle critic** (`solver/dmc/nets.py` `oracle_value_logits`) — takes `targets` as an explicit
   argument over a detached trunk. Contained because no play-path method has an argument it could
   arrive through.
3. **`--oracle-features`** (`solver/dmc/selfplay.observations_of`) — a genuine leak, a perfect-information
   ceiling probe. Contained by `DMCAgent.__init__` raising `SystemExit` and by `describe()` marking the
   row `servable: False`.

The one asymmetry worth re-checking on every audit: the belief heads detach the trunk, but `aux_head`
deliberately does **not**, and one of its four labels (`opponent_runway`) is pooled opponent-hand
ground truth. Label side only, so not a leak — but it is the boundary's softest edge.

Related: [[leak-audit-test-gaps]]
