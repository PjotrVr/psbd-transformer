# Math style

## All mathematics

- Original (source) notation is the default. Descriptive form is produced only on request. This is the reverse of the code default, where descriptive is the default.
- A symbol table always accompanies a formula, regardless of how many symbols are involved.
- Exception: symbol tables are for personal reference, not for publication. When writing an official or published text (blog post, paper, anything for an audience), omit the symbol table and define symbols in prose instead.
- Every symbol defined at the point the formula is introduced.
- Notation consistency: once a name or convention is chosen, use it for the rest of the response or project, no drifting back.
- Numbers in digit form always, never alphabetical.
- Citations name at most 2 authors. With more than 2, write "He et al." style, never a full author list.
- Math delimiters depend on the target. Two different renderers, two different sets of correct rules.

### Chat output (KaTeX inside Markdown), the default

- `$...$` inline, `$$...$$` block, nothing else. No `\[`, `\(`, `\begin{equation}`, `\begin{align}`.
- Use `\begin{aligned}` inside `$$...$$` when alignment is needed. KaTeX has no counters, so `equation` cannot work and `align` support is unreliable.
- Verify every `$` and `$$` pair opens and closes before sending, and that no stray underscore or asterisk outside math could be read as Markdown. Re-render mentally once when a response has many formulas.
- No preamble exists, so no `\newcommand` and no package-specific commands.

### `.tex` files (real LaTeX, compiled)

- These rules replace the chat rules whenever the deliverable is an actual `.tex` file, including a study mode artifact.
- Do not use `$$...$$`. It is plain TeX, ignores the `fleqn` class option, and gives inconsistent vertical spacing.
- Display math uses `\[...\]`, or `equation` / `align` with amsmath loaded. Inline math uses `$...$` or `\(...\)`.
- Do not use `aligned` as a top level display. It is meant to nest inside another display environment. Use `align` or `split` instead.
- Preamble, packages (amsmath, amssymb, mathtools), `\newcommand` macros, `\label` and `\ref` cross-references are all available and should be used normally.
- Markdown does not exist in a `.tex` file, so the stray-underscore check does not apply.
- Assumptions and requirements stated explicitly before the work starts, not discovered midway.
- All numeric results computed via code execution and then reported, never mental arithmetic presented as exact.
- Result stated first, derivation after it.

## Conventions

- Vector/matrix layout (column vs row, numerator vs denominator layout): pick whichever fits the situation, but state the choice explicitly.
- Matrix form is the default presentation. Explicit indices are the second option, used when matrix form might be hard to follow.
- No dimensional analysis, limiting-case checks, or degenerate-input sanity checks unless requested.
- No complexity, memory footprint, or convergence-rate annotations unless requested.

## Original (default) vs descriptive (on request) notation

- Original form is the default and reproduces the source exactly (Greek letters, sub/superscript conventions), kept even when convoluted, so it can be matched line by line against the paper. Always followed by a symbol table.
- Descriptive form, produced on request only, replaces Greek and single letters with names saying what the quantity is. Subscripts kept only when they carry real indexing meaning (time step, sample index) and dropped when decoration. Expression structure left intact (renaming, not rederivation).
- Motivation: many papers use convoluted notation on purpose, and clean notation is preferred where it's not required to match the source.

## Derivations

- Every step shown, no jumping.
- Non-obvious steps name the justifying rule (chain rule, Jensen, integration by parts, change of variables, Bayes).
- Steps only valid under a condition state that condition rather than assuming silently (e.g. swapping gradient and integral).
- Approximations flagged where they enter, with a note on what was dropped.
- Derivation itself is symbolic, numbers arrive at the end, computed.
- Matches the intended workflow: derivations by hand, calculations by computer.
- Derivations must be verified, by whatever means is appropriate (symbolic check, numerical check, independent rederivation). Verification is not optional, the method is a judgment call.

## Citations and conflicting conventions

- Never state authors, venue, or year unless verified through search. Say so explicitly when unsure rather than guessing.
- Established results from the literature kept clearly separate from novel claims or proposals.
- When a formula comes from a specific paper, name the paper and use its notation for the original form.
- When papers use conflicting conventions for the same object: pick one, say which one is being used, and name the other along with the paper it comes from. Example phrasing: using the $f(x) = ax + c$ form here, while $y = ax + b$ comes from paper Z and is not used.
- When reading 2 or 3 papers at once and all of them have bad notation, inventing a new mixed notation is a valid and often preferred option. Take the best pieces from each source, or introduce fresh symbols, and say which piece came from where so it can still be matched back to each paper.

## Uncertainty

- Flag inferred or extrapolated claims separately from claims backed by a specific paper, computation, or stated result.
- Never present speculation with the confidence of verified fact.

## Study mode

Study mode ("zero to hero") has its own separate guide. Exercises and the zero to hero opening belong there, not to ordinary math answers.

## Worked examples

These examples define the format, not the rules above.

### Default form: original notation, then symbol table

Adam, as the paper writes it.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $m_t$ | first raw moment estimate |
| $v_t$ | second raw moment estimate |
| $g_t$ | gradient at step $t$ |
| $\beta_1, \beta_2$ | exponential decay rates for the moments |
| $\alpha$ | step size |
| $\epsilon$ | numerical stability constant |
| $\theta_t$ | parameters at step $t$ |

Descriptive form, only produced on request:

$$
\begin{aligned}
\text{first}_t &= \text{decay}_1 \cdot \text{first}_{t-1} + (1 - \text{decay}_1) \cdot \text{grad}_t \\
\widehat{\text{first}}_t &= \frac{\text{first}_t}{1 - \text{decay}_1^t} \\
\text{params}_t &= \text{params}_{t-1} - \text{lr} \cdot \frac{\widehat{\text{first}}_t}{\sqrt{\widehat{\text{second}}_t} + \text{eps}}
\end{aligned}
$$

### Result first, then assumptions, then derivation

The log derivative trick.

Result:

$$
\nabla_\theta \mathbb{E}_{p_\theta(x)}\left[f(x)\right] = \mathbb{E}_{p_\theta(x)}\left[f(x) \nabla_\theta \log p_\theta(x)\right]
$$

Assumptions, stated before the derivation rather than found inside it:

1. $p_\theta(x)$ is differentiable with respect to $\theta$.
2. The support of $p_\theta$ does not depend on $\theta$.
3. The integrand is dominated, so gradient and integral can be exchanged.
4. $p_\theta(x) > 0$ wherever the integrand is non-zero.
5. $f$ does not depend on $\theta$.

Derivation:

$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{p_\theta(x)}\left[f(x)\right] &= \nabla_\theta \int p_\theta(x) f(x) \, dx && \text{definition of expectation} \\
&= \int \nabla_\theta p_\theta(x) \, f(x) \, dx && \text{assumption 3, exchange} \\
&= \int p_\theta(x) \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} f(x) \, dx && \text{multiply and divide, assumption 4} \\
&= \int p_\theta(x) \nabla_\theta \log p_\theta(x) \, f(x) \, dx && \text{chain rule on } \log \\
&= \mathbb{E}_{p_\theta(x)}\left[f(x) \nabla_\theta \log p_\theta(x)\right] && \text{fold back into expectation}
\end{aligned}
$$
