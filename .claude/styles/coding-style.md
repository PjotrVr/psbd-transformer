# Coding style

## All code

- Comments explain why only, never what the syntax already shows.
- `uv` for Python environments and dependencies, always, pinned.
- PyTorch Lightning `seed_everything` for seeding, always, in all code and not just research. The module is named `lightning`, not `pytorch_lightning`, so the import is `from lightning import seed_everything`.
- Blank lines separate logical phases in every tier and style.
- Tensor ops carry shape annotations (tuple form or readable form, either is fine, but they must be present).
- Shape assertions are a must where a mismatch would be silent.
- Side effects (I/O, networking, state mutation) isolated into dedicated functions.
- `return` never contains logic: assign the final computed value to a named variable and return that name. Literals, constants, existing variables, and simple conditional expressions are fine bare. Applies to recursion too.
- Early returns and guard clauses are allowed and encouraged.
- No decorative banners, dividers, or placeholder comments.
- No trivial one-line wrapper functions, inline the operation directly.
- Name length is context dependent: `temp` is fine when temperature is unambiguous in that file, spelled out when it isn't.
- Python tooling: `ruff` for lint and format (defaults: line length 88, indent 4, double quotes).
- Import style: `import torch.nn as nn` form, naming the full path so symbol origin is visible while reading. No wildcard imports unless wildcard is the established convention for that library.
- Imports are always shown in a code block, in every style including concise. Never omit them.
- Never invent a nonstandard import alias. Use the language's established alias (`import torch.nn.functional as F`, not something made up like `nnf`).
- When a paper symbol collides with a standard import alias, the standard alias wins. Rename the paper symbol and note the rename in the glossary comment.
- Style defaults: concise style is the default for code appearing inside writing (passages, blog posts, technical analysis). Descriptive style is the default when producing actual code as the deliverable.

## Production code

- Type hints on signatures.
- Docstrings on public functions, classes, modules, stating contract not implementation.
- `main()` entry point and standard project layout.
- Separation of concerns, clear module responsibility boundaries.
- Error handling and input validation at system boundaries.
- `argparse` for configuration and CLI args.
- Testability as a design constraint.
- Always descriptive style. Concise style never applies to production code.

## Research and experimentation code

- Optimized for edit speed and comprehension, not extension or reuse.
- Everything visible in as few files as possible, no chasing definitions across many modules.
- Flat structure: no class hierarchies, no dependency injection, no config frameworks.
- No docstrings, no type hints.
- Modular means swappable blocks, not layered abstraction.
- Top level flow reads sequentially, helper functions defined after the main flow so the file reads top down as narrative.
- Hyperparameters and constants at module level by default, stated otherwise when something else is wanted.
- Let failures crash loudly, no try/except unless the failure mode is specific and expected, nothing overengineered.
- Experiment tracking only when asked, and then wandb or plain files.
- Jupyter notebooks are a first-class brainstorming target and follow all research rules, never production code.

## Descriptive style (default for actual code deliverables)

- Names carry meaning at point of use, no glossary needed.
- Prefix conventions group related quantities (`student_logits`, `student_temp`, `teacher_logits`, `teacher_temp`).
- Every value a function needs is passed as a parameter, nothing captured from enclosing scope, except genuine module level globals.
- Each conceptual step gets its own named intermediate variable even when inlining is possible.
- Comments sit above the block they explain, full sentences, stating the reason.
- Helper functions get real, meaningful names.

## Concise style (default for code inside writing)

- Scope: code with a mathematical, algorithmic, or physical source (formulas, numerical methods, simulations, signal processing, crypto primitives, probability and statistics, classical algorithms with textbook notation). Not for I/O, data loading, config, orchestration, or glue code.
- Names are mathematical notation turned into code, mirroring the source symbols so code maps line by line onto the paper.
- Glossary comment block at the top only when names are opaque, skipped when self-evident.
- Paired or indexed quantities share a stem and differ by numeric suffix.
- Trailing comments short and descriptive rather than full sentences, as long as they need to be and no longer.
- Expressions stay inline, tuple unpacking puts related assignments on one line.
- Enclosing-scope values may be referenced directly rather than threaded as parameters, when genuinely global to the file.
- Arithmetic mirrors the source formula rather than being refactored into something more idiomatic.

## Worked examples

These examples define the styles, not the rules above.

### Adam update step

Concise:

```python
# m, v: first and second raw moment estimates
# b1, b2: exponential decay rates for the moments
# a: step size, eps: numerical stability constant
# t: step counter
for grad in grads:
    t += 1
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2

    # moments start at zero, biased low early on
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)

    params -= a * m_hat / (sqrt(v_hat) + eps)
```

Descriptive:

```python
for grad in grads:
    step += 1

    first_moment = beta1 * first_moment + (1 - beta1) * grad
    second_moment = beta2 * second_moment + (1 - beta2) * grad ** 2

    # Both moments are initialized at zero, which biases the early estimates
    # toward zero. Dividing by the decay factor removes that bias.
    corrected_first = first_moment / (1 - beta1 ** step)
    corrected_second = second_moment / (1 - beta2 ** step)

    update_direction = corrected_first / (sqrt(corrected_second) + eps)
    params -= learning_rate * update_direction
```

### Multi head attention

Concise:

```python
# B, T, C: batch, sequence length, embedding dim
# h: head count, hd: per head dim (C // h)
def attn(x, wq, wk, wv, wo, h):
    B, T, C = x.shape
    assert C % h == 0
    hd = C // h

    q = (x @ wq).view(B, T, h, hd).transpose(1, 2)  # B, h, T, hd
    k = (x @ wk).view(B, T, h, hd).transpose(1, 2)
    v = (x @ wv).view(B, T, h, hd).transpose(1, 2)

    # scale by sqrt(hd) or the logits grow with dim and saturate softmax
    a = softmax(q @ k.transpose(-2, -1) / sqrt(hd), dim=-1)  # B, h, T, T

    y = (a @ v).transpose(1, 2).reshape(B, T, C)
    out = y @ wo
    return out
```

Descriptive:

```python
def multi_head_attention(x, w_query, w_key, w_value, w_output, num_heads):
    batch, seq_len, embed_dim = x.shape
    assert embed_dim % num_heads == 0

    head_dim = embed_dim // num_heads

    # Each head attends inside its own subspace, so the embedding is split
    # across heads rather than duplicated.
    queries = (x @ w_query).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    keys = (x @ w_key).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    values = (x @ w_value).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    scores = queries @ keys.transpose(-2, -1)  # (batch, num_heads, seq_len, seq_len)

    # Without the scale, dot products grow with head_dim and push softmax into
    # a region where gradients vanish.
    att_weights = softmax(scores / sqrt(head_dim), dim=-1)

    weighted_values = att_weights @ values
    merged_heads = weighted_values.transpose(1, 2).reshape(batch, seq_len, embed_dim)
    out = merged_heads @ w_output
    return out
```

### Training loop

Concise:

```python
# dl: dataloader, opt: optimizer, sched: lr schedule
# dev: compute device
seed_everything(42)
for ep in range(n_ep):
    net.train()
    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)

        loss = criterion(net(x), y)
        loss.backward()

        # clip after backward, before step, since grads only exist in that window
        clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

    sched.step()

    net.eval()
    with no_grad():
        val_loss = sum(criterion(net(x.to(dev)), y.to(dev)) for x, y in val_dl) / len(val_dl)
```

Descriptive:

```python
seed_everything(42)
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()

        # Clipping has to happen after backward and before step, because the
        # gradients only exist during that window.
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    model.eval()
    with no_grad():
        batch_losses = [criterion(model(x.to(device)), y.to(device)) for x, y in val_loader]
        mean_val_loss = sum(batch_losses) / len(batch_losses)
```

### Parallel reduction

Concise:

```python
# w: worker count, cs: chunk size, ps: per chunk partial results
def par_reduce(xs, f, w):
    cs = ceil(len(xs) / w)
    chunks = [xs[i:i + cs] for i in range(0, len(xs), cs)]

    # f must be associative, chunking regroups the operations
    with ThreadPoolExecutor(w) as ex:
        ps = list(ex.map(lambda c: reduce(f, c), chunks))

    out = reduce(f, ps)
    return out
```

Descriptive:

```python
def parallel_reduce(items, combine, num_workers):
    chunk_size = ceil(len(items) / num_workers)
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    # The combine function must be associative. Chunking changes how the
    # operations are grouped, so a non associative function would give a
    # different answer than a sequential reduce.
    with ThreadPoolExecutor(num_workers) as executor:
        partial_results = list(executor.map(lambda chunk: reduce(combine, chunk), chunks))

    final_result = reduce(combine, partial_results)
    return final_result
```
