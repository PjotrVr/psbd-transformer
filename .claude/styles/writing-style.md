# Writing Style

## Punctuation and formatting (all writing)

- Do not use semicolons in regular writing, summaries, reviews, or code comments. Semicolons are fine only where a programming language requires them syntactically
- Do not use em dashes in any writing
- Do not use arrow characters in regular writing
- Do not use hyphens as sentence punctuation (standard hyphenation inside compound words is fine)
- Numbers always in digit form, never alphabetical
- Precise, simple technical language preferred in specifications and system prompts
- Plans and options laid out before a final artifact is produced, so the preferred one can be picked first

## Technical writing (blog posts, research writing, passages)

Writing covers only programming, mathematics, machine learning, and adjacent technical topics.

### Audience

- Default assumption: the reader is already familiar with the field
- The intended audience is stated explicitly whenever it differs, and that statement overrides the default

### Vocabulary

- Simple English by default. Use a complex or precise word when it is genuinely the right word
- Definitions never go inside the passage itself. Instead, outside the passage, explain what the complex word means and why it was the right choice in that spot

### Sentence construction

- Do not use "it is not X, it is Y" contrastive constructions. They read as LLM written. Cut the negated half and state the claim directly. Example of the fix: "A network that fits its training data worse than the shallower version of itself is failing to optimize" rather than "... is not memorizing too much, it is failing to optimize"
- No editorializing filler after a code block or section, e.g. "The block is short because the idea is short." Let the content stand and move on

### Opening

- Default: go straight into the problem, with some explanation of why the work is being done
- Research papers usually open with motivation instead. This is a judgment call, but straight into the problem is the default

### Voice

- First person by default
- Switch to the research register (passive voice, or "we") when asked, or when the piece is for a research paper

### Paragraphs

- One topic per paragraph, strictly
- A paragraph is never a single sentence
- Length varies with what the topic needs, longer or shorter is fine
- Structure: topic sentence at the beginning, supporting sentences in the middle, concluding sentence at the end

### Structure

- Bullet lists are allowed when they are genuinely the better choice
- Headers: specific noun phrase by default (e.g. "Variance growth in high dimensions", "Deriving the constant"), naming the subject of the section without asserting a claim about it
- Bare label headers (Introduction, Method, Experiments, Discussion) for research papers, where the venue dictates the format anyway
- Claim/full sentence headers are a specific exception, not a rule, and are not used by default
- Question headers and imperative headers are not wanted. Imperative reads like a recipe, which is rarely what is being written
- Medium style tutorials are not the goal. The writing is technical analysis, ideas, and similar work
- Header depth: 2 levels maximum, never more
- No default length target. A piece runs as long as the scope requires. The scope and what needs covering is stated up front, and lacking sections should be flagged. Length is too subjective to fix in advance, so no word count is requested or imposed

### Hedging

- Light hedging, direct by default. Heavier qualification only for research papers
- When hedging is used, the reason for it is explained outside the piece, never inside it

### Code and formulas inside prose

- Default: state the idea and let the reader work through the block themselves
- Explain the block line by line only when it is genuinely novel or convoluted, or when the stated audience would plausibly not follow it

### Failed attempts

- Write about the final approach, plus obvious failed attempts where they are useful
- Do not produce an exhaustive list of everything that did not work

### Revisions

- Targeted edits by default: return only the changed paragraphs with an explanation, not the whole piece
- Full rewrites only when requested

### Endings

- Stop when the content is finished. No summary or wrap up by default
- A conclusion can be added on request when the piece actually needs one

## Cross-cutting principle

- Commentary about the writing (why a word was chosen, why hedging was used) belongs in the answer, never inside the passage. The passage stays clean and the reasoning is delivered separately
