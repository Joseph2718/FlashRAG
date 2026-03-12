# PL-Rank + OptiSet Plan Notes

This note is a working reference for the idea of applying `PL-Rank-1` to the OptiSet setting by reducing a setwise evidence-selection problem to a finite listwise ranking problem over sampled candidate sets.

It is intentionally brief and practical. The goal is to preserve the main methodological decisions, novelty claim, and immediate next checks while the notebook evolves.

## Core Claim

The key bridge is:

- In the personal notes, the original formulation is a setwise problem: for each query `q`, choose an evidence set `S` from a candidate pool `D_q`, and evaluate it with a set utility `u(S; q)`.
- In OptiSet Algorithm 2, the model is not optimizing over all possible subsets directly. Instead, for each training instance it is given a finite sampled set list `S = {s_1, ..., s_m}` with quality signals `P`, and it learns a distribution over those sampled sets.
- That means the training problem has already been reduced from "search over all sets" to "rank items in a finite list," where each item is itself a set.

This is the main methodological opening for the project:

- Treat each candidate set as one ranked item.
- Score each set with a set utility or transformed preference score.
- Apply listwise learning machinery over the sampled set list.
- Investigate whether `PL-Rank-1` can replace or improve upon OptiSet's current listwise training objective in this reduced problem.

## Why This Matters

This gives a clean novelty statement:

- OptiSet shows how to reduce setwise evidence selection to a finite listwise ranking problem over sampled sets.
- PL-Rank was designed for efficient gradient estimation under Plackett-Luce ranking models.
- Therefore, the project can ask whether `PL-Rank-1` is a better optimization method for OptiSet-style set-listwise training than the current softmax/KL training scheme, especially in terms of variance, sample efficiency, and downstream evidence quality.

Put differently:

- OptiSet provides the reduction.
- PL-Rank provides the estimator family.
- The thesis contribution is to connect them empirically and methodologically.

## Important Distinction

There are two different "PL-Rank for set selection" stories, and they should not be conflated.

### Story A: Passage-ranking induces a set

- Rank passages with a PL policy over documents.
- Take the top-`K` prefix as a set.
- Evaluate the resulting set with `u(S(y); q)`.

This is what the current notebook mostly does.

### Story B: Sets are the ranked items

- Build a finite candidate family `S_q = {S_1, ..., S_m}`.
- Treat each `S_i` as one item in a list.
- Learn a PL model over sets, not over passages.

This is the interpretation most aligned with OptiSet Algorithm 2 and with the "reducing setwise to listwise" thesis framing.

For the thesis, Story B should be the main novelty framing. Story A remains useful as a debugging baseline and as a bridge from classical LTR / RL estimators.

## What OptiSet Algorithm 2 Is Doing

OptiSet Algorithm 2:

- takes training tuples `(q, D, S, P)`,
- where `S` is a sampled candidate set list,
- and `P` is a quality signal over those sets,
- then trains a model with:
  - a cross-entropy term toward the best sampled set,
  - plus a KL term between a target softmax over set quality scores and the model's softmax over set log-probabilities.

So Algorithm 2 already defines:

- a finite training list of sets,
- a target distribution over those sets,
- and a model distribution over those sets.

That is the exact point where a PL-ranking view over sets becomes natural.

## Thesis-Safe Novelty Statement

A careful version of the novelty claim is:

"We reinterpret OptiSet's sampled candidate-set training procedure as a finite listwise ranking problem over sets, and study whether PL-Rank-1 can serve as an alternative optimization method for this setting. This connects setwise evidence selection in RAG with Plackett-Luce learning-to-rank and policy-gradient variance reduction."

This wording is safer than claiming:

- that OptiSet itself uses PL-Rank,
- or that PL-Rank automatically applies to arbitrary set utility without adaptation.

## Current Notebook Insight

From the current notebook:

- `PL-Rank-1` already appears to reduce variance relative to naive `REINFORCE`.
- `PG-RANK (Gumbel+RLOO)` appears lower-variance than `PL-Rank-1` in the current HotpotQA nDCG reranking setup.
- That does not necessarily mean `PL-Rank-1` is wrong.
- It may simply mean that the current experiment favors a strong multi-sample leave-one-out baseline.

Also important:

- the current Gumbel ranking samplers end in `argsort`,
- so changing `tau` does not currently change the discrete sampled ranking order,
- which means the notebook, as written, cannot really test the hypothesis that lower Gumbel temperature itself reduces ranking variance.

## Immediate Research Questions

The current working RQs should be:

1. In a fixed-candidate HotpotQA reranking setting, how do `REINFORCE`, `PG-RANK`, and `PL-Rank-1` compare in gradient variance and sample efficiency?
2. After reducing setwise evidence selection to a listwise ranking problem over sampled candidate sets, can `PL-Rank-1` optimize the set-ranking problem effectively?
3. Does this optimization improve evidence quality, redundancy reduction, or downstream answer quality relative to OptiSet's original set-listwise objective?

## Recommended Experimental Ladder

Do not jump directly to the full OptiSet codepath. Build upward in layers.

### Phase 1: Debug estimator behavior in the current notebook

Use fixed BM25 candidates and HotpotQA labels.

Goals:

- verify variance plots,
- verify gradient-direction agreement,
- verify whether `PL-Rank-1` is behaving sensibly,
- verify that current Gumbel temperature does not affect sampled rankings.

### Phase 2: Toy setwise experiment without an LLM

Construct a finite candidate set list from BM25 top-`D`.

Define a simple set utility such as:

- coverage of gold titles / supporting sentences,
- minus redundancy penalty,
- optionally plus a compactness reward.

Example:

`U(S; q) = coverage(S; q) - lambda * redundancy(S) - mu * max(0, |S| - K)`

Goals:

- test the setwise-to-listwise reduction cleanly,
- avoid expensive generator calls,
- study `PL-Rank-1` over sets before moving to generator-based utility.

### Phase 3: OptiSet-style sampled-set training

Use:

- sampled candidate sets per query,
- utility signals derived from generator log-perplexity or answer quality,
- OptiSet-style target distributions over sets.

Then compare:

- original OptiSet softmax/KL objective,
- PL-over-sets objective with `PL-Rank-1`,
- possibly a REINFORCE-over-sets baseline.

## What To Reuse From FlashRAG

FlashRAG is a good scaffold for the fixed-candidate setup:

- `examples/methods/run_exp.py` already has a BM25 path.
- `examples/methods/ndcg_hotpotqa.py` already provides HotpotQA-specific nDCG helpers.
- The current notebook already maps BEIR / BM25 candidates into tensors suitable for controlled estimator experiments.

This means the project can stay grounded in one reproducible retrieval stack while gradually adding setwise modeling.

## What To Reuse From OptiSet

When the OptiSet code zip is available, the most relevant pieces to inspect first will likely be:

- candidate-set construction,
- set serialization / formatting for the selector,
- quality-score construction for sampled sets,
- Algorithm 2 training loss implementation,
- data structures for `(q, D, S, P)`.

The goal is not to merge entire repos blindly. The goal is to extract:

- how sets are represented,
- how candidate-set lists are built,
- how labels are stored,
- and where a PL-based alternative objective could be inserted.

## Oosterhuis Model Note

`/Users/josephop/2021-SIGIR-plackett-luce-1/utils/nnmodel.py` is worth keeping in mind as a future model reference, but it is not the urgent next step.

Right now it appears to be:

- a small TensorFlow MLP initializer,
- useful as a clue about the original codebase's model family,
- but not a drop-in replacement for the current PyTorch `TinyListwiseRanker`.

So the recommended order is:

1. first stabilize the estimator diagnostics with the current tiny reranker,
2. then, if needed, replace the tiny scorer with a slightly richer PyTorch scorer inspired by the Oosterhuis setup,
3. only after that consider porting broader code patterns from the original PL-Rank repository.

In other words, `nnmodel.py` is a good future upgrade reference, not an immediate dependency.

## Key Caveat

`PL-Rank-1` is naturally suited to PL ranking problems. If the set utility is arbitrary and non-decomposable, the estimator may need adaptation depending on how the ranking over sets is parameterized.

So the central methodological question is not:

"Can we directly drop PL-Rank into any set utility?"

It is:

"Once candidate sets are treated as ranked items under a PL model, does PL-Rank-1 offer a better optimization route than the current OptiSet listwise objective?"

That is the correct scientific question.

## Short Action List

1. Run the new notebook diagnostics that were added for:
   - Gumbel `tau` invariance,
   - variance ratios,
   - mean-gradient cosine alignment.
2. Add a toy sampled-set experiment in the notebook where sets, not passages, are the ranked items.
3. When needed, inspect the OptiSet zip and identify the exact training/data files around Algorithm 2.
4. Decide whether the first PL-over-sets comparison should use:
   - a toy coverage-redundancy utility,
   - or OptiSet's generator-based utility directly.

## Reminder To Future Me

- Keep the fixed-candidate regime explicit.
- Separate passage-ranking experiments from set-ranking experiments.
- Do not claim temperature effects from code paths that end in rank-preserving `argsort`.
- The main thesis value is the bridge: OptiSet reduction + PL-Rank estimator + empirical evaluation.
