# Chapter 5: RNNs vs Transformers

[← Seq2Seq & Attention](./04-seq2seq-attention.md) | [Back to README](../README.md)

---

## The Shift in NLP

Since the 2017 paper *"Attention Is All You Need"* (Vaswani et al.), **Transformers** have replaced RNNs as the dominant architecture for most large-scale NLP tasks, powering models like BERT, GPT, and T5.

## Key Differences

| Aspect | RNN / LSTM / GRU | Transformer |
|---|---|---|
| Core mechanism | Recurrence (sequential hidden state) | Self-attention (parallel, all positions at once) |
| Parallelization | Poor — must process tokens in order | Excellent — all tokens processed simultaneously |
| Long-range dependencies | Moderate (LSTM/GRU help, but still limited) | Strong — direct connections between any two positions |
| Training speed on long sequences | Slow | Fast, given enough compute |
| Memory usage | Lower | Higher (attention is quadratic in sequence length) |
| Positional information | Implicit (via order of processing) | Must be added explicitly (positional encodings) |

## Why Self-Attention Wins for Long Sequences

In an RNN, information from an early token has to pass through *every* intermediate timestep to reach a later one — the "path length" between two tokens is proportional to their distance in the sequence. This is exactly why long-range dependencies are hard for RNNs, even with gating.

In a Transformer, every token can attend directly to every other token in a single step, regardless of distance — the path length is constant (`O(1)`). This is the central reason Transformers handle long-range context so much better.

## Does That Mean RNNs Are Obsolete?

Not entirely. RNNs are still relevant for:

- **Streaming / online prediction** — RNNs process one token at a time, so they're a natural fit for real-time systems (e.g. live speech transcription) where you don't have the whole sequence upfront.
- **Resource-constrained environments** — edge devices, mobile apps, embedded systems, where a full Transformer is too heavy.
- **Short sequences** — when sequences are short, the Transformer's parallelization advantage matters less, and RNNs can be simpler and cheaper.
- **Educational value** — understanding RNNs is still the clearest path to understanding *why* attention and Transformers were invented in the first place.

## Summary

| If you need... | Consider |
|---|---|
| State-of-the-art accuracy on large datasets | Transformer |
| Low latency streaming/online inference | RNN / LSTM / GRU |
| Small model size for edge devices | RNN / LSTM / GRU |
| Very long documents / long-range reasoning | Transformer |
| To learn the fundamentals of sequence modeling | Start with RNNs, then move to attention & Transformers |

---

[Back to README](../README.md)