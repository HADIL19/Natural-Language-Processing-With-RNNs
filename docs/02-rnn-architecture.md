# Chapter 2: RNN Architecture

[← Introduction](./01-introduction.md) | [Next: LSTM & GRU →](./03-lstm-gru.md)

---

## The Core Idea

A **Recurrent Neural Network** processes a sequence one element at a time, while maintaining a **hidden state** that acts as memory of everything seen so far. The same weights are reused at every timestep — this is what makes it "recurrent."

## The Math

At each timestep `t`, the RNN takes:
- Input `x_t` (e.g., the embedding of the current word)
- Previous hidden state `h_(t-1)`

And computes:

```
h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b_h)
y_t = W_hy · h_t + b_y
```

Where:
- `W_xh` — weights mapping input → hidden
- `W_hh` — weights mapping previous hidden state → current hidden state
- `W_hy` — weights mapping hidden state → output
- `b_h`, `b_y` — bias terms

The hidden state `h_t` is passed forward to the next timestep, carrying accumulated context.

## Unrolling Through Time

Although an RNN is often drawn as a single cell with a loop, during training it is "unrolled" across all timesteps in the sequence — effectively becoming a deep feedforward network where each "layer" corresponds to one timestep, and all layers share the same weights.

## Types of RNN Input/Output Structures

| Type | Structure | Example |
|---|---|---|
| One-to-One | single input → single output | standard classifier (not sequential) |
| One-to-Many | single input → sequence output | image captioning |
| Many-to-One | sequence input → single output | sentiment analysis |
| Many-to-Many (aligned) | sequence in → sequence out, same length | POS tagging |
| Many-to-Many (unaligned) | sequence in → sequence out, different length | machine translation |

## Training: Backpropagation Through Time (BPTT)

RNNs are trained using **BPTT**: gradients are computed by unrolling the network across timesteps and propagating error backward through each one. Because the same weights are reused at every step, gradients from all timesteps accumulate.

This is also the source of RNNs' biggest weakness — see the next chapter.

## Limitation: Vanishing & Exploding Gradients

When gradients are repeatedly multiplied across many timesteps:
- If values are consistently < 1, gradients shrink toward zero (**vanishing gradients**) — the network "forgets" long-range dependencies.
- If values are consistently > 1, gradients grow explosively (**exploding gradients**) — training becomes unstable.

**Exploding gradients** are commonly mitigated with **gradient clipping**. **Vanishing gradients** are the main motivation for LSTM and GRU architectures, covered next.

---

[Next: LSTM & GRU →](./03-lstm-gru.md)