# Chapter 3: LSTM and GRU

[← RNN Architecture](./02-rnn-architecture.md) | [Next: Seq2Seq & Attention →](./04-seq2seq-attention.md)

---

## Why We Need Gated RNNs

Vanilla RNNs struggle to retain information across long sequences because of vanishing gradients (see Chapter 2). **LSTM** and **GRU** solve this with gating mechanisms that let the network learn *what to remember and what to forget*.

## Long Short-Term Memory (LSTM)

Introduced by Hochreiter & Schmidhuber (1997). LSTM introduces a separate **cell state** (`C_t`) that runs through the sequence largely unchanged, plus three gates that control information flow:

| Gate | Purpose |
|---|---|
| **Forget gate** (`f_t`) | Decides what to discard from the cell state |
| **Input gate** (`i_t`) | Decides what new information to add |
| **Output gate** (`o_t`) | Decides what part of the cell state to expose as output |

**Simplified equations:**

```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)
i_t = σ(W_i · [h_(t-1), x_t] + b_i)
o_t = σ(W_o · [h_(t-1), x_t] + b_o)
C̃_t = tanh(W_c · [h_(t-1), x_t] + b_c)

C_t = f_t * C_(t-1) + i_t * C̃_t
h_t = o_t * tanh(C_t)
```

The gates use sigmoid activations (`σ`), producing values between 0 and 1 that act as "how much to let through."

## Gated Recurrent Unit (GRU)

Introduced by Cho et al. (2014) as a simplified alternative to LSTM. GRU merges the cell state and hidden state, and uses only two gates:

| Gate | Purpose |
|---|---|
| **Reset gate** (`r_t`) | Controls how much past information to forget |
| **Update gate** (`z_t`) | Controls how much of the new candidate state to use |

```
r_t = σ(W_r · [h_(t-1), x_t] + b_r)
z_t = σ(W_z · [h_(t-1), x_t] + b_z)
h̃_t = tanh(W_h · [r_t * h_(t-1), x_t] + b_h)

h_t = (1 - z_t) * h_(t-1) + z_t * h̃_t
```

## Comparing the Three

| Feature | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| Gates | 0 | 3 | 2 |
| Separate cell state | No | Yes | No |
| Long-term memory | Poor | Strong | Strong |
| Parameters | Fewest | Most | Moderate |
| Training speed | Fast | Slower | Faster than LSTM |
| Typical use | Short sequences, toy problems | Long sequences, high accuracy needs | Long sequences, faster training |

## Practical Guidance

- Start with **GRU** for faster iteration and fewer parameters.
- Switch to **LSTM** if you need to model longer or more complex dependencies and have the compute budget.
- Both are usually used with multiple layers stacked, and often bidirectionally (processing the sequence forward *and* backward) for tasks where future context matters (e.g., NER, POS tagging).

---

[Next: Seq2Seq & Attention →](./04-seq2seq-attention.md)