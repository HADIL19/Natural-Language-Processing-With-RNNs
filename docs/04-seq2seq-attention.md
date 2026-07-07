# Chapter 4: Sequence-to-Sequence Models & Attention

[← LSTM & GRU](./03-lstm-gru.md) | [Next: RNNs vs Transformers →](./05-rnn-vs-transformers.md)

---

## The Problem: Different Input/Output Lengths

Many NLP tasks map a sequence to another sequence of a *different* length — machine translation, summarization, and question answering are classic examples. A single RNN handling many-to-many aligned tasks won't work here, because output length isn't tied to input length.

## Encoder-Decoder (Seq2Seq) Architecture

Introduced by Sutskever et al. (2014). The idea:

1. **Encoder RNN** reads the entire input sequence, producing a final hidden state — a fixed-size **context vector** summarizing the whole input.
2. **Decoder RNN** is initialized with that context vector and generates the output sequence one token at a time, feeding its own previous output back in as the next input.

```
Input:  "I love NLP"
Encoder: reads word by word → produces context vector
Decoder: context vector → "J'aime le NLP" (generated word by word)
```

## The Bottleneck Problem

Compressing an entire sentence (or paragraph) into a single fixed-size vector loses information, especially for long inputs. The decoder has to reconstruct everything from that one vector.

## Attention: Letting the Decoder Look Back

Introduced by Bahdanau et al. (2015). Instead of relying on just the final encoder hidden state, **attention** lets the decoder look at *all* encoder hidden states at every decoding step, and learn which ones matter most for producing the current output word.

**Conceptually:**
1. At each decoding step, compute an "alignment score" between the decoder's current state and every encoder hidden state.
2. Convert these scores into weights via softmax.
3. Take a weighted sum of encoder hidden states — the **context vector** for this step.
4. Use that context vector (along with the decoder's own state) to predict the next word.

This means the model can dynamically focus on relevant parts of the input — e.g., when translating "the black cat" the decoder can "attend" strongly to "cat" when generating the corresponding target word, regardless of position.

## Why This Matters Beyond RNNs

Attention was originally introduced to improve RNN-based Seq2Seq models, but it turned out to be so powerful that researchers eventually asked: *what if we removed the recurrence entirely and built a model purely out of attention?*

That question led directly to the **Transformer** architecture (Vaswani et al., 2017) — the subject of the next chapter.

---

[Next: RNNs vs Transformers →](./05-rnn-vs-transformers.md)