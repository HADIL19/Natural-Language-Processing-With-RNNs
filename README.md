# Natural Language Processing with RNNs

A complete guide to understanding, building, and applying Recurrent Neural Networks (RNNs) for Natural Language Processing (NLP) tasks.

---

## Table of Contents

1. [Overview](#overview)
2. [Why RNNs for NLP?](#why-rnns-for-nlp)
3. [Core Concepts](#core-concepts)
4. [RNN Architecture](#rnn-architecture)
5. [Types of RNNs](#types-of-rnns)
6. [RNN Variants: LSTM & GRU](#rnn-variants-lstm--gru)
7. [Common NLP Tasks Using RNNs](#common-nlp-tasks-using-rnns)
8. [Training an RNN](#training-an-rnn)
9. [Challenges & Limitations](#challenges--limitations)
10. [RNNs vs Transformers](#rnns-vs-transformers)
11. [Minimal Code Example](#minimal-code-example)
12. [Resources](#resources)
13. [Repo Structure & Hands-on Notebooks](#repo-structure--hands-on-notebooks)

---

## Overview

**Natural Language Processing (NLP)** is the field of AI concerned with enabling computers to understand, interpret, and generate human language. Because language is inherently **sequential** (word order matters, context builds over time), models used for NLP need a way to remember what came before.

**Recurrent Neural Networks (RNNs)** are a class of neural networks designed specifically for sequential data. Unlike standard feedforward networks, RNNs have loops that allow information to persist — a "memory" of previous inputs — making them a natural fit for text, speech, and time-series data.

---

## Why RNNs for NLP?

Traditional neural networks assume inputs are independent of one another. But in language:

- The meaning of a word often depends on the words before it ("bank" means something different in "river bank" vs "savings bank").
- Sentences have variable length.
- Order matters ("dog bites man" ≠ "man bites dog").

RNNs address this by processing text **one token at a time**, while maintaining a **hidden state** that carries information from earlier steps in the sequence forward.

---

## Core Concepts

| Concept | Description |
|---|---|
| **Sequence** | An ordered list of tokens (words, characters, subwords) |
| **Hidden state (h_t)** | A vector summarizing information seen so far in the sequence |
| **Timestep** | A single step in the sequence (e.g., one word) |
| **Recurrence** | The same weights are reused at every timestep |
| **Embedding** | A dense vector representation of a token (e.g., Word2Vec, GloVe, or learned embeddings) |
| **Backpropagation Through Time (BPTT)** | The algorithm used to train RNNs across timesteps |

---

## RNN Architecture

At each timestep `t`, an RNN takes:
- The current input `x_t` (e.g., an embedded word)
- The previous hidden state `h_(t-1)`

And produces:
- A new hidden state `h_t`
- Optionally, an output `y_t`

**Core equations:**

```
h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b_h)
y_t = W_hy · h_t + b_y
```

Where:
- `W_xh` = weights from input to hidden layer
- `W_hh` = weights from previous hidden state to current hidden state
- `W_hy` = weights from hidden state to output
- `b_h`, `b_y` = bias terms

The same weight matrices are shared across all timesteps — this is what makes the network "recurrent" and allows it to generalize across sequences of any length.

---

## Types of RNNs

RNNs are categorized by how inputs and outputs are structured:

| Type | Structure | Example Use Case |
|---|---|---|
| **One-to-One** | Single input → single output | Standard classification (not sequential) |
| **One-to-Many** | Single input → sequence output | Image captioning |
| **Many-to-One** | Sequence input → single output | Sentiment analysis |
| **Many-to-Many (aligned)** | Sequence input → sequence output, same length | Part-of-speech tagging |
| **Many-to-Many (unaligned)** | Sequence input → sequence output, different length | Machine translation (uses encoder-decoder) |

---

## RNN Variants: LSTM & GRU

Vanilla RNNs struggle to retain information over long sequences due to the **vanishing gradient problem**. Two popular architectures solve this:

### Long Short-Term Memory (LSTM)
Introduced by Hochreiter & Schmidhuber (1997). LSTMs add a **cell state** and three gates:
- **Forget gate** — decides what to discard from memory
- **Input gate** — decides what new information to store
- **Output gate** — decides what to output from memory

This gating mechanism allows LSTMs to preserve relevant information over long sequences.

### Gated Recurrent Unit (GRU)
Introduced by Cho et al. (2014). A simplified version of LSTM with only two gates:
- **Reset gate**
- **Update gate**

GRUs are computationally cheaper than LSTMs and often perform comparably.

| Feature | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| Gates | None | 3 | 2 |
| Long-term memory | Poor | Strong | Strong |
| Training speed | Fast | Slower | Faster than LSTM |
| Parameters | Fewest | Most | Moderate |

---

## Common NLP Tasks Using RNNs

- **Text classification** (sentiment analysis, spam detection)
- **Named Entity Recognition (NER)**
- **Part-of-speech tagging**
- **Language modeling** (predicting the next word)
- **Machine translation** (Seq2Seq / encoder-decoder models)
- **Text generation**
- **Speech recognition**
- **Text summarization**
- **Chatbots / dialogue systems**

### Encoder-Decoder (Seq2Seq) Architecture
Used for tasks where input and output sequences differ in length (e.g., translation):
- **Encoder RNN**: reads the input sequence and compresses it into a context vector (final hidden state).
- **Decoder RNN**: generates the output sequence, using the context vector as its starting hidden state.
- Often enhanced with an **attention mechanism**, which lets the decoder focus on relevant parts of the input at each output step rather than relying solely on a single fixed context vector.

---

## Training an RNN

1. **Tokenize** text into words/subwords/characters.
2. **Convert tokens to embeddings** (learned or pretrained like GloVe/Word2Vec).
3. **Feed sequence through the RNN**, updating hidden state at each timestep.
4. **Compute loss** (e.g., cross-entropy for classification/language modeling).
5. **Backpropagate Through Time (BPTT)** — gradients flow backward through every timestep.
6. **Update weights** using an optimizer (Adam, RMSprop, etc.).
7. Use techniques like **gradient clipping** to prevent exploding gradients.

---

## Challenges & Limitations

- **Vanishing/exploding gradients**: Gradients shrink or grow exponentially over long sequences, making it hard to learn long-range dependencies (mitigated by LSTM/GRU, gradient clipping).
- **Sequential computation**: RNNs process tokens one at a time, so they can't be parallelized across the sequence dimension — this makes training slow on long sequences.
- **Limited long-range context**: Even LSTMs/GRUs struggle with very long documents compared to attention-based models.
- **Fixed context bottleneck**: In basic Seq2Seq, compressing an entire input into one vector loses information (solved by attention).

---

## RNNs vs Transformers

Since 2017, **Transformers** (which rely entirely on attention mechanisms, no recurrence) have largely replaced RNNs as the state of the art for most NLP tasks.

| Aspect | RNN/LSTM/GRU | Transformer |
|---|---|---|
| Parallelization | Poor (sequential) | Excellent (parallel across sequence) |
| Long-range dependencies | Moderate (LSTM/GRU help) | Strong (via self-attention) |
| Training speed on long sequences | Slow | Fast (with enough compute) |
| Memory usage | Lower | Higher (quadratic in sequence length for standard attention) |
| Common today for | Small-scale/streaming tasks, time-series, edge devices | Most modern large-scale NLP (BERT, GPT, T5, etc.) |

RNNs are still useful for: lightweight applications, streaming/online prediction, embedded/edge devices, and cases where sequence length is short and compute is limited.

---

## Minimal Code Example

A simple text classification RNN using PyTorch:

```python
import torch
import torch.nn as nn

class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.rnn(embedded)
        final_hidden = hidden[-1]              # last layer's hidden state
        return self.fc(final_hidden)

# Example usage
model = SimpleRNNClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=2)
```

And in Keras/TensorFlow:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(256),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## Resources

### 📄 Foundational Papers
- Elman, J. (1990) — *Finding Structure in Time* (early RNN concept)
- Hochreiter, S. & Schmidhuber, J. (1997) — *Long Short-Term Memory* (LSTM paper)
- Cho, K. et al. (2014) — *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (GRU paper)
- Sutskever, I. et al. (2014) — *Sequence to Sequence Learning with Neural Networks*
- Bahdanau, D. et al. (2015) — *Neural Machine Translation by Jointly Learning to Align and Translate* (attention mechanism)
- Vaswani, A. et al. (2017) — *Attention Is All You Need* (Transformer paper, for comparison)

### 🎓 Courses
- Stanford **CS224n**: Natural Language Processing with Deep Learning
- DeepLearning.AI: **Sequence Models** (Coursera, part of the Deep Learning Specialization)
- Fast.ai: **Practical Deep Learning for Coders** (NLP sections)
- MIT **6.S191**: Introduction to Deep Learning

### 📚 Books
- *Speech and Language Processing* — Daniel Jurafsky & James H. Martin (free draft available online)
- *Deep Learning* — Ian Goodfellow, Yoshua Bengio, Aaron Courville
- *Natural Language Processing with PyTorch* — Delip Rao & Brian McMahan

### 🛠️ Libraries & Frameworks
- **PyTorch** — `torch.nn.RNN`, `torch.nn.LSTM`, `torch.nn.GRU`
- **TensorFlow / Keras** — `tf.keras.layers.SimpleRNN`, `LSTM`, `GRU`
- **spaCy** — industrial-strength NLP pipeline
- **NLTK** — classic NLP toolkit for preprocessing
- **Hugging Face Transformers** — for comparing RNN approaches with modern transformer models
- **Gensim** — for word embeddings (Word2Vec, FastText)

### 🌐 Tutorials & Blog Posts
- Christopher Olah — *Understanding LSTM Networks* (colah.github.io)
- Andrej Karpathy — *The Unreasonable Effectiveness of Recurrent Neural Networks*
- TensorFlow official tutorials on text generation and classification with RNNs
- PyTorch official tutorials: *NLP From Scratch* series

### 📊 Datasets to Practice On
- IMDB Reviews (sentiment analysis)
- AG News (text classification)
- Penn Treebank (language modeling)
- WMT (machine translation)
- CoNLL-2003 (named entity recognition)

> 🔎 Looking for a longer, categorized list (videos, extra courses, extra datasets)? See [`resources.md`](./resources.md).

---

## Summary

RNNs brought sequential memory to neural networks, enabling early breakthroughs in NLP such as machine translation and text generation. While Transformers have since become the dominant architecture for large-scale NLP, understanding RNNs — and their LSTM/GRU variants — remains essential for grasping the evolution of sequence modeling and for building lightweight, efficient models where full-scale Transformers are overkill.

---

## Repo Structure & Hands-on Notebooks

This repo also includes deep-dive chapters and runnable notebooks that expand on everything above.

### 📖 Chapters (in `docs/`)

1. [Introduction to NLP and Sequential Data](./docs/01-introduction.md)
2. [RNN Architecture](./docs/02-rnn-architecture.md)
3. [LSTM and GRU](./docs/03-lstm-gru.md)
4. [Sequence-to-Sequence Models & Attention](./docs/04-seq2seq-attention.md)
5. [RNNs vs Transformers](./docs/05-rnn-vs-transformers.md)

### 💻 Notebooks (in `notebooks/`)

| Notebook | What it covers |
|---|---|
| [`01_simple_rnn_pytorch.ipynb`](./notebooks/01_simple_rnn_pytorch.ipynb) | A vanilla RNN built from scratch in PyTorch, trained to generate text character-by-character |
| [`02_lstm_sentiment_classifier.ipynb`](./notebooks/02_lstm_sentiment_classifier.ipynb) | An LSTM-based sentiment classifier trained on the IMDB dataset (Keras/TensorFlow) |
| [`03_seq2seq_translation.ipynb`](./notebooks/03_seq2seq_translation.ipynb) | An encoder-decoder model with attention, trained on a toy sequence-reversal task (PyTorch) |

### 🚀 Getting Started

```bash
git clone https://github.com/HADIL19/Natural-Language-Processing-With-RNNs.git
cd Natural-Language-Processing-With-RNNs
pip install -r requirements.txt
jupyter notebook
```

### 🗂️ Full Folder Layout

```
├── README.md                              ← this file
├── LICENSE
├── requirements.txt
├── resources.md
├── docs/
│   ├── 01-introduction.md
│   ├── 02-rnn-architecture.md
│   ├── 03-lstm-gru.md
│   ├── 04-seq2seq-attention.md
│   └── 05-rnn-vs-transformers.md
└── notebooks/
    ├── 01_simple_rnn_pytorch.ipynb
    ├── 02_lstm_sentiment_classifier.ipynb
    └── 03_seq2seq_translation.ipynb
```

### 🤝 Contributing

This is an educational, evolving resource. Contributions are welcome — feel free to open an issue or pull request if you'd like to fix an error, clarify an explanation, add another notebook, or expand the resources list.

### 📄 License

This project is licensed under the [MIT License](./LICENSE).