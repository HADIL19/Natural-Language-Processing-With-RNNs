# Chapter 1: Introduction to NLP and Sequential Data

[← Back to README](../README.md) | [Next: RNN Architecture →](./02-rnn-architecture.md)

---

## What is Natural Language Processing?

**Natural Language Processing (NLP)** is the branch of artificial intelligence focused on enabling computers to read, understand, interpret, and generate human language. It sits at the intersection of linguistics, computer science, and machine learning.

Common NLP applications include:
- Search engines and question answering
- Machine translation
- Sentiment analysis
- Chatbots and virtual assistants
- Text summarization
- Speech recognition

## Why Is Language Hard for Computers?

Language is:
- **Ambiguous** — the same word can mean different things depending on context ("bank" = riverbank or financial institution).
- **Sequential** — meaning is built up over time, word by word.
- **Variable-length** — sentences and documents differ enormously in size.
- **Context-dependent** — earlier words affect the interpretation of later ones.

Traditional feedforward neural networks assume all inputs are fixed-size and independent of each other. That assumption breaks down for language, which is exactly the gap **Recurrent Neural Networks (RNNs)** were designed to fill.

## From Words to Numbers

Before any neural network can process text, we need to turn words into numbers. Common approaches:

| Method | Description |
|---|---|
| **One-hot encoding** | Each word is a sparse vector with a single 1; no notion of similarity |
| **Word2Vec / GloVe** | Dense vectors where similar words have similar vectors (pretrained) |
| **Learned embeddings** | An embedding layer trained jointly with the model |
| **Subword tokenization** (BPE, WordPiece) | Splits rare words into common subword units |

Once text is converted to a sequence of vectors, it can be fed into an RNN one timestep at a time.

## What You'll Learn in This Repo

1. How RNNs process sequences and maintain memory
2. Why vanilla RNNs struggle with long sequences (vanishing gradients)
3. How LSTM and GRU solve that problem with gating mechanisms
4. How encoder-decoder (Seq2Seq) models and attention work
5. How RNNs compare to modern Transformer-based models
6. Hands-on code you can run yourself in `notebooks/`

---

[Next: RNN Architecture →](./02-rnn-architecture.md)