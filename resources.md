# Resources

[← Back to README](./README.md)

A curated list of papers, courses, books, tools, and datasets for learning NLP with RNNs (and beyond).

---

## 📄 Foundational Papers

| Paper | Authors (Year) | Why it matters |
|---|---|---|
| *Finding Structure in Time* | Elman (1990) | Early formulation of recurrent networks for sequences |
| *Long Short-Term Memory* | Hochreiter & Schmidhuber (1997) | Introduces LSTM, solving vanishing gradients |
| *Learning Phrase Representations using RNN Encoder-Decoder* | Cho et al. (2014) | Introduces GRU and the encoder-decoder framework |
| *Sequence to Sequence Learning with Neural Networks* | Sutskever et al. (2014) | Popularized Seq2Seq for translation |
| *Neural Machine Translation by Jointly Learning to Align and Translate* | Bahdanau et al. (2015) | Introduces attention mechanism |
| *Attention Is All You Need* | Vaswani et al. (2017) | Introduces the Transformer — essential for context |
| *Effective Approaches to Attention-based NMT* | Luong et al. (2015) | Compares global vs local attention |

## 🎓 Courses

- **Stanford CS224n** — Natural Language Processing with Deep Learning (lecture videos + assignments freely available)
- **DeepLearning.AI Sequence Models** — Coursera, part of the Deep Learning Specialization (Andrew Ng)
- **Fast.ai — Practical Deep Learning for Coders** — NLP-focused lessons
- **MIT 6.S191** — Introduction to Deep Learning (has a dedicated RNN lecture)

## 📚 Books

- *Speech and Language Processing* — Jurafsky & Martin (free draft chapters available online)
- *Deep Learning* — Goodfellow, Bengio, Courville (Chapter 10 covers RNNs in depth)
- *Natural Language Processing with PyTorch* — Rao & McMahan
- *Neural Network Methods for Natural Language Processing* — Yoav Goldberg

## 🛠️ Libraries & Frameworks

| Library | Use |
|---|---|
| **PyTorch** | `torch.nn.RNN`, `LSTM`, `GRU` — flexible, widely used in research |
| **TensorFlow / Keras** | `tf.keras.layers.SimpleRNN`, `LSTM`, `GRU` — good for quick prototyping |
| **spaCy** | Industrial-strength NLP pipeline (tokenization, NER, POS tagging) |
| **NLTK** | Classic toolkit for preprocessing, tokenization, corpora |
| **Gensim** | Word embeddings — Word2Vec, FastText |
| **Hugging Face Transformers** | Compare RNN approaches against modern Transformer models |

## 🌐 Tutorials & Blog Posts

- Christopher Olah — *Understanding LSTM Networks* (colah.github.io) — the clearest visual explanation of LSTMs available
- Andrej Karpathy — *The Unreasonable Effectiveness of Recurrent Neural Networks*
- TensorFlow official tutorials — text generation and classification with RNNs
- PyTorch official tutorials — *NLP From Scratch* series

## 📊 Datasets for Practice

| Dataset | Task |
|---|---|
| IMDB Reviews | Sentiment analysis (binary classification) |
| AG News | Topic classification |
| Penn Treebank | Language modeling |
| WMT (various years) | Machine translation |
| CoNLL-2003 | Named entity recognition |
| SQuAD | Question answering (more advanced) |

## 🎥 Videos

- StatQuest — *RNNs, Clearly Explained* (great for building intuition)
- 3Blue1Brown-style visual explainers on recurrent networks and backpropagation

---

[← Back to README](./README.md)