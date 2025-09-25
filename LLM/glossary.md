

### 🔑 **LLM Glossary**

#### **1. Autoregressive Model**
A model that generates output one token at a time, predicting the next token based on previously generated tokens.  
**Example**: GPT models are autoregressive — they generate text by predicting one word at a time.

#### **2. Generative Model**
A model that can produce new data instances similar to the training data.  
**In NLP**: It generates coherent text, answers, summaries, etc.  
**Contrast**: Discriminative models classify or label data.

#### **3. Representative Model**
A model that learns to represent data in a compressed or abstract form, often used for tasks like clustering or dimensionality reduction.  
**Example**: BERT is more representative than generative — it learns contextual embeddings.

#### **4. Attention Mechanism**
A technique that allows models to focus on relevant parts of the input when making predictions.  
**Key Idea**: Not all words in a sentence are equally important for understanding meaning.  
**Example**: In “The cat sat on the mat,” attention helps the model focus on “cat” when predicting “sat.”

#### **5. Self-Attention**
A specific form of attention where a sequence attends to itself — each token looks at other tokens in the same sequence to gather context.  
**Used in**: Transformer models.

#### **6. Transformer**
A neural network architecture based entirely on attention mechanisms, replacing older RNNs and CNNs in NLP tasks.  
**Introduced in**: “Attention is All You Need” (2017).

#### **7. Token**
A unit of text used by LLMs — can be a word, subword, or character depending on the tokenizer.  
**Example**: “unbelievable” might be split into “un”, “believ”, “able”.

#### **8. Embedding**
A numerical representation of a token in a continuous vector space.  
**Purpose**: Captures semantic meaning — similar words have similar embeddings.

#### **9. Fine-Tuning**
Training a pre-trained model on a specific dataset to adapt it to a particular task or domain.  
**Example**: Fine-tuning GPT on legal documents for legal advice.

#### **10. Prompt Engineering**
Crafting input prompts to guide the model toward desired outputs.  
**Example**: “Translate this to French: Hello” vs. “What is the French word for ‘Hello’?”

#### **11. Context Window**
The maximum number of tokens a model can consider at once.  
**Example**: GPT-4 has a context window of up to 128k tokens in some versions.

#### **12. Positional Encoding**
Adds information about the position of tokens in a sequence, since transformers don’t inherently understand order.  
**Used in**: Transformer architecture.

#### **13. Pretraining**
Initial training of a model on a large corpus of general data to learn language patterns.  
**Followed by**: Fine-tuning for specific tasks.

#### **14. Zero-shot / Few-shot Learning**
- **Zero-shot**: Model performs a task without seeing examples.
- **Few-shot**: Model is given a few examples in the prompt to guide its response.

#### **15. Temperature (in Sampling)**
Controls randomness in generation:
- **Low temperature (e.g., 0.2)**: More deterministic.
- **High temperature (e.g., 0.9)**: More creative and diverse.

#### **16. Top-k / Top-p Sampling**
- **Top-k**: Chooses from the top *k* most likely tokens.
- **Top-p (nucleus sampling)**: Chooses from the smallest set of tokens whose cumulative probability exceeds *p*.

#### **17. Hallucination**
When a model generates plausible-sounding but incorrect or fabricated information.  
**Example**: Citing a non-existent research paper.

#### **18. Alignment**
Ensuring the model’s behavior aligns with human values, ethics, and intentions.  
**Includes**: Safety, fairness, and avoiding harmful outputs.
