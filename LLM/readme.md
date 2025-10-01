# LLM


Let’s unpack **LLMs (Large Language Models)** step by step, in simple terms and then a bit deeper.

---

## 🔹 1. What is an LLM?

* **Definition:**
  A **Large Language Model** is an artificial intelligence model trained on massive amounts of text data to **understand and generate human-like language**.

* **“Large”** = refers to the **huge number of parameters** (millions → billions → trillions). Parameters are the model’s "knobs" that get tuned during training.

* **“Language Model”** = predicts the **next word (token)** in a sequence, given the previous words.

---

## 🔹 2. How does it work?

Think of an LLM like an **autocorrect / autocomplete on steroids**:

1. You give it some input text (called a **prompt**).
2. It calculates probabilities of what word (or token) should come next.
3. It picks words step by step → producing coherent sentences, paragraphs, or even code.

Example:
Prompt: *“The capital of France is”*
LLM Prediction: *“Paris.”*

---

## 🔹 3. What makes them “smart”?

* They don’t just memorize text — they learn **patterns of language, facts, reasoning, and even problem-solving**.
* Trained on diverse sources (books, articles, code, conversations), they develop an internal representation of knowledge.

---

## 🔹 4. Key Features

* **Generative:** Can create new content (stories, answers, code).
* **Contextual:** Understand meaning from surrounding words.
* **Scalable:** Bigger models generally understand and generate better.
* **Versatile:** Can be used for chatbots, summarization, translation, coding, research, etc.

---

## 🔹 5. Example LLMs

* **GPT (OpenAI)** → ChatGPT, GPT-4
* **PaLM / Gemini (Google DeepMind)**
* **Claude (Anthropic)**
* **LLaMA (Meta)**
* **Cohere, Mistral, Falcon**

---

## 🔹 6. Analogy

Imagine:

* **Pretraining** = Teaching a student *all books in the library*.
* **LLM** = That student can now write essays, answer questions, or generate poems — because they’ve absorbed so much knowledge about language and the world.

---

## ✅ One-line definition

**An LLM is a very large neural network trained on massive text data to predict and generate human-like language.**
---
There are three main types of Large Language Model (LLM) architectures, based on how they process and generate text: **Encoder**, **Decoder**, and **Encoder-Decoder**. Each has a distinct role and is suitable for different types of language tasks. Here’s a clear explanation of each:

---

## 1. Encoder-Only Models

- **What they do:**  
  Focus on understanding and representing input text.
- **How they work:**  
  Take some text and convert it into a meaningful representation (vector/embedding) that captures its meaning and context.
- **Strengths:**  
  Great at comprehension tasks, such as:
  - Classification (e.g., spam detection)
  - Named Entity Recognition (NER)
  - Sentiment analysis
  - Text similarity
- **Limitation:**  
  Not typically used for generating long sequences of text.
- **Famous Example:**  
  - **BERT** (Bidirectional Encoder Representations from Transformers)
- **Analogy:**  
  Like a reader who understands and summarizes text, but doesn’t write new stories.
<img width="564" height="296" alt="image" src="https://github.com/user-attachments/assets/2fa2578d-505d-4a4e-9a2f-65460ab793f8" />

---

## 2. Decoder-Only Models

- **What they do:**  
  Generate new text, one token at a time, based on previous context.
- **How they work:**  
  Given a prompt, they predict the next word, then the next, and so on—producing coherent output.
- **Strengths:**  
  Excellent for generative tasks, such as:
  - Text completion
  - Story or code generation
  - Chatbots and dialogue systems
- **Limitation:**  
  Not bidirectional; can’t look at the whole input at once (only the left/context side).
- **Famous Examples:**  
  - **GPT** (Generative Pre-trained Transformer: GPT-2, GPT-3, GPT-4, etc.)
- **Analogy:**  
  Like an author who writes stories, predicting each word as they go.
<img width="932" height="232" alt="image" src="https://github.com/user-attachments/assets/ec277ad5-0d20-4aba-926d-48be3cdb595f" />

---

## 3. Encoder-Decoder Models (Seq2Seq)

- **What they do:**  
  Encode the input into a representation, then decode it to generate output—transforming one sequence into another.
- **How they work:**  
  The encoder reads the input and summarizes it; the decoder then takes this summary and generates the desired output.
- **Strengths:**  
  Perfect for sequence transformation tasks, such as:
  - Machine translation (e.g., English → French)
  - Summarization
  - Paraphrasing
  - Question answering
- **Famous Examples:**  
  - **T5** (Text-to-Text Transfer Transformer)
  - **BART** (Bidirectional and Auto-Regressive Transformer)
- **Analogy:**  
  Like a translator: listens to a sentence in one language, understands it, and then says it in another language.

<img width="759" height="316" alt="image" src="https://github.com/user-attachments/assets/e7510297-f867-48c9-b0bc-8f918eed7081" />
<img width="1160" height="470" alt="image" src="https://github.com/user-attachments/assets/346ffd3f-9ff8-420d-be32-7d4a59939d28" />

Text to Text Transformer from HuggingFace
---

### Summary Table

| Type              | Example(s)   | Typical Tasks                      | Directionality      |
|-------------------|--------------|------------------------------------|---------------------|
| Encoder           | BERT         | Understanding, classification      | Bidirectional       |
| Decoder           | GPT series   | Text generation, completion        | Left-to-right       |
| Encoder-Decoder   | T5, BART     | Translation, summarization, Q&A    | Both (seq2seq)      |

---

**In short:**  
- **Encoders** = Understand input.  
- **Decoders** = Generate output.  
- **Encoder-Decoders** = Transform input into output (translation, summarization, etc.).
---

Let’s dive into the **architecture of LLMs** in a way that’s beginner-friendly 🌟

---

# 🔹 The Secret Sauce: **Transformers**

Most modern LLMs (GPT, Claude, Gemini, LLaMA, etc.) are built on a neural network architecture called the **Transformer** (introduced in 2017 in the paper *“Attention is All You Need”*).

---

## 1. Core Idea: **Tokens and Sequences**

* Text is broken into **tokens** (pieces of words, e.g., *“playing” → “play” + “ing”*).
* The model processes these tokens in **sequences**.
* Goal = predict the next token given the previous ones.

---

## 2. The Transformer Components

### 🔸 a) Embeddings

* Words/tokens are turned into **vectors** (numbers in high-dimensional space).
* Example: *“cat”* → `[0.12, -0.87, 0.45, …]`

### 🔸 b) Self-Attention (the magic ✨)

* Each token looks at **all the other tokens** in the sentence to figure out which ones matter most.
* Example:

  * Sentence: *“The cat sat on the mat because it was soft.”*
  * The word *“it”* should connect to *“mat”*, not *“cat”*.
  * Self-attention helps the model figure that out.

⚡ This is why it’s called *“Attention is All You Need”*.

### 🔸 c) Transformer Layers

* The model has many stacked **layers of attention + feedforward networks**.
* Each layer refines the representation of tokens.
* More layers = deeper understanding.

### 🔸 d) Output Layer

* After processing, the model outputs a probability distribution for the next token.
* Example:

  * “The capital of France is …”
  * Probabilities: `[Paris: 0.95, London: 0.02, Rome: 0.01, …]`
  * Picks *Paris*.

---

## 3. Scaling Up = LLM

* Small transformer → can translate short phrases.
* Gigantic transformer (billions/trillions of parameters) → can write essays, code, reason, and answer complex questions.

---

## 4. Visual Analogy 🖼️

Think of it like a **classroom**:

* **Embeddings** = turning words into student ID cards.
* **Self-attention** = students looking around to see which classmates’ answers are relevant before answering.
* **Layers** = multiple rounds of classroom discussions, refining understanding.
* **Output** = the final agreed-upon answer.

---

## ✅ Summary

* LLMs use the **Transformer architecture**.
* Key innovation = **self-attention** (lets the model capture context and meaning across long text).
* With enough layers + data + parameters, Transformers become **Large Language Models** capable of reasoning, coding, and conversation.

---


<img width="890" height="967" alt="image" src="https://github.com/user-attachments/assets/97009cb8-646b-421c-95c9-d300152259cb" />



| Stage                  | Purpose                                 | Data Type                        | Analogy                             |
| ---------------------- | --------------------------------------- | -------------------------------- | ----------------------------------- |
| **Pretraining**        | General language & knowledge            | Internet-scale text              | Learning the whole language         |
| **Instruction Tuning** | Follow natural language instructions    | Instruction–response pairs       | Learning how to follow directions   |
| **RLHF**               | Align with human values (helpful, safe) | Human feedback & preference data | Getting coached by a teacher        |
| **Task Fine-tuning**   | Specialize for domain/task              | Domain-specific data             | Becoming a doctor, lawyer, or coder |

---

👉 So:

* **Pretraining** = foundation (language + facts).
* **Instruction Tuning** = makes it *usable*.
* **RLHF** = makes it *safe & aligned*.
* **Fine-tuning** = makes it *domain-specialized*.


<img width="1345" height="556" alt="image" src="https://github.com/user-attachments/assets/dd43ab93-6329-4358-8d7d-0a754cf82443" />


<img width="1490" height="980" alt="image" src="https://github.com/user-attachments/assets/7ac11b0c-2cc1-4e22-bcbd-f4efa7447761" />

---


## 🔹 1. **Pretraining**

* **What happens?**

  * All parameters (billions/trillions of weights) are trained **from scratch** (starting from random initialization or sometimes from smaller pretrained checkpoints).
* **Effect:**

  * The model learns general language structure, grammar, facts, reasoning.
* ✅ **Parameters updated:** Yes, **all** of them.

---

## 🔹 2. **Instruction Tuning**

* **What happens?**

  * The pretrained model is fine-tuned on curated instruction–response datasets.
  * Usually updates **all parameters** (full fine-tuning) or sometimes uses **parameter-efficient fine-tuning (PEFT)** like LoRA (only a small subset of parameters or added adapter layers).
* **Effect:**

  * Model learns to follow instructions better, instead of just predicting next words.
* ✅ **Parameters updated:** Yes — all or subset, depending on method.

---

## 🔹 3. **RLHF**

* **What happens?**

  * Phase 1: A **reward model** (separate smaller model) is trained on human preferences.
  * Phase 2: The main LLM is further trained using reinforcement learning (PPO or similar), adjusting its parameters to maximize “human-preferred” outputs.
* **Effect:**

  * Model becomes more aligned, polite, safe, and useful.
* ✅ **Parameters updated:** Yes — the LLM’s parameters are adjusted again during policy optimization.

---

## 🔹 4. **Task Fine-tuning**

* **What happens?**

  * The model is fine-tuned on **domain-specific data** (e.g., legal, medical, finance).
  * Can be **full fine-tuning** or **PEFT** (LoRA, adapters, prefix tuning).
* **Effect:**

  * Model specializes in domain knowledge.
* ✅ **Parameters updated:** Yes — but often only a subset (to save cost


# MUG UP QUESTIONS
The correct answer is:  
**C. 100 MB**

---

### Explanation:

According to Oracle Cloud Infrastructure (OCI) documentation, the **maximum file size** for ingestion into an Object Storage bucket as a **Generative AI knowledge base** is:

- **100 MB per file**  
- Supported file types include **PDF, TXT, JSON, HTML, and Markdown (MD)**  
- Files exceeding this limit are **ignored during ingestion**[1](https://docs.oracle.com/en-us/iaas/Content/generative-ai-agents/data-requirements.htm).

---

### Additional Notes:
- You can ingest **up to 10,000 files** per data source.
- Only **one Object Storage bucket** is allowed per data source.
- If needed, you can **request a limit increase** through OCI support[2](https://docs.oracle.com/en-us/iaas/Content/generative-ai-agents/limits.htm).



# Oracle AI Professional Certification

https://www.linkedin.com/pulse/cheersheet-1z0-1127-25-oracle-cloud-infrastructure-2025-david-edwards-xdqde


Fundamentals
1. What are Large Language Models (LLMs)?


Definition: Probabilistic models of text that compute distributions over vocabulary.
Core Function: Given input text, predict the most likely next word(s).
“Large”: Refers to the number of trainable parameters. No fixed threshold, but typically billions of parameters.
2. Architectures of LLMs


Encoders: Convert text into embeddings (numeric representations of meaning).
Decoders: Generate text by predicting the next token.
Encoder-Decoder Models: Combine both, mainly used for translation and sequence-to-sequence tasks.
3. Prompting and Prompt Engineering


Prompting: Changing model inputs to influence output.
Prompt Engineering: Refining inputs to get desired results.
4. Risks of Prompting


Prompt Injection: Malicious manipulation of inputs to force unintended outputs (similar to SQL injection).
Leakage: Attackers can trick LLMs into revealing developer prompts.
Data Risks: Potential exposure of private or sensitive data from training.
5. Training Methods


Fine-Tuning: Retraining all parameters for a specific task. Expensive but effective.
Parameter-Efficient Fine-Tuning (PEFT): Train only a small subset of parameters (e.g., LoRA).
Soft Prompting: Train special tokens (prompt embeddings) without changing core parameters.
Continual Pretraining: Adapt the model to a new domain with large amounts of new text.
6. Decoding (Generating Text)


Greedy Decoding: Always pick the most probable token.
Random Sampling: Pick words based on a probability distribution.
Temperature: Controls randomness; lower = deterministic, higher = more creative.
Nucleus Sampling (Top-p): Restrict sampling to the most probable portion of the distribution.
Beam Search: Generate multiple sequences simultaneously, keep the best ones.
7. Key Challenges


Hallucinations: Model generates fluent but incorrect information.
Bias and Safety: Outputs may reflect harmful or biased data from training.
Cost: Training very large models requires thousands of GPUs and significant resources.
Practice Questions
Q1. What does “large” in LLM primarily refer to? a) Vocabulary size b) Training dataset size c) Number of parameters d) Number of GPUs used

Q2. Which architecture is best suited for semantic search? a) Decoder b) Encoder c) Encoder-Decoder d) Random Forest

Q3. Which of the following is an example of a decoder model? a) BERT b) GPT-4 c) ResNet d) Word2Vec

Q4. What is the main purpose of prompting? a) Changing model weights b) Altering model input to influence output c) Reducing model size d) Speeding up training

Q5. Zero-shot prompting involves: a) Providing no task description, b) Providing examples of the task, c) Only providing the task description with no examples, d) Feeding only numbers

Q6. Which prompting technique encourages step-by-step reasoning? a) Chain-of-Thought prompting b) Zero-shot prompting c) Greedy prompting d) LoRA prompting

Q7. Which training method adds new parameters without changing original ones? a) Fine-tuning b) LoRA c) Soft Prompting d) Continual Pretraining

Q8. What does temperature control in decoding? a) The speed of model training b) The size of the vocabulary c) The randomness of token selection d) The number of parameters updated

Q9. Which decoding method always picks the most likely token? a) Random Sampling b) Beam Search c) Greedy Decoding d) Nucleus Sampling

Q10. Which of the following is a risk of deploying LLMs? a) Faster processing b) Prompt Injection c) Higher accuracy d) Lower memory usage

Answer Key

c) Number of parameters
b) Encoder
b) GPT-4
b) Altering model input to influence output
c) Only providing a task description with no examples
a) Chain-of-Thought prompting
c) Soft Prompting
c) The randomness of token selection
c) Greedy Decoding
b) Prompt Injection
OCI Generative AI Service
1. Introduction to OCI Generative AI


OCI Generative AI Service: Fully managed, serverless platform for building generative AI apps.
Access: Single API for multiple foundational models (Cohere & Meta).
2. Pre-trained Models

Chat Models


Command-R-Plus:
Command-R (16k):
Meta Llama 3.1/3.2/3.3:
Embedding Models


Convert text into vector representations.
Use cases: semantic search, clustering, classification, RAG.
Cohere embed-english & embed-multilingual (100+ languages).
V3 embed models: Higher retrieval quality for noisy datasets.
Dimensions: 1024-d vectors (standard), 384-d (lite).
Limit: 512 tokens/input, max 96 inputs/run.
3. Prompting & Prompt Engineering


Prompt = input text provided to LLM.
Prompt Engineering = refining inputs for desired outputs.
Types of Prompting


Zero-shot: Only task description.
Few-shot (k-shot): Include k examples.
Chain-of-Thought (CoT): Step-by-step reasoning.
Zero-shot CoT: Add “Let’s think step by step.”
In-context learning: Supply demonstrations in a prompt.
Prompt Parameters


Preamble Override: Change tone/style (e.g., pirate tone).
Temperature: Controls randomness (0 = deterministic, 1 = diverse).
Top-k: Choose from the top-k highest probability tokens.
Top-p (nucleus sampling): Choose from the smallest probability set that adds up to p.
Frequency Penalty: Reduces repetition based on frequency.
Presence Penalty: Penalizes repeated tokens regardless of frequency after the first occurrence.
4. Customization Options


Training from scratch is not recommended (expensive, data-hungry).
Comparison

Article content
5. Fine-Tuning & Inference

Fine-tuning - Process of retraining an LLM (fully or partially) on domain-specific data to adapt it to a particular task or style.

Inference - The serving phase, when a fine-tuned or base model is deployed and generates predictions/answers in response to prompts.

Fine-Tuning Workflow


Collect training data (task/domain-specific).
Upload to Object Storage.
Launch fine-tuning job (choose base model + tuning method).
Training runs on a GPU cluster.
Store fine-tuned weights (encrypted in Object Storage).
Register the model in OCI Generative AI.
Inference Workflow:
Evaluation Metrics:
Inference Workflow


User sends a prompt/query (via app → API endpoint).
Model receives prompt (fine-tuned or base model).
Model generates output tokens based on probability distribution (with decoding strategies: greedy, top-k, top-p, temperature).
Output returned → response to end-user.
Optional: Log, trace, moderation, citations (if RAG).
Used to measure model quality during or after fine-tuning:


Perplexity: How well the model predicts tokens (lower = better).
BLEU (Bilingual Evaluation Understudy): Measures translation/text similarity by overlapping n-grams.
ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures recall/overlap between generated and reference text (common in summarization).
Accuracy / F1-score: Used for classification tasks.
Loss function: Tracks how well the model is minimizing prediction error during training.
6. Dedicated AI Clusters

Fine-Tuning Clusters


Purpose: Used to train/adapt models with custom data.
Billing: Pay only for the duration of fine-tuning.
Example: Fine-tuning a Cohere Command-R model with 8 units over 12 hours.

Cost = 8 units × hourly rate × 12 hrs.

Hosting (Inference) Clusters


Purpose: Used to serve models in production (answer user prompts).
Billing: Minimum 744 hours/month (24×7).
Hosting a Llama 13B model with 2 units for customer support chatbot.

Article content
Sample Exam Questions

Q1. Which OCI chat model allows up to 128k tokens per input? a) Command-R (16k) b) Command-R-Plus c) Llama 3.1 (70B) d) Cohere embed-english

Q2. What is the main use case for embedding models? a) Conversational chat b) Text-to-image c) Semantic search d) Translation only

Q3. Which fine-tuning method updates only ~0.01% of weights? a) Vanilla fine-tuning b) T-Few c) LoRA d) Zero-shot prompting

Q4. Which parameter controls the randomness of the output? a) Frequency penalty b) Temperature c) Top-k d) Context length

Q5. What is a major advantage of RAG? a) Eliminates the cost of GPUs b) Provides grounded answers with enterprise data c) Removes the need for tokenization d) Always reduces latency

Q6. Which evaluation metric is better for generative AI? a) Accuracy b) Loss c) BLEU d) ROUGE

Q7. Which cluster type is used for Llama models? a) Large Cohere Dedicated b) Small Cohere Dedicated c) Large Meta Dedicated d) Embedding Cohere Dedicated

Q8. Which parameter reduces repeated phrases based on count? a) Temperature b) Top-p c) Frequency penalty d) Presence penalty

Q9. What is the minimum hosting commitment for OCI clusters? a) 1 hour b) 10 hours c) 744 hours d) 96 hours

Q10. You are deploying a fine-tuned Llama model on OCI and need to ensure that your model weights are encrypted and only accessible within your tenancy. Which service should you configure to manage and control the encryption keys?

a) OCI IAM b) OCI Object Storage c) OCI Key Management d) OCI Data Guard

Answer Key

b) Command-R-Plus
c) Semantic search
b) T-Few
b) Temperature
b) Provides grounded answers with enterprise data
b) Loss
c) Large Meta Dedicated
c) Frequency penalty
c) 744 hours
c) OCI Key Management
LangChain, RAG, and Oracle 23ai Integration
1. LangChain Overview


LangChain: A framework for building LLM-powered applications that are context-aware.
PromptTemplate → for single-string inputs (good for simple LLMs).
ChatPromptTemplate → for multi-turn conversational formats, lets you structure messages with roles like system, human, AI.
What is Memory in LangChain?


By default, LLMs are stateless → each query is processed independently.
Memory lets the application remember previous interactions so the model can generate context-aware answers.
It does not change the LLM itself; instead, LangChain manages how past inputs/outputs are stored and passed back into the model.
How does Memory Work with LangChain?


User asks a question → LangChain records it.
LLM generates a response → also recorded.
On the next interaction, LangChain retrieves the conversation history (or a summary of it).
This history is appended to the new prompt so the LLM sees context.
2. OCI Generative AI + Oracle 23ai Integration


Embeddings: Generated via OCI Generative AI service.
Oracle 23ai SELECT AI: Converts natural language → SQL queries using OCI Generative AI.
Oracle 23ai Vector Store: Stores embeddings (VECTOR datatype). Supports similarity search.
3. Retrieval Augmented Generation (RAG)

Why RAG?


Traditional LLMs: Limited by training data (may be outdated, biased).
RAG: Retrieves relevant, up-to-date info → feeds it as context to LLM.
4. RAG Pipeline Phases


Ingestion
Embedding & Storage
Retrieval
Generation
Summary


Ingestion - Load + Split documents → PDFReader, TextSplitter.
Embed + Store → OCIEmbeddingModel, OracleVS.
Retrieve → Similarity search (k=3, cosine or dot product).
Chain → RetrievalQA + ChatOCIGenAI.
Execute → Query → Retrieve → Generate.
5. Conversational Memory in RAG


Chat = Q&A sequence.
Memory: Maintains conversation history as context.
LangChain provides memory classes to persist summaries, entities, or raw text.
Example:
6. Vector Indexes for Similarity Search

Why indexes?

As your chunk store grows, brute-force similarity over all embeddings becomes slow. Vector indexes narrow the search space while preserving relevant neighbors for fast top-K retrieval.

Two common index families (Oracle 23ai)

1) HNSW — Graph-based neighbor index


What it is: A Hierarchical Navigable Small World graph where each vector links to nearby vectors across multiple layers.
Why use it: Very fast approximate nearest-neighbor (ANN) search with strong recall at low latency, well-suited for interactive RAG/chat.
Trade-offs: Requires memory to store the graph; build time and memory rise with corpus size.
Best for: Low-latency Q&A, semantic search where the user waits for responses.
2) IVF — Inverted File (partition) index


What it is: A partition/cluster-based index (Inverted File Flat) that routes a query to the most relevant partitions, then searches inside them.
Why use it: Efficient at scale by avoiding a full scan; good throughput when your corpus is large.
Trade-offs: Recall depends on how many partitions you probe; slightly higher tuning needs to balance speed vs. accuracy.
Best for: Very large collections where you need predictable query times.
Similarity metrics


Dot product: Considers magnitude + angle between vectors.
Cosine similarity: Considers angle only (magnitude-invariant). Choose based on how your embeddings were trained and how you want “closeness” measured.
How it fits the RAG flow


Embed chunks → store vectors in Oracle 23ai’s VECTOR column.
Index with HNSW (graph) or IVF (partition) to accelerate top-K search.
Retrieve top matches using dot or cosine similarity.
Generate an answer with LLM using the retrieved context
Sample Exam Questions
Q1. Which LangChain component is used to preserve past conversation for context? a) Prompt Template b) Memory c) Chain d) Vector Store

Q2. In LangChain, which template is designed for conversational inputs? a) Prompt Template b) ChainTemplate c) ChatPromptTemplate d) DialogueTemplate

Q3. Which Oracle feature allows you to query databases using natural language? a) Oracle Cloud Guard b) Oracle Data Guard c) Oracle SELECT AI d) Oracle AutoML

Q4. Which similarity metric considers both magnitude and angle between embeddings? a) Cosine Similarity b) Dot Product c) Euclidean Distance d) Jaccard Index

Q5. Why is chunk overlap used in text splitting? a) To reduce cost b) To maintain semantic continuity c) To increase retrieval speed d) To avoid indexing

Q6. What type of index is HNSW? a) Partition-based index b) Graph-based neighbor index c) Flat index d) Semantic index

Q7. In RAG, what is the role of embeddings? a) Tokenize text into smaller parts b) Convert text into numerical vectors for semantic similarity c) Generate SQL queries d) Reduce context window length

Q8. Which LangChain class is used for chaining retrieval + LLM together? a) ChatPromptTemplate b) RetrievalQA c) MemoryChain d) OracleVS

Q9. What is a key benefit of RAG compared to training larger LLMs? a) Reduces context window size b) Eliminates need for embeddings c) Provides up-to-date, domain-specific answers without retraining d) Increases training dataset size

Q10. Which Oracle 23ai datatype is used for storing embeddings? a) BLOB b) JSON c) VECTOR d) VARCHAR

Answer Key

b) Memory
c) ChatPromptTemplate
c) Oracle SELECT AI
b) Dot Product
b) To maintain semantic continuity
b) Graph-based neighbor index
b) Convert text into numerical vectors for semantic similarity
b) RetrievalQA
c) Provides up-to-date, domain-specific answers without retraining
c) VECTOR
OCI Generative AI Agents
1. Overview


OCI Generative AI Agents: Fully managed service combining LLMs with intelligent retrieval for contextual, actionable responses.
Purpose: Automates tasks like booking, querying, and summarization by combining reasoning, acting, planning, and persona.
2. Core Architecture


Interface: Chatbot, web app, voice, or API.
External Knowledge: RAG (Retrieval-Augmented Generation) ensures answers are grounded in external data.
Feedback Loop: Responses feed into memory for better context.
3. Key Concepts


Generative AI Model: LLM trained on large data for NLU + NLG.
Agent: Autonomous system built on LLM + RAG.
Answerability: Model responds relevantly to queries.
Groundedness: Responses traceable to data sources (citations).
4. Data Access Hierarchy


Data Store: Where data resides (Object Storage, DB).
Data Source: Connection details for accessing the store.
Knowledge Base: Vector storage system organizing ingested data for retrieval.
5. Supported Data Options


OCI Object Storage → Managed ingestion, supports PDF & TXT (≤100MB).
OCI OpenSearch → Bring-your-own indexed data.
Oracle DB 23ai Vector Store → Custom embeddings, SQL-based retrieval functions.
6. Data Ingestion


Extract, transform, and store data into the knowledge base.
Hybrid Search = Lexical + Semantic search for higher accuracy.
Ingestion Jobs: Add, retry, update, or cancel ingestion processes.
7. Additional Features


Session: Maintains context across exchanges (timeout 1hr–7d).
Endpoint: Access point for agents to connect externally.
Trace: Logs full conversation history.
Citation: Sources of agent responses for transparency.
Content Moderation: Filters harmful content (input, output, or both).
8. Database Guidelines


Create vector tables (DOCID, body, vector).
Ensure the embedding model used for query = the model used for data.
Define retrieval functions returning DOCID, body, and score.
Supports cosine similarity or Euclidean distance.
9. Agent Workflow


Create Knowledge Base → Choose data store (Object Storage or Oracle DB).
Ingest Data → PDFs, TXTs, or vectorized data.
Create Agent → Define welcome message, select knowledge base.
Create Endpoint → Connect to apps; configure session, moderation, trace, citation.
Chat & Test → Query agent, view responses, citations, trace.
10. Default Limits (Highlights)


1 KB → 1 data source.
Files: Up to 1,000 per source (100MB each).
Sessions: Timeout 3,600s default (extendable).
Practice Questions (MCQs)
Q1. What is the role of the Knowledge Base in OCI Generative AI Agents? a) Provides APIs for LLM b) Organizes ingested data for retrieval c) Stores only prompts d) Manages system logs

Q2. Which feature ensures model responses are traceable to original data sources? a) Trace b) Persona c) Groundedness d) Content Moderation

Q3. Which input helps maintain continuity in conversations? a) Prompt b) Tools c) Memory d) Citation

Q4. Which OCI service allows uploading PDF/TXT files for ingestion by Generative AI Agents? a) OCI IAM b) OCI Object Storage c) OCI Vault d) OCI Logging

Q5. Hybrid search combines: a) Semantic + Exact match search b) Vector + Image search c) RAG + Prompt engineering d) SQL + Graph search

Q6. Which component defines the connection details for data retrieval? a) Knowledge Base b) Data Store c) Data Source d) Embedding Model

Q7. In Oracle DB guidelines, what must align between queries and stored vectors? a) File formats b) Table schemas c) Embedding models d) Indexing methods

Q8. What is the default session timeout for OCI Agents? a) 300 seconds b) 1 hour c) 24 hours d) 7 days

Q9. Which feature tracks the full conversation history for monitoring? a) Trace b) Session c) Content Moderation d) Endpoint

Q10. Which search method is used for meaning-based retrieval? a) Lexical b) Semantic c) Hybrid d) Full-text

Answer Key: Q1-b, Q2-c, Q3-c, Q4-b, Q5-a, Q6-c, Q7-c, Q8-b, Q9-a, Q10-b

Fine-Tuning GPU Sizes & Calculations (OCI Generative AI)
1. Why GPU Sizing Matters


Fine-tuning LLMs requires specialized GPU clusters.
Choosing the wrong size = wasted cost or poor performance.
OCI provides Dedicated AI Clusters.
2. GPU Cluster Options

OCI clusters are provisioned in units. Each unit represents a set of GPUs + networking capacity.


Fine-tuning: Needs multiple units (to process large parameter updates).
Inference/Hosting: Can use fewer units, but must commit to 744 hours/month (24×7).
Typical Configurations

Article content
3. Cost & Time Estimation

When planning fine-tuning, calculate based on:


Model size (parameters) → larger = more GPU memory needed.
Dataset size (tokens) → more tokens = longer training.
Cluster units → parallelism increases speed, but cost scales linearly.
Formula for Fine-tuning Cost

Cost = (Number of Units × Hourly Rate per Unit × Hours Run)

Example:


Fine-tune on 8 Cohere units.
Training job runs for 12 hours.
Rate = $1 per hour per unit (OCI pricing table).
👉 Total = 8 × 1 × 12 = 96 dollars.

4. Hosting (Inference) Calculations

Hosting requires full-time availability.

Formula:

Cost = (Number of Units × Hourly Rate per Unit × 744 hours/month)

Example:


Hosting with 2 units.
Hourly = $Y/unit.
Monthly Cost = 2 × Y × 744 = 1488 * Y dollars/month.

Fine-tuning Methods Explained
Oracle’s Generative AI supports several methods depending on the base model:


Vanilla fine-tuning
LoRA (Low-Rank Adaptation)
T-Few (Task-specific Few-shot fine-tuning)
Article content
5. Performance Tuning Guidelines


Small models (≤13B params) → fine-tune with 4–8 units.
Medium models (30–70B params) → 8–16+ units.
Large models (100B+ params) → distributed fine-tuning across 32+ units.
Tips:


Always run pilot fine-tuning jobs with a sample dataset → estimate time & cost.
Use LoRA or T-Few parameter-efficient tuning to cut GPU needs.
For global availability → deploy smaller hosting clusters per region (US, EU, APAC).
6. Exam Pointers


Fine-tuning = high burst GPU needs, short time window.
Hosting = fewer GPUs, always-on, billed monthly (744 hrs).
LoRA/T-Few = less GPU, cheaper training.
Formula knowledge (Units × Rate × Hours) may appear in scenario-based questions.
Moke Exam:

50-Question Mock Exam (Multiple Choice)
Below is a 50-question practice exam that reflects the style and coverage of the Oracle OCI Generative AI Professional certification. Each question is multiple-choice with one correct answer (except where noted). Answers are listed separately at the end.

Domain 1: LLM Fundamentals (Large Language Models)

What best describes a Large Language Model (LLM)? A. A neural network trained on massive amounts of text data to predict and generate language. B. A rule-based program for understanding grammar and syntax. C. A database that stores and retrieves large text documents. D. A cloud service for translating languages in real-time.
Which statement correctly contrasts encoder vs. decoder model architectures? A. Encoders generate long passages of text, whereas decoders produce only embeddings. B. Encoders excel at transforming text into vector representations, while decoders excel at generating text from prompts. C. Decoder models are used only for image data, encoders only for text data. D. Decoders require labeled data to operate; encoders do not.
You want an LLM to translate a sentence from English to French. Which prompt engineering approach provides the model the best guidance? A. Zero-shot prompting – simply ask for the translation with no examples. B. One-shot prompting – provide the English sentence once and ask for French. C. Few-shot prompting – include a couple of example English–French translations in the prompt before asking for the new translation. D. Chain-of-thought prompting – instruct the model to explain its reasoning step by step in French.
What is a known limitation of large language models that RAG (Retrieval-Augmented Generation) aims to address? A. LLMs often refuse to answer questions due to a lack of training data. B. LLMs may produce outdated or incorrect facts (hallucinations) because their knowledge is limited to training data. C. LLMs cannot handle more than a single question at a time. D. LLMs are unable to generate any content without an external knowledge base.
During deployment, how can an LLM be vulnerable to prompt injection? A. Users can craft inputs that trick the model into ignoring system instructions and producing disallowed output. B. The model’s weights can be directly altered via certain prompts. C. Prompt injection refers to overloading the model with too long a prompt. D. It’s a method of improving model accuracy by injecting clarifying prompts.
Which of the following is an example of a code model (LLM specialized for code)? A. A transformer model trained on GitHub code that can complete and generate functions in Python or Java. B. An LLM fine-tuned on medical research papers for clinical Q&A. A generative adversarial network (GAN) that produces source code. D. A compiler that translates high-level code to machine code using AI.
What does a multi-modal LLM refer to? A. An AI model that can handle more than one human language (multilingual model). B. A model that processes and/or generates multiple types of data (e.g., text, images, audio). C. Any LLM with over 100 billion parameters. D. A model using multiple neural network architectures in parallel.
Which hyperparameter would you adjust to make an LLM’s output less random and more deterministic? A. Increase the temperature. B. Decrease the temperature. C. Increase the max token limit. D. Use a larger context window.
Your LLM outputs are good, but sometimes too brief. Which parameter or method helps in getting longer, more detailed responses? A. Increase the max_tokens (generation length limit) for the model’s output. B. Lower the temperature to 0.0. C. Use zero-shot prompting instead of few-shot. D. Enable beam search with a beam size of 1.
Which training approach allows adapting a large pre-trained LLM to a new task by updating only a small number of additional weights? A. Full fine-tuning – update all model parameters on the new task data. B. Prompt engineering – no model parameters are updated at all. C. Parameter-efficient tuning (e.g, LoRA) – introduce small trainable weight matrices or “adapters” instead of retraining the whole model. D. Continual pretraining – train on large unlabeled data from scratch.
Domain 2: OCI Generative AI Service

Which is true about the OCI Generative AI service? A. It’s a fully managed, serverless service that provides access to large language models via a unified API. B. It requires customers to bring their own GPU hardware. C. It is only for computer vision tasks, not text generation. D. It cannot be accessed through the OCI Console, only via API.
What does it mean that OCI GenAI provides “single API” access to multiple models? A. You must integrate separate APIs for each foundation model vendor. B. The same unified endpoint and API format lets you switch between different underlying models with minimal code changes. C. All users share one global API key for the service. D. The API only supports one model at a time.
Which foundation models are available out of the box in OCI’s Generative AI Service for text tasks? A. Models from Meta (Llama 2) and Cohere (Command family for chat; Embed for embeddings). B. Only Oracle’s proprietary LLMs are trained in-house. C. OpenAI’s GPT-4 and GPT-3 models. D. Google’s PaLM model family.
What is the primary intended use of the embedding models in OCI GenAI? A. To generate conversational dialogue responses. B. To convert text into vector representations for semantic search and similarity tasks. C. To fine-tune on code datasets. D. To translate documents between languages.
You need to analyze a large PDF (300 pages) for semantic search. Which model and approach should you use on OCI GenAI? A. Use a chat completion model to directly input all 300 pages as a prompt. B. Use the embedding model to convert chunks of the document into vectors, then use similarity search for relevant content. C. Fine-tune the chat model on the PDF content first. D. This is not possible with the OCI GenAI service.
The Cohere Command-R vs. Command-R-Plus models in OCI GenAI differ primarily in what way? A. The R-Plus model supports a far larger context window (prompt size up to 128k tokens) and higher performance, whereas Command-R is limited to 16k context. B. Command-R-Plus handles only code, Command-R handles only text. C. Command-R is for English, R-Plus is for multilingual tasks. D. R-Plus is cheaper to run but less capable than R.
Which is a supported use case of OCI’s pre-trained foundation models? A. Generating images from text descriptions. B. Summarizing a document or answering questions in a chat format. C. Training a new model from scratch using a custom architecture. D. Real-time video translation.
Why would you choose to fine-tune a foundation LLM via OCI GenAI? A. To adjust the model’s weights so it performs better on domain-specific tasks or data (e.g., your industry’s terminology). B. To significantly reduce the model’s size by pruning parameters. C. To increase the context window of the model, D. Fine-tuning is not possible in OCI GenAI; only prompting is supported.
Oracle’s GenAI service implements an efficient fine-tuning method called T-Few. What is a key characteristic of T-Few fine-tuning? A. It fully trains the entire model on your data. B. It inserts new adapter layers and updates only a small fraction of the model’s weights, reducing training time and cost. C. It uses reinforcement learning from human feedback instead of gradient descent. D. It requires at least 1 million training examples to be effective.
Before fine-tuning a model on OCI GenAI, what resource must you have or create? A. A Kubernetes cluster in OCI for the training job. B. A dedicated AI cluster (GPU cluster) is allocated to run the fine-tuning job. C. A Docker container with the model weights. D. An Object Storage bucket to manually upload the model.
In OCI GenAI, what is a model endpoint? A. A saved checkpoint of a training run. B. A network endpoint URL where a specific model (base or fine-tuned) is deployed for inference requests. C. The internal API the service uses to call the foundation model. D. The logging interface for model outputs.
After you fine-tune a foundation model on OCI GenAI, where are the resulting custom model weights stored? A. In Oracle’s central model repository (shared across tenants). B. In your OCI Object Storage, within your tenancy. C. They are merged back into the base model and are not accessible separately. D. On the GPU cluster indefinitely.
How does OCI GenAI ensure that one customer’s fine-tuning activities don’t interfere with another’s? A. By assigning dedicated GPU clusters and isolated networking (RDMA) for each customer’s workload. B. By running jobs sequentially for each region. C. Through virtualization of GPUs with hypervisors. D. By only allowing one user of the service at a time.
Which OCI service is used to control access to the Generative AI Service APIs and endpoints? A. Oracle Cloud Guard. B. OCI Identity and Access Management (IAM). C. Oracle Data Safe. D. OCI Key Management.
What role does OCI Key Management (Vault) play in the GenAI service? A. It stores API keys for accessing GenAI. B. It securely manages the encryption keys for the hosted foundation models and fine-tuning artifacts. C. It monitors the service for security threats. D. It manages SSH keys for GPU cluster access.
Your company’s compliance policy says “no customer data used for AI should leave the tenant’s boundary.” How does OCI GenAI support this? A. All data sent to the GenAI models is first anonymized by Oracle. B. Fine-tuned models and any data embeddings are stored within your own tenancy’s infrastructure (e.g., your Object Storage and database). C. Oracle shares the fine-tuned model with others but not the raw data. D. The service cannot guarantee this – data always leaves the tenant boundary.
Which of the following is NOT a feature or characteristic of OCI’s Generative AI Service? A. Choice of multiple pre-trained LLMs (Cohere, Llama 2) for generation and embedding tasks. B. Automatic scaling and serverless usage – you do not manage compute instances. C. Built-in support for deploying the models on-premises. D. The ability to fine-tune foundation models with your dataset.
What is the purpose of the GenAI Playground in the OCI Console? A. It is a visual interface for testing prompts and models interactively, without writing code. B. It’s a training environment for model fine-tuning. C. It’s a monitoring dashboard for model endpoints. D. It is a game that teaches you how to use AI.
Which use case would embedding models + semantic search be better for than a standard keyword search? A. Finding documents that are relevant in meaning to “investment banking trends,” even if they don’t contain those exact words. B. Finding documents that contain the exact phrase “investment banking trends.” C. Searching by document title only. D. Ensuring results are ordered by date.
After deploying a custom model to an endpoint, how do you integrate it into an application? A. By calling the endpoint’s REST API with appropriate auth, passing prompts, and getting model inferences in response. B. By connecting the OCI Streaming service to the endpoint. C. By using an SDK only, the model cannot be called via REST. D. By importing the model file into your application manually.
Domain 3: Implementing RAG (Retrieval-Augmented Generation) with OCI GenAI and LangChain

What is Retrieval-Augmented Generation (RAG) in the context of LLM applications? A. A method to train language models faster. B. Using a search or database retrieval step to fetch relevant context, and providing that context to an LLM to ground its answer. C. Running an LLM on a very large input (retrieving all possible data at once). D. A type of prompt format for arithmetic reasoning.
Why is RAG useful for Q&A chatbots? A. It bypasses the need for an LLM entirely. B. It allows the LLM to provide up-to-date, specific information from a document set, reducing hallucination and extending knowledge beyond the model’s training. C. It significantly increases the LLM’s parameter count. D. It ensures the LLM will never make errors.
What are the main stages of a basic RAG pipeline? A. Ingestion (load and chunk documents into a vector index), Retrieval (find relevant chunks by similarity to the query), and Generation (the LLM produces an answer using the retrieved context). B. Training, Validation, and Deployment. C. Tokenization, Embedding, and Decoding. D. Authentication, Transformation, Response.
Why are documents split into chunks during the RAG ingestion phase? A. LLMs can only read short texts due to token limits, so splitting ensures each chunk can fit into the model’s context. B. To increase the total number of vectors for more storage usage. C. To make the embeddings more random. D. To ensure each word becomes its chunk.
What is an embedding in the context of NLP and semantic search? A. A fixed-length numeric vector that represents the semantic meaning of text (words, sentences, or documents). B. A hyperlink inside a document. C. A type of database index for keywords. D. A summary of a document.
How can you generate text embeddings using OCI’s services? A. By using the OCI Generative AI Embedding model endpoint to get vector representations of text. B. By running a Hadoop cluster on OCI. C. Only by using third-party libraries outside OCI. D. The OCI GenAI service does not support embeddings.
Oracle Database 23c introduced vector support. Which statement is true about Oracle’s vector store capability? A. It requires data to be stored as BLOBs; there is no special vector type. B. It provides a new VECTOR data type for columns to hold embedding vectors, and SQL functions to perform similarity search (e.g., cosine distance). C. It only works with image data, not text embeddings. D. Oracle 23c automatically trains LLMs inside the database.
Which similarity metric compares two text embedding vectors by measuring the angle between them while ignoring magnitude? A. Euclidean distance. B. Dot product. C. Cosine similarity. D. Manhattan distance.
What is the purpose of using an index (like HNSW or IVF) in a vector database? A. To compress the vectors into smaller dimensions. B. To accelerate similarity search by organizing vectors for faster nearest-neighbor lookup. C. To convert vectors back into text. D. To ensure exact keyword matching on vector data.
In LangChain, how do you combine the LLM and retrieval steps to implement RAG Q&A? A. Use a specialized chain (e.g, RetrievalQA) that automatically queries a vector store retriever for relevant documents and passes them to the LLM for answer generation. B. Manually call the database, then manually call the LLM in code. C. Fine-tune the LLM on the documents instead of retrieving. D. LangChain cannot handle that; you must write your pipeline.
In a multi-turn chatbot built with LangChain, what component is used to ensure the AI remembers previous conversation turns? A. The Memory module, which stores prior messages or a summary, so the LLM can incorporate past context into new answers. B. The VectorStore, which keeps all past dialogues as embeddings. C. A special prompt that repeats everything said so far (no built-in component). D. LangChain does this automatically without any configuration.
Oracle’s implementation of RAG often uses Oracle DB as the vector store. What must be ensured when using an Oracle Database as a knowledge base for RAG? A. The database version is 19c or lower. B. The same embedding model used to generate the chunk vectors is used to encode incoming queries, to ensure vectors are comparable. C. All text data must be in one large row. D. Only one query can be processed at a time.
What is a benefit of semantic search over traditional keyword (lexical) search in the RAG context? A. It finds results that are related in meaning, even if exact keywords differ or are not present. B. It is 100% precise with no irrelevant results. C. It ignores the actual content and matches only metadata. D. It’s always faster than keyword search.
If you set return_source_documents=True in a LangChain RetrievalQA chain, what happens? A. The LLM’s answer will include citations or the actual source text for transparency. B. The chain will output the retrieved documents (or their references) along with the answer, allowing you to see which sources were used. C. The LLM will quote entire documents in its answer. D. The chain will return only the documents and not answer the question.
Domain 4: OCI Generative AI Agents Service

What is the Oracle Cloud Generative AI Agents service? A. A fully managed service that lets you create LLM-powered agents (chatbots) that use large language models plus an intelligent retrieval system to answer queries using your enterprise data. B. A tool for training new foundation models. C. A hardware device for running AI models. D. A SaaS application for human call center agents.
In the context of OCI GenAI Agents, what is a Knowledge Base? A. A collection of rules that the agent follows. B. The vector-indexed datastore of your ingested content, which the agent can query for relevant information. C. The pre-trained knowledge of the base LLM. D. A log of all conversations the agent has had.
Which data source types are supported for populating an Oracle GenAI Agent’s knowledge base? (Choose 2) A. Object Storage bucket – the service can ingest PDF or text files from a bucket. B. OCI OpenSearch index – use an OCI Search with an OpenSearch index that’s already loaded with data. C. Oracle Database 23c vector store – bring your table of vectors and a similarity search function. D. On-premises Hadoop file system. (Note: Two options are correct.)
When using an Object Storage bucket as a knowledge base, which is NOT a requirement or limitation? A. Files must be in PDF or plain text format (up to 100 MB each). B. Only one bucket can be used per data source. C. Images within PDFs are ignored entirely (cannot be processed). D. Charts in PDFs should be 2D with labeled axes for the AI to interpret them.
You want to use an Oracle Autonomous Database as a knowledge base for an agent. What must you do? A. Manually convert your data to embeddings using an external tool first. B. Create a table with text and vector columns (for chunks and their embeddings) and implement a PL/SQL vector search function that the agent will call for retrieval. C. Export the entire database to JSON files and put them in Object Storage. D. It’s not possible to connect the GenAI Agent to a database.
What is the purpose of an Agent Endpoint in OCI Generative AI Agents? A. It’s the interface where you configure the agent’s personality. B. It is a deployment point that you create, which gives a stable REST endpoint or chat interface for interacting with your agent. C. It’s an OCI Monitoring alarm for the agent. D. It’s the vector database connection.
Which of the following operations can OCI’s Generative AI Agent perform that a basic LLM chatbot alone cannot? A. Maintain conversational context across turns. B. Call external tools or APIs (e.g., database queries, booking systems) as decided by the LLM’s reasoning. C. Generate text in English. D. Translate from English to French.
What is the function of the Trace feature in the GenAI Agents service? A. It logs and displays the sequence of steps the agent took for each user query – including the prompts, retrieved data, and LLM’s intermediate reasoning. B. It traces network packets for debugging connectivity. C. It summarizes the conversation after each turn. D. It re-trains the agent based on user feedback.
How does the agent provide citations in responses, and why is this useful? A. It outputs the embedding vector to prove that it used the knowledge base. B. It appends source titles/URLs and page references for facts in its answer, so users can verify information back to the original documents. C. It cites the LLM model (e.g., “Answer generated by Llama-2”). Citations are used to credit the developers of the agent.
What is content moderation in OCI GenAI Agents? A. A feature that filters out or masks hateful, harmful, or policy-violating content in user prompts or the agent’s responses. B. A way to limit how much content the agent can output (rate limiting). C. A process of curating which files go into the knowledge base. D. An Oracle support service for monitoring your agent.
What does enabling hybrid search for a knowledge base do? A. It stores half the data as vectors and half as text. B. It combines semantic vector search with keyword (lexical) search to improve result relevance. C. It uses two different LLMs for answering. D. It allows the agent to search the internet.
Answer Key:


A
B
C
B
A
A
B
B
A
C
A
B
A
B
B
A
B
A
B
B
B
B
A
B
B
B
C
A
A
A
B
B
A
A
A
A
B
C
B
A
A
B
A
B
A
B
A, B
C
B
B
B
A
B
A
B
Hard Questions Exam 2 OCI 2025 Generative AI Professional (1Z0-1127-25)

1. Knowledge Base Integration
Your team configures a GenAI Agent with an Object Storage knowledge base. The PDF files contain multilingual text, tables, and scanned diagrams. Which data elements will the Agent reliably ingest and use for retrieval?

A. All multilingual text content, including extracted text from embedded diagrams. B. Text-based content only; embedded diagrams and images will be ignored. C. Multilingual text only if explicitly labeled with UTF-8 encoding. D. Only English text content and labeled tables.

2. Model Selection
When choosing between OCI’s Cohere-based LLM and an OpenSearch-powered embedding model for an Agent, which task requires embedding models rather than generative LLMs?

A. Summarizing customer complaints. B. Classifying text into sentiment categories. C. Performing similarity search over large knowledge bases. D. Answering open-ended questions from users.

3. Vector Database Integration
An enterprise wants to use its own Oracle Database 23ai with vector support for semantic search in a GenAI Agent. Which statement is correct?

A. GenAI Agents can natively connect to Oracle Database vector stores as a knowledge base. B. GenAI Agents cannot directly use Database 23ai; data must first be staged into Object Storage or OpenSearch. C. GenAI Agents can use 23ai only for structured SQL queries, not vector similarity. D. GenAI Agents auto-convert 23ai vector tables into embeddings without configuration.

4. Fine-tuning Limits
Your team wants to fine-tune a foundation model with customer call transcripts. Which of the following is a limitation of OCI fine-tuning?

A. You can only fine-tune embedding models, not generative models. B. Fine-tuned models cannot be deployed into private subnets. C. Fine-tuned models inherit usage quotas and limits of the base model family. D. Fine-tuned models automatically overwrite the base model.

5. Governance
Which framework is explicitly referenced in OCI’s GenAI governance best practices for risk management?

A. COBIT 2019 B. NIST AI Risk Management Framework (AI RMF) C. ITIL v4 D. ISO 22301

6. Agent vs Chatbot
Which operation can an OCI Generative AI Agent perform that a standalone LLM chatbot cannot? A. Maintain conversational context. B. Generate multilingual text. C. Call external APIs/tools during reasoning. D. Perform text summarization.

7. Latency Optimization
You deploy a GenAI Agent that queries a 1 TB knowledge base in Object Storage. Latency is high. Which OCI-native optimization is recommended?

A. Convert all files to image-based PDFs for faster parsing. B. Pre-chunk documents into smaller text blocks before ingestion. C. Store files in multiple buckets and point the Agent to all of them. D. Increase the beam size of the generative model.

8. Model Deployment
You need to deploy a fine-tuned foundation model into a production environment that handles confidential healthcare data. The compliance team requires network isolation and no public internet exposure. Which OCI deployment option best satisfies this requirement?

A. Deploy the model in a public endpoint with VCN security lists blocking all inbound traffic. B. Deploy the model as a private endpoint within a VCN subnet, accessible only via private IPs. C. Use the base foundation model directly, since only fine-tuned models require isolation. D. Host the model on Object Storage with signed URLs for restricted access.

9. Model Privacy
Which statement is true about data sent to OCI Generative AI APIs?

A. Customer data may be retained for model training unless disabled in settings. B. Customer data is never used to train Oracle’s base foundation models. C. Data is stored for training by default for 30 days. D. Data retention is required unless the tenant uses a private subnet.

10. Retrieval-Augmented Generation (RAG)
In OCI GenAI, what is the role of the embedding model in a RAG pipeline?

A. Generate the final answer in natural language. B. Map user queries and documents into a shared vector space for similarity search. C. Tokenize text into subwords for faster LLM inference. D. Ensure data privacy by masking sensitive fields.

11. Multi-Agent Coordination
You design a system where one GenAI Agent handles customer inquiries, and another Agent handles ticket creation via API. What mechanism ensures the Agents can coordinate securely?

A. Use OCI Service Connectors with IAM policies. B. Use embedded prompts with role-based delegation. C. Use Agent-to-Agent handoff with OCI Functions as middleware. D. Configure both Agents in a single session with expanded context windows.

12. Cost Optimization
Your CIO complains about high costs from using the largest foundation models for all GenAI queries. Which OCI feature helps cut costs while balancing accuracy?

A. Autoscaling Object Storage buckets. B. Dynamic model selection with smaller models for lightweight tasks. C. Increasing beam size to reduce retries. D. Using Free Tier foundation models.

13. File Size Limits
What is the maximum file size for ingestion into an Object Storage bucket as a GenAI knowledge base?

A. 10 MB B. 50 MB C. 100 MB D. Unlimited, if partitioned

14. Tool Invocation
Which best describes how an Agent decides to call an external tool?

A. Tools are invoked only at fixed checkpoints configured by the developer. B. The LLM’s reasoning determines if/when to call the tool based on user input and context. C. Tools must be triggered manually by a system administrator. D. Agents cannot call external tools directly.

15. Multilingual Limits
Your Agent must handle queries in English, French, and Japanese. Which is a limitation?

A. Embedding models support English only. B. Generative models support English and French, but not Japanese. C. Embedding models support multilingual text, but accuracy may vary across languages. D. Multilingual support requires fine-tuning.

16. SLA Awareness
Which SLA condition applies to OCI Generative AI service availability?

A. GenAI inherits the same SLA as Object Storage. B. Oracle publishes distinct SLA metrics for GenAI APIs separate from core OCI services. C. SLAs apply only when using private foundation models. D. SLAs apply only to Agents, not base models.

17. Knowledge Base Sync
If you update a PDF in the Object Storage bucket linked to a GenAI knowledge base, what must you do to ensure the Agent uses the new version?

A. Nothing; Agents sync in real time with Object Storage. B. Re-trigger an ingestion job to refresh embeddings. C. Delete and recreate the entire knowledge base. D. Update the IAM policy attached to the Agent.

18. Model Hallucination
Which mitigation best reduces hallucination in OCI GenAI Agents? A. Increase the temperature parameter. B. Restrict generation to retrieval-augmented context only. C. Disable embeddings. D. Switch from generative to embedding-only responses.

19. IAM Roles
To allow a GenAI Agent to access data in an OCI Object Storage bucket, which IAM principle is required? A. Grant OBJECT_WRITE and OBJECT_DELETE privileges. B. Grant OBJECT_READ access to the Agent’s dynamic group. C. Grant tenancy-level ADMIN rights to the Agent. D. No IAM permissions are needed; Agents bypass IAM.

20. Model Drift
Your fine-tuned sentiment analysis model starts misclassifying slang terms. Which is the most appropriate mitigation in OCI?

A. Increase beam size. B. Collect new training data, including slang, and refine. C. Adjust Object Storage bucket policies. D. Switch to an embedding model.

Answer Key (for self-check)
1: B 2: C 3: B 4: C 5: B 6: C 7: B 8: B 9: B 10: B 11: C 12: B 13: C 14: B 15: C 16: B 17: B 18: B 19: B 20: B

Resources







Here’s a **cleanly formatted revision sheet** for the Oracle AI Professional Certification:

---

## 🧠 Oracle AI Professional Certification – Revision Sheet  
**Created to help you prepare and pass the exam**

---

### 🔹 Fundamentals

**1. What are Large Language Models (LLMs)?**  
- **Definition**: Probabilistic models of text that compute distributions over vocabulary.  
- **Core Function**: Given input text, predict the most likely next word(s).  
- **“Large”**: Refers to the number of trainable parameters—typically in the billions.

---

### 🔹 Architectures of LLMs

- **Encoders**: Convert text into embeddings (numeric representations of meaning).  
- **Decoders**: Generate text by predicting the next token.  
- **Encoder-Decoder Models**: Combine both; mainly used for translation and sequence-to-sequence tasks.

---

### 🔹 Prompting and Prompt Engineering

- **Prompting**: Changing model inputs to influence output.  
- **Prompt Engineering**: Refining inputs to get desired results.

---

### 🔹 Risks of Prompting

- **Prompt Injection**: Malicious manipulation of inputs to force unintended outputs (similar to SQL injection).  
- **Leakage**: Attackers can trick LLMs into revealing developer prompts.  
- **Data Risks**: Potential exposure of private or sensitive data from training.

---

### 🔹 Training Methods

- **Fine-Tuning**: Retraining all parameters for a specific task. Expensive but effective.  
- **Parameter-Efficient Fine-Tuning (PEFT)**: Train only a small subset of parameters (e.g., LoRA).  
- **Soft Prompting**: Train special tokens (prompt embeddings) without changing core parameters.  
- **Continual Pretraining**: Adapt the model to a new domain with large amounts of new text.

---

### 🔹 Decoding (Generating Text)

- **Greedy Decoding**: Always pick the most probable token.  
- **Random Sampling**: Pick words based on a probability distribution.  
- **Temperature**: Controls randomness; lower = deterministic, higher = more creative.  
- **Nucleus Sampling (Top-p)**: Restrict sampling to the most probable portion of the distribution.  
- **Beam Search**: Generate multiple sequences simultaneously, keep the best ones.

---

### 🔹 Key Challenges

- **Hallucinations**: Model generates fluent but incorrect information.  
- **Bias and Safety**: Outputs may reflect harmful or biased data from training.  
- **Cost**: Training very large models requires thousands of GPUs and significant resources.

---

Here’s a clean and professional formatting of your content on **OCI Generative AI Service**:

---

# **OCI Generative AI Service Overview**

## **1. Introduction to OCI Generative AI**

- **OCI Generative AI Service**: A fully managed, serverless platform for building generative AI applications.
- **Access**: Single API supporting multiple foundational models (e.g., **Cohere**, **Meta**).

---

## **2. Pre-trained Models**

### **Chat Models**
- **Command-R-Plus**
- **Command-R (16k)**
- **Meta Llama 3.1 / 3.2 / 3.3**

### **Embedding Models**
- Convert text into vector representations.
- **Use Cases**: Semantic search, clustering, classification, Retrieval-Augmented Generation (RAG).
- **Models**:
  - **Cohere embed-english** & **embed-multilingual** (supports 100+ languages)
  - **V3 embed models**: Improved retrieval quality for noisy datasets
- **Vector Dimensions**:
  - Standard: 1024-d
  - Lite: 384-d
- **Limits**:
  - 512 tokens per input
  - Max 96 inputs per run

---

## **3. Prompting & Prompt Engineering**

### **Definitions**
- **Prompt**: Input text provided to the LLM.
- **Prompt Engineering**: Refining inputs to achieve desired outputs.

### **Types of Prompting**
- **Zero-shot**: Only task description.
- **Few-shot (k-shot)**: Includes *k* examples.
- **Chain-of-Thought (CoT)**: Step-by-step reasoning.
- **Zero-shot CoT**: Add “Let’s think step by step.”
- **In-context learning**: Supply demonstrations within the prompt.

### **Prompt Parameters**
- **Preamble Override**: Change tone/style (e.g., pirate tone).
- **Temperature**: Controls randomness (0 = deterministic, 1 = diverse).
- **Top-k**: Select from top *k* highest probability tokens.
- **Top-p (nucleus sampling)**: Select from smallest set of tokens summing to *p*.
- **Frequency Penalty**: Reduces repetition based on frequency.
- **Presence Penalty**: Penalizes repeated tokens regardless of frequency.

---

## **4. Customization Options**

- **Training from scratch** is **not recommended** due to high cost and data requirements.

---

## **5. Fine-Tuning & Inference**

### **Fine-Tuning**
- Retraining an LLM (fully or partially) on domain-specific data.
- **Workflow**:
  1. Collect training data.
  2. Upload to Object Storage.
  3. Launch fine-tuning job (select base model + method).
  4. Training runs on GPU cluster.
  5. Store encrypted fine-tuned weights in Object Storage.
  6. Register model in OCI Generative AI.

### **Inference**
- Serving phase: Model generates predictions/answers from prompts.
- **Workflow**:
  1. User sends prompt via app → API.
  2. Model receives prompt.
  3. Generates output tokens using decoding strategies (greedy, top-k, top-p, temperature).
  4. Returns output to end-user.
  5. Optional: Logging, tracing, moderation, citations (for RAG).

### **Evaluation Metrics**
- **Perplexity**: Predictive accuracy (lower = better).
- **BLEU**: Translation/text similarity via n-gram overlap.
- **ROUGE**: Recall/overlap (used in summarization).
- **Accuracy / F1-score**: Classification tasks.
- **Loss Function**: Measures prediction error during training.

---

## **6. Dedicated AI Clusters**

### **Fine-Tuning Clusters**
- **Purpose**: Train/adapt models with custom data.
- **Billing**: Pay only for fine-tuning duration.

**Example**:  
Fine-tuning a Cohere Command-R model with 8 units over 12 hours:  
**Cost** = 8 units × hourly rate × 12 hrs

### **Hosting (Inference) Clusters**
- **Purpose**: Serve models in production.
- **Billing**: Minimum 744 hours/month (24×7).

**Example**:  
Hosting a Llama 13B model with 2 units for a customer support chatbot.

---
Here’s a professionally formatted version of your content on **LangChain, RAG, and Oracle 23ai Integration**:

---

# **LangChain, RAG, and Oracle 23ai Integration**

## **1. LangChain Overview**

**LangChain** is a framework for building LLM-powered applications that are context-aware.

### **Prompt Templates**
- **PromptTemplate**: For single-string inputs; ideal for simple LLMs.
- **ChatPromptTemplate**: Supports multi-turn conversational formats with structured roles (e.g., system, human, AI).

### **Memory in LangChain**
- LLMs are **stateless** by default—each query is processed independently.
- **Memory** enables context-aware responses by remembering previous interactions.
- LangChain manages how past inputs/outputs are stored and reused; it does **not** modify the LLM itself.

### **How Memory Works**
1. User asks a question → LangChain records it.
2. LLM generates a response → also recorded.
3. On the next interaction, LangChain retrieves conversation history or a summary.
4. This history is appended to the new prompt → LLM sees context.

---

## **2. OCI Generative AI + Oracle 23ai Integration**

- **Embeddings**: Generated using **OCI Generative AI Service**.
- **Oracle 23ai SELECT AI**: Converts natural language into SQL queries using OCI Generative AI.
- **Oracle 23ai Vector Store**:
  - Stores embeddings using the `VECTOR` datatype.
  - Supports **similarity search** for retrieval tasks.

---

## **3. Retrieval-Augmented Generation (RAG)**

### **Why RAG?**
- Traditional LLMs are limited by static training data (may be outdated or biased).
- **RAG** retrieves relevant, up-to-date information and feeds it as context to the LLM.

---

## **4. RAG Pipeline Phases**

### **Phases**
1. **Ingestion**: Load and split documents (e.g., `PDFReader`, `TextSplitter`).
2. **Embedding & Storage**: Use `OCIEmbeddingModel` and store in `OracleVS`.
3. **Retrieval**: Perform similarity search (e.g., *k=3*, cosine or dot product).
4. **Generation**: Use `RetrievalQA` + `ChatOCIGenAI` to generate answers.

### **Execution Flow**
- Query → Retrieve → Generate

---

## **5. Conversational Memory in RAG**

- **Chat**: A sequence of Q&A interactions.
- **Memory**: Maintains conversation history as context.
- LangChain provides memory classes to persist:
  - Summaries
  - Entities
  - Raw text

---

## **6. Vector Indexes for Similarity Search**

### **Why Use Indexes?**
- As the chunk store grows, brute-force similarity search becomes slow.
- **Vector indexes** narrow the search space for fast top-K retrieval.

### **Index Types in Oracle 23ai**

#### **1. HNSW (Hierarchical Navigable Small World)**
- **What it is**: Graph-based index linking vectors across layers.
- **Why use it**: Fast approximate nearest-neighbor (ANN) search with strong recall and low latency.
- **Trade-offs**: Requires memory; build time and memory scale with corpus size.
- **Best for**: Low-latency Q&A, semantic search.

#### **2. IVF (Inverted File Index)**
- **What it is**: Cluster-based index routing queries to relevant partitions.
- **Why use it**: Efficient at scale; avoids full scans.
- **Trade-offs**: Recall depends on partition probing; requires tuning.
- **Best for**: Large collections with predictable query times.

### **Similarity Metrics**
- **Dot Product**: Considers both magnitude and angle.
- **Cosine Similarity**: Considers angle only (magnitude-invariant).

### **RAG Flow Integration**
1. Embed chunks → store vectors in Oracle 23ai `VECTOR` column.
2. Index using **HNSW** or **IVF**.
3. Retrieve top matches using dot or cosine similarity.
4. Generate answer using LLM with retrieved context.

---
Here’s a clean and professional formatting of your content on **OCI Generative AI Agents**:

---

# **OCI Generative AI Agents**

## **1. Overview**

**OCI Generative AI Agents** is a fully managed service that combines LLMs with intelligent retrieval to deliver contextual, actionable responses.

- **Purpose**: Automates tasks such as booking, querying, and summarization by integrating:
  - Reasoning
  - Acting
  - Planning
  - Persona

---

## **2. Core Architecture**

- **Interface Options**: Chatbot, web app, voice, or API
- **External Knowledge**: Uses **RAG (Retrieval-Augmented Generation)** to ground responses in external data
- **Feedback Loop**: Responses feed into memory for improved context over time

---

## **3. Key Concepts**

- **Generative AI Model**: LLM trained for Natural Language Understanding (NLU) and Generation (NLG)
- **Agent**: Autonomous system built on LLM + RAG
- **Answerability**: Ability to respond relevantly to queries
- **Groundedness**: Responses are traceable to data sources (via citations)

---

## **4. Data Access Hierarchy**

- **Data Store**: Physical location of data (e.g., Object Storage, Oracle DB)
- **Data Source**: Connection details for accessing the store
- **Knowledge Base**: Vector storage system organizing ingested data for retrieval

---

## **5. Supported Data Options**

- **OCI Object Storage**: Managed ingestion; supports PDF & TXT files (≤100MB)
- **OCI OpenSearch**: Bring-your-own indexed data
- **Oracle DB 23ai Vector Store**: Custom embeddings with SQL-based retrieval functions

---

## **6. Data Ingestion**

- **Process**: Extract → Transform → Store into the knowledge base
- **Hybrid Search**: Combines lexical + semantic search for higher accuracy
- **Ingestion Jobs**: Add, retry, update, or cancel ingestion tasks

---

## **7. Additional Features**

- **Session**: Maintains context across exchanges (timeout range: 1 hour to 7 days)
- **Endpoint**: External access point for agents
- **Trace**: Logs full conversation history
- **Citation**: Displays sources of agent responses
- **Content Moderation**: Filters harmful content (input, output, or both)

---

## **8. Database Guidelines**

- Create vector tables with fields: `DOCID`, `body`, `vector`
- Ensure the **same embedding model** is used for both data and query
- Define retrieval functions that return: `DOCID`, `body`, `score`
- Supports similarity metrics:
  - **Cosine similarity**
  - **Euclidean distance**

---

## **9. Agent Workflow**

1. **Create Knowledge Base**  
   → Choose data store (Object Storage or Oracle DB)

2. **Ingest Data**  
   → PDFs, TXTs, or pre-vectorized data

3. **Create Agent**  
   → Define welcome message, select knowledge base

4. **Create Endpoint**  
   → Connect to apps; configure session, moderation, trace, citation

5. **Chat & Test**  
   → Query agent, view responses, citations, and trace logs

---

## **10. Default Limits (Highlights)**

- **1 KB** → 1 data source
- **Files**: Up to 1,000 per source (100MB each)
- **Sessions**: Default timeout = 3,600 seconds (extendable)

---
Here’s a professionally formatted version of your content on **Fine-Tuning GPU Sizes & Calculations (OCI Generative AI)**:

---

# **Fine-Tuning GPU Sizes & Calculations (OCI Generative AI)**

## **1. Why GPU Sizing Matters**

- Fine-tuning large language models (LLMs) requires specialized GPU clusters.
- Choosing the wrong size can lead to:
  - **Wasted cost**
  - **Poor performance**
- OCI provides **Dedicated AI Clusters** for this purpose.

---

## **2. GPU Cluster Options**

- OCI clusters are provisioned in **units**.
  - Each unit includes a set of GPUs and networking capacity.
- **Fine-tuning**:
  - Requires **multiple units** to handle large parameter updates.
- **Inference/Hosting**:
  - Can use fewer units.
  - Must commit to **744 hours/month** (24×7 availability).

### **Typical Configurations**
*(Details typically provided in article content)*

---

## **3. Cost & Time Estimation**

### **Key Factors**
- **Model size (parameters)** → Larger models need more GPU memory.
- **Dataset size (tokens)** → More tokens = longer training time.
- **Cluster units** → More units = faster training, but cost scales linearly.

### **Fine-Tuning Cost Formula**
```
Cost = Number of Units × Hourly Rate per Unit × Hours Run
```

#### **Example**
- 8 Cohere units
- 12-hour training job
- Rate = $1/hour/unit

👉 **Total Cost** = 8 × 1 × 12 = **$96**

---

## **4. Hosting (Inference) Calculations**

### **Hosting Cost Formula**
```
Cost = Number of Units × Hourly Rate per Unit × 744 hours/month
```

#### **Example**
- 2 units
- Hourly rate = $Y/unit

👉 **Monthly Cost** = 2 × Y × 744 = **1,488 × Y dollars/month**

---

## **5. Fine-Tuning Methods Explained**

OCI Generative AI supports multiple fine-tuning methods depending on the base model:

- **Vanilla Fine-Tuning**
- **LoRA (Low-Rank Adaptation)**
- **T-Few (Task-specific Few-shot Fine-Tuning)**

*(Details typically provided in article content)*

---

## **6. Performance Tuning Guidelines**

| Model Size        | Recommended Units |
|-------------------|-------------------|
| Small (≤13B)      | 4–8 units         |
| Medium (30–70B)   | 8–16+ units       |
| Large (100B+)     | 32+ units (distributed) |

### **Tips**
- Run **pilot jobs** with sample datasets to estimate time and cost.
- Use **LoRA** or **T-Few** for **parameter-efficient tuning** → reduces GPU needs.
- For **global availability**, deploy smaller hosting clusters per region (US, EU, APAC).

---

## **7. Exam Pointers**

- **Fine-tuning**: High GPU demand, short duration.
- **Hosting**: Fewer GPUs, always-on, billed monthly (744 hrs).
- **LoRA/T-Few**: Lower GPU usage, cost-effective.
- **Formula knowledge**:
  ```
  Units × Rate × Hours
  ```
  May appear in scenario-based questions.

---

# MOCK QUESTIONS
Here’s a well-formatted version of your **Practice Questions** with **answers and explanations** included for each:

---

## **Practice Questions: OCI Generative AI & LLM Concepts**

---

### **Q1. What does “large” in LLM primarily refer to?  
a) Vocabulary size  
b) Training dataset size  
c) Number of parameters  
d) Number of GPUs used**

**✅ Answer: c) Number of parameters**  
**Explanation**: The term "large" in LLM refers to the **number of parameters** in the model. These parameters define the model's capacity to learn and represent complex patterns in data.

In the context of **Large Language Models (LLMs)**, **parameters** refer to the internal variables that the model learns during training. These parameters are the core components that enable the model to understand and generate human-like text.

---

### 🔍 **What Are Parameters in an LLM?**

- **Definition**: Parameters are numerical values (typically weights and biases) in the neural network that determine how input data is transformed into output.
- **Function**: They control how the model processes language—how it understands context, syntax, semantics, and generates coherent responses.
- **Scale**: LLMs like GPT-3 or GPT-4 have **billions** (or even **trillions**) of parameters. For example:
  - GPT-3: ~175 billion parameters
  - GPT-4: Estimated to be even larger (exact number not publicly confirmed)

---

### 🧠 **Why Parameters Matter**

- **More parameters = more capacity** to learn complex patterns in language.
- They allow the model to:
  - Understand nuanced meanings
  - Maintain context across long conversations
  - Generate diverse and accurate responses

---

### 📊 **Analogy**

Think of parameters like the **settings on a massive control panel**. During training, the model adjusts these settings to best predict the next word or sentence based on the input it receives. The more settings (parameters) it has, the more finely it can tune its understanding.

---


---

### **Q2. Which architecture is best suited for semantic search?  
a) Decoder  
b) Encoder  
c) Encoder-Decoder  
d) Random Forest**

**✅ Answer: b) Encoder**  
**Explanation**: **Encoder architectures** (like BERT) are ideal for semantic search because they convert text into embeddings that capture meaning, enabling similarity comparisons.

---

### **Q3. Which of the following is an example of a decoder model?  
a) BERT  
b) GPT-4  
c) ResNet  
d) Word2Vec**

**✅ Answer: b) GPT-4**  
**Explanation**: **GPT-4** is a **decoder-only** transformer model designed for text generation. BERT is encoder-only, ResNet is for image tasks, and Word2Vec is not a transformer.

---

### **Q4. What is the main purpose of prompting?  
a) Changing model weights  
b) Altering model input to influence output  
c) Reducing model size  
d) Speeding up training**

**✅ Answer: b) Altering model input to influence output**  
**Explanation**: Prompting involves crafting inputs to guide the model toward desired outputs without changing its internal parameters.

---

### **Q5. Zero-shot prompting involves:  
a) Providing no task description  
b) Providing examples of the task  
c) Only providing the task description with no examples  
d) Feeding only numbers**

**✅ Answer: c) Only providing the task description with no examples**  
**Explanation**: In **zero-shot prompting**, the model is given a task description but **no examples**—it must infer the task from the prompt alone.

---

### **Q6. Which prompting technique encourages step-by-step reasoning?  
a) Chain-of-Thought prompting  
b) Zero-shot prompting  
c) Greedy prompting  
d) LoRA prompting**

**✅ Answer: a) Chain-of-Thought prompting**  
**Explanation**: **Chain-of-Thought (CoT)** prompting guides the model to reason step-by-step, improving performance on complex tasks.

---

### **Q7. Which training method adds new parameters without changing original ones?  
a) Fine-tuning  
b) LoRA  
c) Soft Prompting  
d) Continual Pretraining**

**✅ Answer: b) LoRA**  
**Explanation**: **LoRA (Low-Rank Adaptation)** introduces new trainable parameters while keeping the original model weights frozen, making it efficient and modular.

**LoRA (Low-Rank Adaptation)** is a technique used to fine-tune large language models (LLMs) efficiently by introducing a small number of trainable parameters, while keeping the original model weights frozen. It’s especially useful when working with very large models where full fine-tuning would be computationally expensive.

---

## 🔍 **What Is LoRA?**

**Low-Rank Adaptation (LoRA)** modifies the model by inserting small, trainable layers into the existing architecture. These layers learn task-specific adjustments without altering the original model weights.

---

## ⚙️ **How LoRA Works**

- Instead of updating all the weights in a large model, LoRA:
  - Adds **low-rank matrices** to certain layers (usually attention layers).
  - Only trains these new matrices.
  - Keeps the original model **frozen** (unchanged).
- During inference, the outputs from the original weights and the LoRA layers are **combined** to produce the final result.

---

## 📈 **Benefits of LoRA**

| Feature | Benefit |
|--------|---------|
| **Efficiency** | Requires far fewer GPU resources |
| **Modularity** | LoRA layers can be added/removed easily |
| **Cost-effective** | Reduces training time and cost |
| **Scalability** | Enables fine-tuning of very large models on modest hardware |

---

## 🧠 **Why “Low-Rank”?**

- In linear algebra, a **low-rank matrix** is one that can be approximated using fewer dimensions.
- LoRA leverages this idea to represent changes in model behavior using **compact, efficient representations**.

---

## 📌 **Use Cases**

- Domain adaptation (e.g., legal, medical, financial texts)
- Task-specific tuning (e.g., summarization, classification)
- Multi-lingual or multi-modal extensions

---

---

### **Q8. What does temperature control in decoding?  
a) The speed of model training  
b) The size of the vocabulary  
c) The randomness of token selection  
d) The number of parameters updated**

**✅ Answer: c) The randomness of token selection**  
**Explanation**: **Temperature** controls how random the output is—lower values make the model more deterministic, higher values increase diversity.

---

### **Q9. Which decoding method always picks the most likely token?  
a) Random Sampling  
b) Beam Search  
c) Greedy Decoding  
d) Nucleus Sampling**

**✅ Answer: c) Greedy Decoding**  
**Explanation**: **Greedy decoding** selects the token with the highest probability at each step, without considering alternative paths.

---

### **Q10. Which of the following is a risk of deploying LLMs?  
a) Faster processing  
b) Prompt Injection  
c) Higher accuracy  
d) Lower memory usage**

**✅ Answer: b) Prompt Injection**  
**Explanation**: **Prompt injection** is a security risk where malicious inputs manipulate the model’s behavior, potentially leading to unintended or harmful outputs.


---

## **Practice Questions: OCI Generative AI**

---

### **Q1. Which OCI chat model allows up to 128k tokens per input?  
a) Command-R (16k)  
b) Command-R-Plus  
c) Llama 3.1 (70B)  
d) Cohere embed-english**

**✅ Answer: b) Command-R-Plus**  
**Explanation**: **Command-R-Plus** supports up to **128k tokens**, making it suitable for long-context tasks like summarization and document analysis.

---

### **Q2. What is the main use case for embedding models?  
a) Conversational chat  
b) Text-to-image  
c) Semantic search  
d) Translation only**

**✅ Answer: c) Semantic search**  
**Explanation**: Embedding models convert text into vector representations, which are primarily used for **semantic search**, clustering, and retrieval tasks.

---

### **Q3. Which fine-tuning method updates only ~0.01% of weights?  
a) Vanilla fine-tuning  
b) T-Few  
c) LoRA  
d) Zero-shot prompting**

**✅ Answer: c) LoRA**  
**Explanation**: **LoRA (Low-Rank Adaptation)** introduces a small number of trainable parameters (~0.01%) while keeping the original model weights frozen, making it efficient and modular.

---

### **Q4. Which parameter controls the randomness of the output?  
a) Frequency penalty  
b) Temperature  
c) Top-k  
d) Context length**

**✅ Answer: b) Temperature**  
**Explanation**: **Temperature** controls the randomness in token selection during generation. Lower values produce more deterministic outputs; higher values increase diversity.

---

### **Q5. What is a major advantage of RAG?  
a) Eliminates the cost of GPUs  
b) Provides grounded answers with enterprise data  
c) Removes the need for tokenization  
d) Always reduces latency**

**✅ Answer: b) Provides grounded answers with enterprise data**  
**Explanation**: **Retrieval-Augmented Generation (RAG)** enhances LLMs by retrieving relevant external data, ensuring responses are **grounded and accurate**.

---

### **Q6. Which evaluation metric is better for generative AI?  
a) Accuracy  
b) Loss  
c) BLEU  
d) ROUGE**

**✅ Answer: d) ROUGE**  
**Explanation**: **ROUGE** measures the overlap between generated and reference text, making it ideal for evaluating **summarization and generative tasks**.

---

### **Q7. Which cluster type is used for Llama models?  
a) Large Cohere Dedicated  
b) Small Cohere Dedicated  
c) Large Meta Dedicated  
d) Embedding Cohere Dedicated**

**✅ Answer: c) Large Meta Dedicated**  
**Explanation**: **Llama models** are developed by Meta, and they are hosted on **Large Meta Dedicated clusters** in OCI for optimal performance.

---

### **Q8. Which parameter reduces repeated phrases based on count?  
a) Temperature  
b) Top-p  
c) Frequency penalty  
d) Presence penalty**

**✅ Answer: c) Frequency penalty**  
**Explanation**: **Frequency penalty** reduces the likelihood of repeating tokens based on how often they’ve already appeared in the output.

---

### **Q9. What is the minimum hosting commitment for OCI clusters?  
a) 1 hour  
b) 10 hours  
c) 744 hours  
d) 96 hours**

**✅ Answer: c) 744 hours**  
**Explanation**: OCI hosting clusters require a **minimum commitment of 744 hours/month**, equivalent to **24×7 availability**.

---

### **Q10. You are deploying a fine-tuned Llama model on OCI and need to ensure that your model weights are encrypted and only accessible within your tenancy. Which service should you configure to manage and control the encryption keys?  
a) OCI IAM  
b) OCI Object Storage  
c) OCI Key Management  
d) OCI Data Guard**

**✅ Answer: c) OCI Key Management**  
**Explanation**: **OCI Key Management** allows you to manage encryption keys securely, ensuring that **model weights stored in Object Storage** are encrypted and tenancy-restricted.

---

