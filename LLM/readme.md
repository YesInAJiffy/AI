# LLM

https://krrai77.medium.com/navigating-the-oci-generative-ai-professional-certification-strategies-and-personal-insights-94446d5e23f3

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

Here are the **maximum token limits** for each of the listed models in **OCI Generative AI**:

---

### **a) Command-R (16k)**  
**✅ Max Token Limit: 16,000 tokens**  
**Explanation**: This model supports a total prompt + response length of **16,000 tokens** per run. It’s optimized for conversational tasks and long-context interactions.[1](https://docs.oracle.com/en-us/iaas/Content/generative-ai/cohere-command-r-16k.htm)

---

### **b) Command-R-Plus**  
**✅ Max Token Limit: 128,000 tokens**  
**Explanation**: Command-R-Plus supports a **context length of up to 128k tokens**, making it ideal for long documents, multi-turn conversations, and complex reasoning tasks.[2](https://stackoverflow.com/questions/78494027/max-token-limit-for-cohere-command-r-model)

---

### **c) Llama 3.1 (70B)**  
**✅ Max Token Limit: Typically 8,192 to 32,000 tokens**  
**Explanation**: While exact limits may vary by deployment, **Meta’s Llama 3.1 (70B)** models generally support **8k to 32k tokens** depending on configuration. OCI may offer extended context versions, but 8k is a safe default unless otherwise specified.[3](https://docs.oracle.com/en-us/iaas/Content/generative-ai/chat-models.htm)

---

### **d) Cohere embed-english**  
**✅ Max Token Limit: 512 tokens per input**  
**Explanation**: Embedding models like **Cohere embed-english** are designed for converting text into vector representations. They support up to **512 tokens per input**, with batch limits (e.g., 96 inputs per run).[3](https://docs.oracle.com/en-us/iaas/Content/generative-ai/chat-models.htm)

---
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
Here’s a clear explanation of the three fine-tuning methods you asked about:

---

## **a) Vanilla Fine-Tuning**

### ✅ **What It Is**  
Vanilla fine-tuning refers to the **standard method** of retraining a pre-trained LLM by updating **all of its parameters** using new, task-specific data.

### 🔧 **How It Works**
- The entire model is retrained on a new dataset.
- Requires significant compute resources (GPU memory, time).
- Offers high flexibility and performance for domain adaptation.

### ⚠️ **Pros & Cons**
- **Pros**: High accuracy, full control over model behavior.
- **Cons**: Expensive, slow, and not modular—each fine-tuned version is a full copy of the model.

---

## **b) T-Few (Task-Specific Few-Shot Fine-Tuning)**

### ✅ **What It Is**  
T-Few is a **parameter-efficient fine-tuning** method that adapts models using **few-shot examples** and minimal updates.

### 🔧 **How It Works**
- Uses a small number of labeled examples.
- Fine-tunes only a subset of the model (e.g., adapters or specific layers).
- Designed for **task-specific adaptation** with minimal data.

### ⚠️ **Pros & Cons**
- **Pros**: Efficient, fast, good for low-resource scenarios.
- **Cons**: May not generalize well to broader tasks.

---

## **c) LoRA (Low-Rank Adaptation)**

### ✅ **What It Is**  
LoRA is a technique that adds **low-rank trainable matrices** to the model, allowing fine-tuning without modifying the original weights.

### 🔧 **How It Works**
- Original model weights are **frozen**.
- LoRA layers are inserted into attention or feed-forward layers.
- Only LoRA parameters are trained (~0.01% of total weights).

### ⚠️ **Pros & Cons**
- **Pros**: Highly efficient, modular, easy to deploy and share.
- **Cons**: Slightly less flexible than full fine-tuning for complex tasks.

---
Here’s a clear comparison between **T-Few** and **LoRA**, two popular **parameter-efficient fine-tuning** methods used in large language models (LLMs):

---

## 🔍 **T-Few vs. LoRA: Key Differences**

| Feature | **T-Few (Task-Specific Few-Shot Fine-Tuning)** | **LoRA (Low-Rank Adaptation)** |
|--------|-----------------------------------------------|-------------------------------|
| **Purpose** | Adapts models using few-shot examples for specific tasks | Efficiently fine-tunes models by adding low-rank matrices |
| **Training Scope** | Updates a small subset of parameters | Adds new trainable parameters; original weights are frozen |
| **Efficiency** | Very efficient for small datasets and quick adaptation | Highly efficient for large models with minimal compute |
| **Data Requirement** | Few-shot examples (small labeled dataset) | Can work with larger datasets but still efficient |
| **Model Modification** | Slight updates to existing layers | Inserts additional layers (low-rank adapters) |
| **Use Case** | Task-specific tuning (e.g., classification, QA) | Broad fine-tuning across domains or tasks |
| **Parameter Update Ratio** | Minimal (few layers or adapters) | ~0.01% of total parameters updated |
| **Deployment** | Lightweight, fast to deploy | Modular—LoRA layers can be added/removed easily |

---

## 🧠 **Summary**

- **T-Few** is ideal when you have **limited labeled data** and need to adapt a model quickly for a **specific task**.
- **LoRA** is better when you want to fine-tune **large models** efficiently without retraining the entire network, and you want to **preserve the original model**.

---

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

Here’s a clear explanation of the four evaluation metrics commonly used in machine learning and generative AI:

---

## 🔹 **a) Accuracy**

### ✅ What It Is:
Accuracy measures the **percentage of correct predictions** made by a model out of all predictions.

### 📊 Formula:
$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

### 📌 Use Case:
- Best for **classification tasks** where classes are balanced.
- Not ideal when dealing with **imbalanced datasets** (e.g., 95% of one class).

---

## 🔹 **b) Loss**

### ✅ What It Is:
Loss is a **numerical value** that represents how far off the model's predictions are from the actual values. It’s used during training to guide model updates.

### 📊 Common Loss Functions:
- **Cross-Entropy Loss** (for classification)
- **Mean Squared Error (MSE)** (for regression)

### 📌 Use Case:
- Used to **optimize model performance** during training.
- Lower loss = better model fit.

---

## 🔹 **c) BLEU (Bilingual Evaluation Understudy)**

### ✅ What It Is:
BLEU is a metric for evaluating **machine translation** and **text generation** by comparing n-gram overlaps between the generated text and reference text.

### 📊 How It Works:
- Measures precision of n-grams (e.g., 1-gram, 2-gram).
- Scores range from 0 to 1 (or 0 to 100 in some tools).

### 📌 Use Case:
- Common in **translation tasks**.
- Works well when exact word matching is important.

---

## 🔹 **d) ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

### ✅ What It Is:
ROUGE evaluates **text summarization** and **generation** by measuring the **recall** of overlapping units (n-grams, sequences, etc.) between generated and reference texts.

### 📊 Variants:
- **ROUGE-N**: Overlap of n-grams
- **ROUGE-L**: Longest common subsequence
- **ROUGE-S**: Skip-bigram

### 📌 Use Case:
- Ideal for **summarization** and **open-ended generation**.
- Focuses on how much of the reference content is captured.

---

## 🧠 Summary Table

| Metric | Best For | Measures | Notes |
|--------|----------|----------|-------|
| **Accuracy** | Classification | Correct predictions | Simple but limited for imbalanced data |
| **Loss** | Training optimization | Prediction error | Guides model updates |
| **BLEU** | Translation | N-gram precision | Sensitive to exact wording |
| **ROUGE** | Summarization | N-gram recall | Captures content overlap |


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

## 🧮 **Breakdown of 744 Hours**

- There are **31 days** in many months.
- Each day has **24 hours**.
  
So:

$$
31 \text{ days} \times 24 \text{ hours/day} = 744 \text{ hours/month}
$$

---

## 🧠 **Why This Matters in OCI Hosting**

When you deploy a model for **inference (hosting)** in **OCI Generative AI**, it's expected to be **available at all times**—ready to respond to user queries, API calls, or chatbot interactions.

- This is called **“always-on” hosting**.
- OCI bills hosting clusters based on this **full-time availability**, hence the **744-hour minimum commitment**.

---

## ✅ **Summary**

| Hosting Mode | Hours/Month | Description |
|--------------|-------------|-------------|
| **24×7 Hosting** | **744 hours** | Model is continuously available for production use |
| **Fine-Tuning** | Flexible (e.g., 12 hours) | Short-term GPU usage for training |


---

### **Q10. You are deploying a fine-tuned Llama model on OCI and need to ensure that your model weights are encrypted and only accessible within your tenancy. Which service should you configure to manage and control the encryption keys?  
a) OCI IAM  
b) OCI Object Storage  
c) OCI Key Management  
d) OCI Data Guard**

**✅ Answer: c) OCI Key Management**  
**Explanation**: **OCI Key Management** allows you to manage encryption keys securely, ensuring that **model weights stored in Object Storage** are encrypted and tenancy-restricted.

---


## **Practice Questions: LangChain, RAG, and Oracle 23ai**

---

### **Q1. Which LangChain component is used to preserve past conversation for context?  
a) Prompt Template  
b) Memory  
c) Chain  
d) Vector Store**

**✅ Answer: b) Memory**  
**Explanation**: In LangChain, **Memory** is used to store previous interactions so the model can generate context-aware responses in multi-turn conversations.

---

### **Q2. In LangChain, which template is designed for conversational inputs?  
a) Prompt Template  
b) ChainTemplate  
c) ChatPromptTemplate  
d) DialogueTemplate**

**✅ Answer: c) ChatPromptTemplate**  
**Explanation**: **ChatPromptTemplate** is specifically designed for multi-turn conversations, allowing structured messages with roles like system, human, and AI.
### 🔍 **What Is `ChatPromptTemplate` in LangChain?**

`ChatPromptTemplate` is a specialized prompt constructor in **LangChain** designed for **multi-turn conversational applications**. It allows developers to structure prompts in a way that mimics real dialogue, using **roles** like `system`, `human`, and `AI`.

---

## 🧠 **Why Use `ChatPromptTemplate`?**

Unlike a simple `PromptTemplate` (which is a single string input), `ChatPromptTemplate` is built for **chat-based models** (like OpenAI's GPT or Cohere Command-R), where the model expects a **sequence of messages** with context.

---

## 🧩 **Key Features**

- **Role-based formatting**: Supports `system`, `human`, and `AI` roles.
- **Multi-turn support**: Maintains conversation history across turns.
- **Dynamic input**: Allows insertion of variables (e.g., user queries, context).
- **Integration with memory**: Works well with LangChain's memory components to preserve context.

---

## 📦 **Example Usage**

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])
```

When rendered with:
```python
prompt.format(user_input="What is the capital of France?")
```

It produces:
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"}
]
```

This format is ideal for chat models that expect structured input.

---

## ✅ **Use Cases**

- Chatbots
- Customer support agents
- Conversational RAG systems
- Multi-turn QA applications

---

In **LangChain**, templates are used to structure prompts that are sent to language models. Each template serves a different purpose depending on the type of interaction or task. Here's an overview of the main templates:

---

## 🔹 **1. PromptTemplate**

### ✅ Purpose:
Used for **single-turn prompts**—simple tasks where the input is a single string.

### 📦 Example:
```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("Translate the following to French: {text}")
```

### 📌 Use Case:
- Text classification
- Translation
- Summarization (single input)

---

## 🔹 **2. ChatPromptTemplate**

### ✅ Purpose:
Designed for **multi-turn conversations**, especially with chat-based models like OpenAI's GPT or Cohere's Command-R.

### 📦 Example:
```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
```

### 📌 Use Case:
- Chatbots
- Conversational agents
- RAG with dialogue memory

---

## 🔹 **3. FewShotPromptTemplate**

### ✅ Purpose:
Used to provide **few-shot examples** in the prompt to guide the model’s behavior.

### 📦 Example:
```python
from langchain.prompts import FewShotPromptTemplate

examples = [{"input": "2+2", "output": "4"}, {"input": "3+5", "output": "8"}]
template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate.from_template("Q: {input}\nA: {output}"),
    prefix="Answer the following math questions:",
    suffix="Q: {question}\nA:"
)
```

### 📌 Use Case:
- Math reasoning
- Code generation
- Task-specific adaptation

---

## 🔹 **4. StructuredPromptTemplate** *(less commonly used)*

### ✅ Purpose:
Used when you need to enforce a specific structure or format in the prompt, often for APIs or structured outputs.

---

## 🧠 Summary Table

| Template Type         | Best For                        | Supports Multi-Turn | Example Use Case         |
|-----------------------|----------------------------------|----------------------|---------------------------|
| **PromptTemplate**     | Simple tasks                    | ❌                   | Translation, classification |
| **ChatPromptTemplate** | Conversational agents           | ✅                   | Chatbots, RAG             |
| **FewShotPromptTemplate** | Few-shot learning examples     | ❌                   | Math, code generation     |
| **StructuredPromptTemplate** | Structured output formatting | ❌                   | API calls, JSON responses |

---


---

### **Q3. Which Oracle feature allows you to query databases using natural language?  
a) Oracle Cloud Guard  
b) Oracle Data Guard  
c) Oracle SELECT AI  
d) Oracle AutoML**

**✅ Answer: c) Oracle SELECT AI**  
**Explanation**: **Oracle SELECT AI** enables users to write **natural language queries** that are automatically converted into SQL using OCI Generative AI.
### 🔍 **What Is Oracle SELECT AI?**

**Oracle SELECT AI** is a feature in **Oracle 23ai** that allows users to **query databases using natural language** instead of writing traditional SQL. It leverages **OCI Generative AI** to interpret user intent and automatically generate accurate SQL queries.

---

## 🧠 **Why It Matters**

- **Simplifies data access** for non-technical users
- **Accelerates analytics** by removing the need to learn SQL
- **Improves productivity** across business and technical teams

---

## ⚙️ **How It Works**

1. **User Input**: You type a question like  
   _“Show me the top 5 customers by revenue in Q2.”_

2. **AI Interpretation**: SELECT AI uses **LLMs** to understand the query.

3. **SQL Generation**: It automatically generates the corresponding SQL query.

4. **Execution**: The query runs against the connected Oracle database.

5. **Results**: You get the data—no SQL knowledge required.

---

## 📦 **Key Features**

- **Natural Language to SQL**: Converts plain English into executable queries.
- **Context-Aware**: Understands schema, table relationships, and business logic.
- **Secure**: Respects user roles and access controls.
- **Integrated with OCI Generative AI**: Uses Oracle’s AI models for interpretation.

---

## ✅ **Use Cases**

- Business analysts querying sales or finance data
- Customer support teams accessing CRM insights
- Executives generating reports without technical help
- Developers speeding up prototyping and testing

---

## 🔐 **Security & Governance**

- SELECT AI respects **database permissions** and **user roles**.
- Queries are executed within the boundaries of what the user is allowed to access.

---


---

### **Q4. Which similarity metric considers both magnitude and angle between embeddings?  
a) Cosine Similarity  
b) Dot Product  
c) Euclidean Distance  
d) Jaccard Index**

**✅ Answer: b) Dot Product**  
**Explanation**: **Dot product** takes into account both the **magnitude and angle** between vectors, making it useful for certain types of similarity search.

---

### **Q5. Why is chunk overlap used in text splitting?  
a) To reduce cost  
b) To maintain semantic continuity  
c) To increase retrieval speed  
d) To avoid indexing**

**✅ Answer: b) To maintain semantic continuity**  
**Explanation**: **Chunk overlap** ensures that important context isn’t lost between chunks, improving the quality of retrieval and generation in RAG pipelines.

---

### **Q6. What type of index is HNSW?  
a) Partition-based index  
b) Graph-based neighbor index  
c) Flat index  
d) Semantic index**

**✅ Answer: b) Graph-based neighbor index**  
**Explanation**: **HNSW (Hierarchical Navigable Small World)** is a **graph-based index** that enables fast approximate nearest-neighbor search with high recall.
### 🔍 What Is **HNSW (Hierarchical Navigable Small World)**?

**HNSW** is a **graph-based indexing algorithm** used for **approximate nearest neighbor (ANN)** search in high-dimensional vector spaces. It’s widely used in **semantic search**, **retrieval-augmented generation (RAG)**, and **recommendation systems** where fast and accurate similarity search is critical.

---

## 🧠 Why HNSW Matters

When you store text embeddings (vectors) in a database or vector store, searching for the most similar ones can be slow if done by brute force. **HNSW** solves this by organizing vectors into a **multi-layer graph**, enabling **fast top-K retrieval** with high accuracy.

---

## ⚙️ How HNSW Works

1. **Graph Structure**:
   - Vectors are connected in a graph where each node links to its nearest neighbors.
   - The graph is **hierarchical**, with multiple layers:
     - Top layers have fewer nodes and longer links (for fast traversal).
     - Lower layers have dense connections (for fine-grained search).

2. **Search Process**:
   - Starts at the top layer and navigates down.
   - Uses greedy search to move closer to the query vector.
   - At the bottom layer, it performs a more detailed search among neighbors.

3. **Insertion**:
   - New vectors are added by connecting them to existing nodes based on similarity.
   - The graph updates dynamically while maintaining efficiency.

---

## 📈 Benefits of HNSW

| Feature | Benefit |
|--------|---------|
| **Speed** | Very fast retrieval even in large datasets |
| **Accuracy** | High recall compared to other ANN methods |
| **Scalability** | Handles millions of vectors efficiently |
| **Low Latency** | Ideal for real-time applications like chatbots and search |

---

## ⚠️ Trade-Offs

- **Memory Usage**: Requires more memory to store the graph structure.
- **Build Time**: Index construction can be slower for very large datasets.
- **Tuning**: Parameters like `M` (max connections) and `ef` (search depth) affect performance.

---

## ✅ Use Cases

- **RAG (Retrieval-Augmented Generation)** in LLMs
- **Semantic search** in enterprise knowledge bases
- **Recommendation engines**
- **Image and audio similarity search**

---
---

### **Q7. In RAG, what is the role of embeddings?  
a) Tokenize text into smaller parts  
b) Convert text into numerical vectors for semantic similarity  
c) Generate SQL queries  
d) Reduce context window length**

**✅ Answer: b) Convert text into numerical vectors for semantic similarity**  
**Explanation**: Embeddings transform text into **vector representations**, which are used to find semantically similar content during retrieval.

---

### **Q8. Which LangChain class is used for chaining retrieval + LLM together?  
a) ChatPromptTemplate  
b) RetrievalQA  
c) MemoryChain  
d) OracleVS**

**✅ Answer: b) RetrievalQA**  
**Explanation**: **RetrievalQA** combines a retriever (e.g., vector store) with an LLM to answer questions based on retrieved context.
### 🔍 What Is **RetrievalQA** in LangChain?

**RetrievalQA** is a high-level class in **LangChain** that combines a **retriever** (like a vector store or search engine) with a **language model (LLM)** to answer questions based on external documents or data.

---

## 🧠 Why Use RetrievalQA?

LLMs are powerful, but they don’t always have access to **up-to-date or domain-specific information**. RetrievalQA solves this by:

- **Retrieving relevant documents** from a knowledge base
- **Feeding them into the LLM** as context
- **Generating grounded answers** based on that context

This is the core of **Retrieval-Augmented Generation (RAG)**.

---

## ⚙️ How RetrievalQA Works

1. **User asks a question**  
   → e.g., “What are the benefits of LoRA fine-tuning?”

2. **Retriever searches** the knowledge base  
   → Finds relevant documents or chunks using embeddings and similarity search

3. **LLM receives the retrieved context**  
   → Uses it to generate a well-informed answer

4. **Answer is returned to the user**

---

## 📦 Example in Code

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

retriever = FAISS.load_local("my_vector_store").as_retriever()
llm = ChatOpenAI()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
response = qa_chain.run("What is RetrievalQA?")
```

---

## ✅ Use Cases

- Enterprise search
- Chatbots with document grounding
- Legal, medical, or financial Q&A
- Internal knowledge base assistants

---

### 🔍 What Does **QA** Stand For in `RetrievalQA`?

**QA** stands for **Question Answering**.

---

## 🧠 What Is RetrievalQA?

`RetrievalQA` is a LangChain class that enables **question answering** by combining:

- A **retriever** (e.g., vector store or search engine)  
- A **language model (LLM)**

It retrieves relevant documents based on a user’s question and then uses the LLM to generate an answer grounded in that retrieved context.

---

## ✅ Summary

| Term | Meaning |
|------|--------|
| **QA** | Question Answering |
| **RetrievalQA** | A LangChain tool for answering questions using retrieved external data and an LLM |


---

### **Q9. What is a key benefit of RAG compared to training larger LLMs?  
a) Reduces context window size  
b) Eliminates need for embeddings  
c) Provides up-to-date, domain-specific answers without retraining  
d) Increases training dataset size**

**✅ Answer: c) Provides up-to-date, domain-specific answers without retraining**  
**Explanation**: **RAG** retrieves external data at runtime, allowing LLMs to generate **grounded, current responses** without needing to retrain the model.

---

### **Q10. Which Oracle 23ai datatype is used for storing embeddings?  
a) BLOB  
b) JSON  
c) VECTOR  
d) VARCHAR**

**✅ Answer: c) VECTOR**  
**Explanation**: Oracle 23ai uses the **VECTOR** datatype to store embeddings, enabling fast similarity search and retrieval in AI-powered applications.

---

Here’s a well-formatted version of your **OCI Generative AI Agents Practice Questions (MCQs)** with **answers and explanations**:

---

## **Practice Questions: OCI Generative AI Agents**

---

### **Q1. What is the role of the Knowledge Base in OCI Generative AI Agents?  
a) Provides APIs for LLM  
b) Organizes ingested data for retrieval  
c) Stores only prompts  
d) Manages system logs**

**✅ Answer: b) Organizes ingested data for retrieval**  
**Explanation**: The **Knowledge Base** is a vector storage system that organizes ingested data so it can be efficiently retrieved during inference.

---

### **Q2. Which feature ensures model responses are traceable to original data sources?  
a) Trace  
b) Persona  
c) Groundedness  
d) Content Moderation**

**✅ Answer: c) Groundedness**  
**Explanation**: **Groundedness** ensures that responses are based on actual data sources, often with citations, making them verifiable and trustworthy.

---

### **Q3. Which input helps maintain continuity in conversations?  
a) Prompt  
b) Tools  
c) Memory  
d) Citation**

**✅ Answer: c) Memory**  
**Explanation**: **Memory** stores previous interactions, allowing the agent to maintain context and continuity across multiple exchanges.

---

### **Q4. Which OCI service allows uploading PDF/TXT files for ingestion by Generative AI Agents?  
a) OCI IAM  
b) OCI Object Storage  
c) OCI Vault  
d) OCI Logging**

**✅ Answer: b) OCI Object Storage**  
**Explanation**: **OCI Object Storage** is used to upload and manage files like PDFs and TXTs, which can then be ingested into the Knowledge Base.

---

### **Q5. Hybrid search combines:  
a) Semantic + Exact match search  
b) Vector + Image search  
c) RAG + Prompt engineering  
d) SQL + Graph search**

**✅ Answer: a) Semantic + Exact match search**  
**Explanation**: **Hybrid search** blends **semantic (vector-based)** and **lexical (exact match)** techniques to improve retrieval accuracy.

### 🔍 What Is **Hybrid Search**?

**Hybrid search** is a technique that combines **semantic search** (meaning-based) and **lexical search** (exact keyword match) to improve the **accuracy and relevance** of information retrieval.

---

## 🧠 Why Hybrid Search Matters

In many real-world applications—like chatbots, enterprise search, or Retrieval-Augmented Generation (RAG)—you want to retrieve documents that are:

- **Semantically relevant** (similar in meaning)
- **Lexically precise** (contain exact terms or phrases)

Hybrid search ensures you get **both**.

---

## ⚙️ How Hybrid Search Works

1. **Lexical Search**:
   - Uses traditional keyword matching (e.g., SQL `LIKE`, inverted indexes).
   - Fast and precise but may miss contextually relevant results.

2. **Semantic Search**:
   - Uses **embeddings** to find documents with similar meaning.
   - Captures context but may miss exact matches.

3. **Hybrid Search**:
   - Combines both methods.
   - Ranks results based on a **weighted score** from both lexical and semantic relevance.

---

## ✅ Use Cases

- **Generative AI Agents** in Oracle 23ai
- **Enterprise knowledge bases**
- **Customer support bots**
- **Legal and medical document retrieval**

---

## 📌 Example

Imagine searching for:  
**"How to fine-tune a language model using LoRA?"**

- **Lexical search** finds documents with the exact phrase “fine-tune” or “LoRA”.
- **Semantic search** finds documents that discuss model adaptation, even if they don’t use those exact words.
- **Hybrid search** gives you the best of both—relevant and precise.

---

---

### **Q6. Which component defines the connection details for data retrieval?  
a) Knowledge Base  
b) Data Store  
c) Data Source  
d) Embedding Model**

**✅ Answer: c) Data Source**  
**Explanation**: The **Data Source** contains the connection details needed to access the underlying **Data Store** (e.g., Object Storage, DB).

---

### **Q7. In Oracle DB guidelines, what must align between queries and stored vectors?  
a) File formats  
b) Table schemas  
c) Embedding models  
d) Indexing methods**

**✅ Answer: c) Embedding models**  
**Explanation**: The **embedding model** used for querying must match the one used to generate and store the vectors to ensure accurate similarity search.

---

### **Q8. What is the default session timeout for OCI Agents?  
a) 300 seconds  
b) 1 hour  
c) 24 hours  
d) 7 days**

**✅ Answer: b) 1 hour**  
**Explanation**: The default **session timeout** for OCI Generative AI Agents is **1 hour**, though it can be extended up to 7 days.
### 🔍 **What Is the Timeout for OCI Generative AI Agents?**

The **default session timeout** for OCI Generative AI Agents is:

> **3600 seconds (1 hour)**  
> This means if there is **no activity** between the user and the agent for **1 hour**, the session automatically ends, and the context is lost.[1](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/generative-ai-agents/get-endpoint.htm)[2](https://docs.oracle.com/en-us/iaas/Content/generative-ai-agents/limits.htm)

---

### ⏳ **Can It Be Extended?**

Yes! You can configure the **idle timeout** to be as long as:

> **7 days (604,800 seconds)**  
> This allows the agent to retain context across longer periods of inactivity, which is useful for extended workflows or asynchronous interactions.[2](https://docs.oracle.com/en-us/iaas/Content/generative-ai-agents/limits.htm)

---

### ✅ **Summary**

| Setting | Value |
|--------|-------|
| **Default Timeout** | 1 hour (3600 seconds) |
| **Maximum Timeout** | 7 days |
| **Effect** | Session ends after timeout → context is lost |

---

### **Q9. Which feature tracks the full conversation history for monitoring?  
a) Trace  
b) Session  
c) Content Moderation  
d) Endpoint**

**✅ Answer: a) Trace**  
**Explanation**: **Trace** logs the entire conversation history, which is useful for debugging, monitoring, and auditing agent interactions.

---

### **Q10. Which search method is used for meaning-based retrieval?  
a) Lexical  
b) Semantic  
c) Hybrid  
d) Full-text**

**✅ Answer: b) Semantic**  
**Explanation**: **Semantic search** uses embeddings to retrieve content based on meaning rather than exact keyword matches.

---

Here’s a well-formatted version of your **Domain 1: LLM Fundamentals Practice Questions**, with **answers and explanations** provided after each question:

---

## **Domain 1: LLM Fundamentals (Large Language Models)**

---

### **Q1. What best describes a Large Language Model (LLM)?  
A. A neural network trained on massive amounts of text data to predict and generate language.  
B. A rule-based program for understanding grammar and syntax.  
C. A database that stores and retrieves large text documents.  
D. A cloud service for translating languages in real-time.**

**✅ Answer: A**  
**Explanation**: An LLM is a **neural network** trained on vast text data to understand, predict, and generate human-like language.

---

### **Q2. Which statement correctly contrasts encoder vs. decoder model architectures?  
A. Encoders generate long passages of text, whereas decoders produce only embeddings.  
B. Encoders excel at transforming text into vector representations, while decoders excel at generating text from prompts.  
C. Decoder models are used only for image data, encoders only for text data.  
D. Decoders require labeled data to operate; encoders do not.**

**✅ Answer: B**  
**Explanation**: **Encoders** are used for tasks like classification and semantic search (embedding generation), while **decoders** are used for text generation.

---

### **Q3. You want an LLM to translate a sentence from English to French. Which prompt engineering approach provides the model the best guidance?  
A. Zero-shot prompting – simply ask for the translation with no examples.  
B. One-shot prompting – provide the English sentence once and ask for French.  
C. Few-shot prompting – include a couple of example English–French translations in the prompt before asking for the new translation.  
D. Chain-of-thought prompting – instruct the model to explain its reasoning step by step in French.**

**✅ Answer: C**  
**Explanation**: **Few-shot prompting** gives the model examples to learn from, improving accuracy for translation and other structured tasks.

---

### **Q4. What is a known limitation of large language models that RAG (Retrieval-Augmented Generation) aims to address?  
A. LLMs often refuse to answer questions due to a lack of training data.  
B. LLMs may produce outdated or incorrect facts (hallucinations) because their knowledge is limited to training data.  
C. LLMs cannot handle more than a single question at a time.  
D. LLMs are unable to generate any content without an external knowledge base.**

**✅ Answer: B**  
**Explanation**: **RAG** helps mitigate **hallucinations** by retrieving up-to-date, external data to ground the model’s responses.

---

### **Q5. During deployment, how can an LLM be vulnerable to prompt injection?  
A. Users can craft inputs that trick the model into ignoring system instructions and producing disallowed output.  
B. The model’s weights can be directly altered via certain prompts.  
C. Prompt injection refers to overloading the model with too long a prompt.  
D. It’s a method of improving model accuracy by injecting clarifying prompts.**

**✅ Answer: A**  
**Explanation**: **Prompt injection** is a security risk where malicious inputs override system instructions, leading to unintended or unsafe outputs.

---

### **Q6. Which of the following is an example of a code model (LLM specialized for code)?  
A. A transformer model trained on GitHub code that can complete and generate functions in Python or Java.  
B. An LLM fine-tuned on medical research papers for clinical Q&A.  
C. A generative adversarial network (GAN) that produces source code.  
D. A compiler that translates high-level code to machine code using AI.**

**✅ Answer: A**  
**Explanation**: Code models like **Codex** or **CodeLlama** are trained on programming data to generate and complete code.

---

### **Q7. What does a multi-modal LLM refer to?  
A. An AI model that can handle more than one human language (multilingual model).  
B. A model that processes and/or generates multiple types of data (e.g., text, images, audio).  
C. Any LLM with over 100 billion parameters.  
D. A model using multiple neural network architectures in parallel.**

**✅ Answer: B**  
**Explanation**: **Multi-modal LLMs** can handle **multiple data types**, such as text, images, audio, and video (e.g., GPT-4 with vision).

---

### **Q8. Which hyperparameter would you adjust to make an LLM’s output less random and more deterministic?  
A. Increase the temperature.  
B. Decrease the temperature.  
C. Increase the max token limit.  
D. Use a larger context window.**

**✅ Answer: B**  
**Explanation**: **Lowering the temperature** makes the model’s output more predictable and consistent.

---

### **Q9. Your LLM outputs are good, but sometimes too brief. Which parameter or method helps in getting longer, more detailed responses?  
A. Increase the max_tokens (generation length limit) for the model’s output.  
B. Lower the temperature to 0.0.  
C. Use zero-shot prompting instead of few-shot.  
D. Enable beam search with a beam size of 1.**

**✅ Answer: A**  
**Explanation**: Increasing **max_tokens** allows the model to generate **longer responses** before stopping.

---

### **Q10. Which training approach allows adapting a large pre-trained LLM to a new task by updating only a small number of additional weights?  
A. Full fine-tuning – update all model parameters on the new task data.  
B. Prompt engineering – no model parameters are updated at all.  
C. Parameter-efficient tuning (e.g., LoRA) – introduce small trainable weight matrices or “adapters” instead of retraining the whole model.  
D. Continual pretraining – train on large unlabeled data from scratch.**

**✅ Answer: C**  
**Explanation**: **LoRA** and similar methods allow efficient adaptation by updating only a **small subset of parameters**, saving compute and time.

---
# STUDY FROM HERE

Here is the **reformatted version** of your questions with the **correct answer included after each full question and all options**:

---

### **1. Which is true about the OCI Generative AI service?**  
A. It’s a fully managed, serverless service that provides access to large language models via a unified API.  
B. It requires customers to bring their own GPU hardware.  
C. It is only for computer vision tasks, not text generation.  
D. It cannot be accessed through the OCI Console, only via API.  
✅ **Correct Answer:** A. It’s a fully managed, serverless service that provides access to large language models via a unified API.

---

### **2. What does it mean that OCI GenAI provides “single API” access to multiple models?**  
A. You must integrate separate APIs for each foundation model vendor.  
B. The same unified endpoint and API format lets you switch between different underlying models with minimal code changes.  
C. All users share one global API key for the service.  
D. The API only supports one model at a time.  
✅ **Correct Answer:** B. The same unified endpoint and API format lets you switch between different underlying models with minimal code changes.

---

### **3. Which foundation models are available out of the box in OCI’s Generative AI Service for text tasks?**  
A. Models from Meta (Llama 2) and Cohere (Command family for chat; Embed for embeddings).  
B. Only Oracle’s proprietary LLMs are trained in-house.  
C. OpenAI’s GPT-4 and GPT-3 models.  
D. Google’s PaLM model family.  
✅ **Correct Answer:** A. Models from Meta (Llama 2) and Cohere (Command family for chat; Embed for embeddings).

---

### **4. What is the primary intended use of the embedding models in OCI GenAI?**  
A. To generate conversational dialogue responses.  
B. To convert text into vector representations for semantic search and similarity tasks.  
C. To fine-tune on code datasets.  
D. To translate documents between languages.  
✅ **Correct Answer:** B. To convert text into vector representations for semantic search and similarity tasks.

---

### **5. You need to analyze a large PDF (300 pages) for semantic search. Which model and approach should you use on OCI GenAI?**  
A. Use a chat completion model to directly input all 300 pages as a prompt.  
B. Use the embedding model to convert chunks of the document into vectors, then use similarity search for relevant content.  
C. Fine-tune the chat model on the PDF content first.  
D. This is not possible with the OCI GenAI service.  
✅ **Correct Answer:** B. Use the embedding model to convert chunks of the document into vectors, then use similarity search for relevant content.

---

### **6. The Cohere Command-R vs. Command-R-Plus models in OCI GenAI differ primarily in what way?**  
A. The R-Plus model supports a far larger context window (prompt size up to 128k tokens) and higher performance, whereas Command-R is limited to 16k context.  
B. Command-R-Plus handles only code, Command-R handles only text.  
C. Command-R is for English, R-Plus is for multilingual tasks.  
D. R-Plus is cheaper to run but less capable than R.  
✅ **Correct Answer:** A. The R-Plus model supports a far larger context window (prompt size up to 128k tokens) and higher performance, whereas Command-R is limited to 16k context.

---

### **7. Which is a supported use case of OCI’s pre-trained foundation models?**  
A. Generating images from text descriptions.  
B. Summarizing a document or answering questions in a chat format.  
C. Training a new model from scratch using a custom architecture.  
D. Real-time video translation.  
✅ **Correct Answer:** B. Summarizing a document or answering questions in a chat format.

---

### **8. Why would you choose to fine-tune a foundation LLM via OCI GenAI?**  
A. To adjust the model’s weights so it performs better on domain-specific tasks or data (e.g., your industry’s terminology).  
B. To significantly reduce the model’s size by pruning parameters.  
C. To increase the context window of the model.  
D. Fine-tuning is not possible in OCI GenAI; only prompting is supported.  
✅ **Correct Answer:** A. To adjust the model’s weights so it performs better on domain-specific tasks or data (e.g., your industry’s terminology).

---

### **9. Oracle’s GenAI service implements an efficient fine-tuning method called T-Few. What is a key characteristic of T-Few fine-tuning?**  
A. It fully trains the entire model on your data.  
B. It inserts new adapter layers and updates only a small fraction of the model’s weights, reducing training time and cost.  
C. It uses reinforcement learning from human feedback instead of gradient descent.  
D. It requires at least 1 million training examples to be effective.  
✅ **Correct Answer:** B. It inserts new adapter layers and updates only a small fraction of the model’s weights, reducing training time and cost.

---

### **10. Before fine-tuning a model on OCI GenAI, what resource must you have or create?**  
A. A Kubernetes cluster in OCI for the training job.  
B. A dedicated AI cluster (GPU cluster) is allocated to run the fine-tuning job.  
C. A Docker container with the model weights.  
D. An Object Storage bucket to manually upload the model.  
✅ **Correct Answer:** D. An Object Storage bucket to manually upload the model.

---

Here is the **continued reformatted Q&A list** for **Domain 2: OCI Generative AI Service**, with each question followed by all options and the correct answer clearly indicated:

---

### **11. In OCI GenAI, what is a model endpoint?**  
A. A saved checkpoint of a training run.  
B. A network endpoint URL where a specific model (base or fine-tuned) is deployed for inference requests.  
C. The internal API the service uses to call the foundation model.  
D. The logging interface for model outputs.  
✅ **Correct Answer:** B. A network endpoint URL where a specific model (base or fine-tuned) is deployed for inference requests.

---

### **12. After you fine-tune a foundation model on OCI GenAI, where are the resulting custom model weights stored?**  
A. In Oracle’s central model repository (shared across tenants).  
B. In your OCI Object Storage, within your tenancy.  
C. They are merged back into the base model and are not accessible separately.  
D. On the GPU cluster indefinitely.  
✅ **Correct Answer:** B. In your OCI Object Storage, within your tenancy.

---

### **13. How does OCI GenAI ensure that one customer’s fine-tuning activities don’t interfere with another’s?**  
A. By assigning dedicated GPU clusters and isolated networking (RDMA) for each customer’s workload.  
B. By running jobs sequentially for each region.  
C. Through virtualization of GPUs with hypervisors.  
D. By only allowing one user of the service at a time.  
✅ **Correct Answer:** A. By assigning dedicated GPU clusters and isolated networking (RDMA) for each customer’s workload.

---

### **14. Which OCI service is used to control access to the Generative AI Service APIs and endpoints?**  
A. Oracle Cloud Guard.  
B. OCI Identity and Access Management (IAM).  
C. Oracle Data Safe.  
D. OCI Key Management.  
✅ **Correct Answer:** B. OCI Identity and Access Management (IAM).

---

### **15. What role does OCI Key Management (Vault) play in the GenAI service?**  
A. It stores API keys for accessing GenAI.  
B. It securely manages the encryption keys for the hosted foundation models and fine-tuning artifacts.  
C. It monitors the service for security threats.  
D. It manages SSH keys for GPU cluster access.  
✅ **Correct Answer:** B. It securely manages the encryption keys for the hosted foundation models and fine-tuning artifacts.

---

### **16. Your company’s compliance policy says “no customer data used for AI should leave the tenant’s boundary.” How does OCI GenAI support this?**  
A. All data sent to the GenAI models is first anonymized by Oracle.  
B. Fine-tuned models and any data embeddings are stored within your own tenancy’s infrastructure (e.g., your Object Storage and database).  
C. Oracle shares the fine-tuned model with others but not the raw data.  
D. The service cannot guarantee this – data always leaves the tenant boundary.  
✅ **Correct Answer:** B. Fine-tuned models and any data embeddings are stored within your own tenancy’s infrastructure (e.g., your Object Storage and database).

---

### **17. Which of the following is NOT a feature or characteristic of OCI’s Generative AI Service?**  
A. Choice of multiple pre-trained LLMs (Cohere, Llama 2) for generation and embedding tasks.  
B. Automatic scaling and serverless usage – you do not manage compute instances.  
C. Built-in support for deploying the models on-premises.  
D. The ability to fine-tune foundation models with your dataset.  
✅ **Correct Answer:** C. Built-in support for deploying the models on-premises.

---

### **18. What is the purpose of the GenAI Playground in the OCI Console?**  
A. It is a visual interface for testing prompts and models interactively, without writing code.  
B. It’s a training environment for model fine-tuning.  
C. It’s a monitoring dashboard for model endpoints.  
D. It is a game that teaches you how to use AI.  
✅ **Correct Answer:** A. It is a visual interface for testing prompts and models interactively, without writing code.

---

### **19. Which use case would embedding models + semantic search be better for than a standard keyword search?**  
A. Finding documents that are relevant in meaning to “investment banking trends,” even if they don’t contain those exact words.  
B. Finding documents that contain the exact phrase “investment banking trends.”  
C. Searching by document title only.  
D. Ensuring results are ordered by date.  
✅ **Correct Answer:** A. Finding documents that are relevant in meaning to “investment banking trends,” even if they don’t contain those exact words.

---

### **20. After deploying a custom model to an endpoint, how do you integrate it into an application?**  
A. By calling the endpoint’s REST API with appropriate auth, passing prompts, and getting model inferences in response.  
B. By connecting the OCI Streaming service to the endpoint.  
C. By using an SDK only, the model cannot be called via REST.  
D. By importing the model file into your application manually.  
✅ **Correct Answer:** A. By calling the endpoint’s REST API with appropriate auth, passing prompts, and getting model inferences in response.

---

Here is the **reformatted Q&A set** for **Retrieval-Augmented Generation (RAG)** in the context of LLM applications, with each question followed by all options and the correct answer clearly indicated:

---

### **1. What is Retrieval-Augmented Generation (RAG) in the context of LLM applications?**  
A. A method to train language models faster.  
B. Using a search or database retrieval step to fetch relevant context, and providing that context to an LLM to ground its answer.  
C. Running an LLM on a very large input (retrieving all possible data at once).  
D. A type of prompt format for arithmetic reasoning.  
✅ **Correct Answer:** B. Using a search or database retrieval step to fetch relevant context, and providing that context to an LLM to ground its answer.

---

### **2. Why is RAG useful for Q&A chatbots?**  
A. It bypasses the need for an LLM entirely.  
B. It allows the LLM to provide up-to-date, specific information from a document set, reducing hallucination and extending knowledge beyond the model’s training.  
C. It significantly increases the LLM’s parameter count.  
D. It ensures the LLM will never make errors.  
✅ **Correct Answer:** B. It allows the LLM to provide up-to-date, specific information from a document set, reducing hallucination and extending knowledge beyond the model’s training.

---

### **3. What are the main stages of a basic RAG pipeline?**  
A. Ingestion (load and chunk documents into a vector index), Retrieval (find relevant chunks by similarity to the query), and Generation (the LLM produces an answer using the retrieved context).  
B. Training, Validation, and Deployment.  
C. Tokenization, Embedding, and Decoding.  
D. Authentication, Transformation, Response.  
✅ **Correct Answer:** A. Ingestion (load and chunk documents into a vector index), Retrieval (find relevant chunks by similarity to the query), and Generation (the LLM produces an answer using the retrieved context).

---

### **4. Why are documents split into chunks during the RAG ingestion phase?**  
A. LLMs can only read short texts due to token limits, so splitting ensures each chunk can fit into the model’s context.  
B. To increase the total number of vectors for more storage usage.  
C. To make the embeddings more random.  
D. To ensure each word becomes its chunk.  
✅ **Correct Answer:** A. LLMs can only read short texts due to token limits, so splitting ensures each chunk can fit into the model’s context.

---

### **5. What is an embedding in the context of NLP and semantic search?**  
A. A fixed-length numeric vector that represents the semantic meaning of text (words, sentences, or documents).  
B. A hyperlink inside a document.  
C. A type of database index for keywords.  
D. A summary of a document.  
✅ **Correct Answer:** A. A fixed-length numeric vector that represents the semantic meaning of text (words, sentences, or documents).

---

### **6. How can you generate text embeddings using OCI’s services?**  
A. By using the OCI Generative AI Embedding model endpoint to get vector representations of text.  
B. By running a Hadoop cluster on OCI.  
C. Only by using third-party libraries outside OCI.  
D. The OCI GenAI service does not support embeddings.  
✅ **Correct Answer:** A. By using the OCI Generative AI Embedding model endpoint to get vector representations of text.

---

### **7. Oracle Database 23c introduced vector support. Which statement is true about Oracle’s vector store capability?**  
A. It requires data to be stored as BLOBs; there is no special vector type.  
B. It provides a new VECTOR data type for columns to hold embedding vectors, and SQL functions to perform similarity search (e.g., cosine distance).  
C. It only works with image data, not text embeddings.  
D. Oracle 23c automatically trains LLMs inside the database.  
✅ **Correct Answer:** B. It provides a new VECTOR data type for columns to hold embedding vectors, and SQL functions to perform similarity search (e.g., cosine distance).

---

### **8. Which similarity metric compares two text embedding vectors by measuring the angle between them while ignoring magnitude?**  
A. Euclidean distance.  
B. Dot product.  
C. Cosine similarity.  
D. Manhattan distance.  
✅ **Correct Answer:** C. Cosine similarity.

---

### **9. What is the purpose of using an index (like HNSW or IVF) in a vector database?**  
A. To compress the vectors into smaller dimensions.  
B. To accelerate similarity search by organizing vectors for faster nearest-neighbor lookup.  
C. To convert vectors back into text.  
D. To ensure exact keyword matching on vector data.  
✅ **Correct Answer:** B. To accelerate similarity search by organizing vectors for faster nearest-neighbor lookup.

Great question! Let’s break this down into two parts:

---

## 🔍 What is Nearest-Neighbor Search?

**Nearest-neighbor search (NNS)** is a technique used to find the **most similar items** to a given query in a dataset. In the context of **vector databases** and **machine learning**, it means:

- You have a set of **vectors** (numerical representations of data like text, images, etc.).
- You want to find the **closest vectors** to a query vector based on a **distance metric** (e.g., cosine similarity, Euclidean distance).
- This is essential for tasks like:
  - Semantic search
  - Recommendation systems
  - Retrieval-Augmented Generation (RAG)

---

## ⚙️ What is HNSW?

**HNSW (Hierarchical Navigable Small World)** is a **graph-based indexing algorithm** for fast approximate nearest-neighbor search.

### Key Features:
- Builds a **multi-layer graph** where each node is a vector.
- Uses **shortcuts** to navigate quickly between nodes.
- Offers **high recall** and **low latency**, making it ideal for real-time applications.

### Benefits:
- Fast search even in large datasets.
- Good balance between accuracy and performance.
- Commonly used in tools like **FAISS**, **Weaviate**, and **Pinecone**.

---

## ⚙️ What is IVF?

**IVF (Inverted File Index)** is a **clustering-based indexing method** used in vector search.

### How It Works:
- Vectors are grouped into **clusters** (using k-means or similar).
- During search, only the **most relevant clusters** are scanned.
- This reduces the number of comparisons and speeds up the search.

### Benefits:
- Efficient for large-scale datasets.
- Often used in **batch processing** or **offline search** scenarios.
- Available in libraries like **FAISS**.

---

## 🧠 Summary Table

| Technique | Type         | Best For                     | Speed | Accuracy |
|-----------|--------------|------------------------------|-------|----------|
| **NNS**   | Search method| Finding similar vectors      | Varies| High     |
| **HNSW**  | Graph-based  | Real-time, high-recall search| Fast  | High     |
| **IVF**   | Cluster-based| Large-scale, efficient search| Faster| Moderate |

---

---

### **10. In LangChain, how do you combine the LLM and retrieval steps to implement RAG Q&A?**  
A. Use a specialized chain (e.g., RetrievalQA) that automatically queries a vector store retriever for relevant documents and passes them to the LLM for answer generation.  
B. Manually call the database, then manually call the LLM in code.  
C. Fine-tune the LLM on the documents instead of retrieving.  
D. LangChain cannot handle that; you must write your pipeline.  
✅ **Correct Answer:** A. Use a specialized chain (e.g., RetrievalQA) that automatically queries a vector store retriever for relevant documents and passes them to the LLM for answer generation.

---

### **11. In a multi-turn chatbot built with LangChain, what component is used to ensure the AI remembers previous conversation turns?**  
A. The Memory module, which stores prior messages or a summary, so the LLM can incorporate past context into new answers.  
B. The VectorStore, which keeps all past dialogues as embeddings.  
C. A special prompt that repeats everything said so far (no built-in component).  
D. LangChain does this automatically without any configuration.  
✅ **Correct Answer:** A. The Memory module, which stores prior messages or a summary, so the LLM can incorporate past context into new answers.

---

### **12. Oracle’s implementation of RAG often uses Oracle DB as the vector store. What must be ensured when using an Oracle Database as a knowledge base for RAG?**  
A. The database version is 19c or lower.  
B. The same embedding model used to generate the chunk vectors is used to encode incoming queries, to ensure vectors are comparable.  
C. All text data must be in one large row.  
D. Only one query can be processed at a time.  
✅ **Correct Answer:** B. The same embedding model used to generate the chunk vectors is used to encode incoming queries, to ensure vectors are comparable.

---

### **13. What is a benefit of semantic search over traditional keyword (lexical) search in the RAG context?**  
A. It finds results that are related in meaning, even if exact keywords differ or are not present.  
B. It is 100% precise with no irrelevant results.  
C. It ignores the actual content and matches only metadata.  
D. It’s always faster than keyword search.  
✅ **Correct Answer:** A. It finds results that are related in meaning, even if exact keywords differ or are not present.

---

### **14. If you set `return_source_documents=True` in a LangChain RetrievalQA chain, what happens?**  
A. The LLM’s answer will include citations or the actual source text for transparency.  
B. The chain will output the retrieved documents (or their references) along with the answer, allowing you to see which sources were used.  
C. The LLM will quote entire documents in its answer.  
D. The chain will return only the documents and not answer the question.  
✅ **Correct Answer:** B. The chain will output the retrieved documents (or their references) along with the answer, allowing you to see which sources were used.

---

Here is the **reformatted Q&A set** for **Oracle Cloud Generative AI Agents**, with each question followed by all options and the correct answer clearly indicated:

---

### **1. What is the Oracle Cloud Generative AI Agents service?**  
A. A fully managed service that lets you create LLM-powered agents (chatbots) that use large language models plus an intelligent retrieval system to answer queries using your enterprise data.  
B. A tool for training new foundation models.  
C. A hardware device for running AI models.  
D. A SaaS application for human call center agents.  
✅ **Correct Answer:** A. A fully managed service that lets you create LLM-powered agents (chatbots) that use large language models plus an intelligent retrieval system to answer queries using your enterprise data.

---

### **2. In the context of OCI GenAI Agents, what is a Knowledge Base?**  
A. A collection of rules that the agent follows.  
B. The vector-indexed datastore of your ingested content, which the agent can query for relevant information.  
C. The pre-trained knowledge of the base LLM.  
D. A log of all conversations the agent has had.  
✅ **Correct Answer:** B. The vector-indexed datastore of your ingested content, which the agent can query for relevant information.

---

### **3. Which data source types are supported for populating an Oracle GenAI Agent’s knowledge base? (Choose 2)**  
A. Object Storage bucket – the service can ingest PDF or text files from a bucket.  
B. OCI OpenSearch index – use an OCI Search with an OpenSearch index that’s already loaded with data.  
C. Oracle Database 23c vector store – bring your table of vectors and a similarity search function.  
D. On-premises Hadoop file system.  
✅ **Correct Answers:** A. Object Storage bucket – the service can ingest PDF or text files from a bucket.  
✅ **B. OCI OpenSearch index – use an OCI Search with an OpenSearch index that’s already loaded with data.**

---

### **4. When using an Object Storage bucket as a knowledge base, which is NOT a requirement or limitation?**  
A. Files must be in PDF or plain text format (up to 100 MB each).  
B. Only one bucket can be used per data source.  
C. Images within PDFs are ignored entirely (cannot be processed).  
D. Charts in PDFs should be 2D with labeled axes for the AI to interpret them.  
✅ **Correct Answer:** D. Charts in PDFs should be 2D with labeled axes for the AI to interpret them.

---

### **5. You want to use an Oracle Autonomous Database as a knowledge base for an agent. What must you do?**  
A. Manually convert your data to embeddings using an external tool first.  
B. Create a table with text and vector columns (for chunks and their embeddings) and implement a PL/SQL vector search function that the agent will call for retrieval.  
C. Export the entire database to JSON files and put them in Object Storage.  
D. It’s not possible to connect the GenAI Agent to a database.  
✅ **Correct Answer:** B. Create a table with text and vector columns (for chunks and their embeddings) and implement a PL/SQL vector search function that the agent will call for retrieval.

---

### **6. What is the purpose of an Agent Endpoint in OCI Generative AI Agents?**  
A. It’s the interface where you configure the agent’s personality.  
B. It is a deployment point that you create, which gives a stable REST endpoint or chat interface for interacting with your agent.  
C. It’s an OCI Monitoring alarm for the agent.  
D. It’s the vector database connection.  
✅ **Correct Answer:** B. It is a deployment point that you create, which gives a stable REST endpoint or chat interface for interacting with your agent.

---

### **7. Which of the following operations can OCI’s Generative AI Agent perform that a basic LLM chatbot alone cannot?**  
A. Maintain conversational context across turns.  
B. Call external tools or APIs (e.g., database queries, booking systems) as decided by the LLM’s reasoning.  
C. Generate text in English.  
D. Translate from English to French.  
✅ **Correct Answer:** B. Call external tools or APIs (e.g., database queries, booking systems) as decided by the LLM’s reasoning.

---

### **8. What is the function of the Trace feature in the GenAI Agents service?**  
A. It logs and displays the sequence of steps the agent took for each user query – including the prompts, retrieved data, and LLM’s intermediate reasoning.  
B. It traces network packets for debugging connectivity.  
C. It summarizes the conversation after each turn.  
D. It re-trains the agent based on user feedback.  
✅ **Correct Answer:** A. It logs and displays the sequence of steps the agent took for each user query – including the prompts, retrieved data, and LLM’s intermediate reasoning.

---

### **9. How does the agent provide citations in responses, and why is this useful?**  
A. It outputs the embedding vector to prove that it used the knowledge base.  
B. It appends source titles/URLs and page references for facts in its answer, so users can verify information back to the original documents.  
C. It cites the LLM model (e.g., “Answer generated by Llama-2”). Citations are used to credit the developers of the agent.  
✅ **Correct Answer:** B. It appends source titles/URLs and page references for facts in its answer, so users can verify information back to the original documents.

---

### **10. What is content moderation in OCI GenAI Agents?**  
A. A feature that filters out or masks hateful, harmful, or policy-violating content in user prompts or the agent’s responses.  
B. A way to limit how much content the agent can output (rate limiting).  
C. A process of curating which files go into the knowledge base.  
D. An Oracle support service for monitoring your agent.  
✅ **Correct Answer:** A. A feature that filters out or masks hateful, harmful, or policy-violating content in user prompts or the agent’s responses.

---

### **11. What does enabling hybrid search for a knowledge base do?**  
A. It stores half the data as vectors and half as text.  
B. It combines semantic vector search with keyword (lexical) search to improve result relevance.  
C. It uses two different LLMs for answering.  
D. It allows the agent to search the internet.  
✅ **Correct Answer:** B. It combines semantic vector search with keyword (lexical) search to improve result relevance.

---

Here is the **reformatted Q&A set** for **OCI Generative AI Agents – Advanced Topics**, with each question followed by all options and the correct answer clearly indicated:

---

### **1. Knowledge Base Integration**  
Your team configures a GenAI Agent with an Object Storage knowledge base. The PDF files contain multilingual text, tables, and scanned diagrams. Which data elements will the Agent reliably ingest and use for retrieval?  
A. All multilingual text content, including extracted text from embedded diagrams.  
B. Text-based content only; embedded diagrams and images will be ignored.  
C. Multilingual text only if explicitly labeled with UTF-8 encoding.  
D. Only English text content and labeled tables.  
✅ **Correct Answer:** B. Text-based content only; embedded diagrams and images will be ignored.

---

### **2. Model Selection**  
When choosing between OCI’s Cohere-based LLM and an OpenSearch-powered embedding model for an Agent, which task requires embedding models rather than generative LLMs?  
A. Summarizing customer complaints.  
B. Classifying text into sentiment categories.  
C. Performing similarity search over large knowledge bases.  
D. Answering open-ended questions from users.  
✅ **Correct Answer:** C. Performing similarity search over large knowledge bases.

---

### **3. Vector Database Integration**  
An enterprise wants to use its own Oracle Database 23ai with vector support for semantic search in a GenAI Agent. Which statement is correct?  
A. GenAI Agents can natively connect to Oracle Database vector stores as a knowledge base.  
B. GenAI Agents cannot directly use Database 23ai; data must first be staged into Object Storage or OpenSearch.  
C. GenAI Agents can use 23ai only for structured SQL queries, not vector similarity.  
D. GenAI Agents auto-convert 23ai vector tables into embeddings without configuration.  
✅ **Correct Answer:** A. GenAI Agents can natively connect to Oracle Database vector stores as a knowledge base.

---

### **4. Fine-tuning Limits**  
Your team wants to fine-tune a foundation model with customer call transcripts. Which of the following is a limitation of OCI fine-tuning?  
A. You can only fine-tune embedding models, not generative models.  
B. Fine-tuned models cannot be deployed into private subnets.  
C. Fine-tuned models inherit usage quotas and limits of the base model family.  
D. Fine-tuned models automatically overwrite the base model.  
✅ **Correct Answer:** C. Fine-tuned models inherit usage quotas and limits of the base model family.

---

### **5. Governance**  
Which framework is explicitly referenced in OCI’s GenAI governance best practices for risk management?  
A. COBIT 2019  
B. NIST AI Risk Management Framework (AI RMF)  
C. ITIL v4  
D. ISO 22301  
✅ **Correct Answer:** B. NIST AI Risk Management Framework (AI RMF)

---

### **6. Agent vs Chatbot**  
Which operation can an OCI Generative AI Agent perform that a standalone LLM chatbot cannot?  
A. Maintain conversational context.  
B. Generate multilingual text.  
C. Call external APIs/tools during reasoning.  
D. Perform text summarization.  
✅ **Correct Answer:** C. Call external APIs/tools during reasoning.

---

### **7. Latency Optimization**  
You deploy a GenAI Agent that queries a 1 TB knowledge base in Object Storage. Latency is high. Which OCI-native optimization is recommended?  
A. Convert all files to image-based PDFs for faster parsing.  
B. Pre-chunk documents into smaller text blocks before ingestion.  
C. Store files in multiple buckets and point the Agent to all of them.  
D. Increase the beam size of the generative model.  
✅ **Correct Answer:** B. Pre-chunk documents into smaller text blocks before ingestion.

---

### **8. Model Deployment**  
You need to deploy a fine-tuned foundation model into a production environment that handles confidential healthcare data. The compliance team requires network isolation and no public internet exposure. Which OCI deployment option best satisfies this requirement?  
A. Deploy the model in a public endpoint with VCN security lists blocking all inbound traffic.  
B. Deploy the model as a private endpoint within a VCN subnet, accessible only via private IPs.  
C. Use the base foundation model directly, since only fine-tuned models require isolation.  
D. Host the model on Object Storage with signed URLs for restricted access.  
✅ **Correct Answer:** B. Deploy the model as a private endpoint within a VCN subnet, accessible only via private IPs.

---

### **9. Model Privacy**  
Which statement is true about data sent to OCI Generative AI APIs?  
A. Customer data may be retained for model training unless disabled in settings.  
B. Customer data is never used to train Oracle’s base foundation models.  
C. Data is stored for training by default for 30 days.  
D. Data retention is required unless the tenant uses a private subnet.  
✅ **Correct Answer:** B. Customer data is never used to train Oracle’s base foundation models.

---

### **10. Retrieval-Augmented Generation (RAG)**  
In OCI GenAI, what is the role of the embedding model in a RAG pipeline?  
A. Generate the final answer in natural language.  
B. Map user queries and documents into a shared vector space for similarity search.  
C. Tokenize text into subwords for faster LLM inference.  
D. Ensure data privacy by masking sensitive fields.  
✅ **Correct Answer:** B. Map user queries and documents into a shared vector space for similarity search.

---

### **11. Multi-Agent Coordination**  
You design a system where one GenAI Agent handles customer inquiries, and another Agent handles ticket creation via API. What mechanism ensures the Agents can coordinate securely?  
A. Use OCI Service Connectors with IAM policies.  
B. Use embedded prompts with role-based delegation.  
C. Use Agent-to-Agent handoff with OCI Functions as middleware.  
D. Configure both Agents in a single session with expanded context windows.  
✅ **Correct Answer:** C. Use Agent-to-Agent handoff with OCI Functions as middleware.

---

### **12. Cost Optimization**  
Your CIO complains about high costs from using the largest foundation models for all GenAI queries. Which OCI feature helps cut costs while balancing accuracy?  
A. Autoscaling Object Storage buckets.  
B. Dynamic model selection with smaller models for lightweight tasks.  
C. Increasing beam size to reduce retries.  
D. Using Free Tier foundation models.  
✅ **Correct Answer:** B. Dynamic model selection with smaller models for lightweight tasks.

---

### **13. File Size Limits**  
What is the maximum file size for ingestion into an Object Storage bucket as a GenAI knowledge base?  
A. 10 MB  
B. 50 MB  
C. 100 MB  
D. Unlimited, if partitioned  
✅ **Correct Answer:** C. 100 MB

---

### **14. Tool Invocation**  
Which best describes how an Agent decides to call an external tool?  
A. Tools are invoked only at fixed checkpoints configured by the developer.  
B. The LLM’s reasoning determines if/when to call the tool based on user input and context.  
C. Tools must be triggered manually by a system administrator.  
D. Agents cannot call external tools directly.  
✅ **Correct Answer:** B. The LLM’s reasoning determines if/when to call the tool based on user input and context.

---

### **15. Multilingual Limits**  
Your Agent must handle queries in English, French, and Japanese. Which is a limitation?  
A. Embedding models support English only.  
B. Generative models support English and French, but not Japanese.  
C. Embedding models support multilingual text, but accuracy may vary across languages.  
D. Multilingual support requires fine-tuning.  
✅ **Correct Answer:** C. Embedding models support multilingual text, but accuracy may vary across languages.

---

### **16. SLA Awareness**  
Which SLA condition applies to OCI Generative AI service availability?  
A. GenAI inherits the same SLA as Object Storage.  
B. Oracle publishes distinct SLA metrics for GenAI APIs separate from core OCI services.  
C. SLAs apply only when using private foundation models.  
D. SLAs apply only to Agents, not base models.  
✅ **Correct Answer:** B. Oracle publishes distinct SLA metrics for GenAI APIs separate from core OCI services.

---

### **17. Knowledge Base Sync**  
If you update a PDF in the Object Storage bucket linked to a GenAI knowledge base, what must you do to ensure the Agent uses the new version?  
A. Nothing; Agents sync in real time with Object Storage.  
B. Re-trigger an ingestion job to refresh embeddings.  
C. Delete and recreate the entire knowledge base.  
D. Update the IAM policy attached to the Agent.  
✅ **Correct Answer:** B. Re-trigger an ingestion job to refresh embeddings.

---

### **18. Model Hallucination**  
Which mitigation best reduces hallucination in OCI GenAI Agents?  
A. Increase the temperature parameter.  
B. Restrict generation to retrieval-augmented context only.  
C. Disable embeddings.  
D. Switch from generative to embedding-only responses.  
✅ **Correct Answer:** B. Restrict generation to retrieval-augmented context only.

---

### **19. IAM Roles**  
To allow a GenAI Agent to access data in an OCI Object Storage bucket, which IAM principle is required?  
A. Grant OBJECT_WRITE and OBJECT_DELETE privileges.  
B. Grant OBJECT_READ access to the Agent’s dynamic group.  
C. Grant tenancy-level ADMIN rights to the Agent.  
D. No IAM permissions are needed; Agents bypass IAM.  
✅ **Correct Answer:** B. Grant OBJECT_READ access to the Agent’s dynamic group.

---

### **20. Model Drift**  
Your fine-tuned sentiment analysis model starts misclassifying slang terms. Which is the most appropriate mitigation in OCI?  
A. Increase beam size.  
B. Collect new training data, including slang, and refine.  
C. Adjust Object Storage bucket policies.  
D. Switch to an embedding model.  
✅ **Correct Answer:** B. Collect new training data, including slang, and refine.

---

Absolutely! Here's the full list of questions with all options, correct answers clearly marked, and **explanations** for each:

---

### **01. How is the `totalTrainingSteps` parameter calculated during fine-tuning in OCI Generative AI?**

**Options:**
a) `totalTrainingSteps = (totalTrainingEpochs * size(trainingDataset)) / trainingBatchSize` ✅  
b) `totalTrainingSteps = (totalTrainingEpochs * trainingBatchSize) / size(trainingDataset)`  
c) `totalTrainingSteps = (size(trainingDataset) * trainingBatchSize) / totalTrainingEpochs`  
d) `totalTrainingSteps = (totalTrainingEpochs + size(trainingDataset)) * trainingBatchSize`

**Explanation:**  
This formula ensures that the model sees the entire dataset for each epoch, divided into batches. It calculates how many steps are needed to complete all epochs.

---

### **02. In which phase of the RAG pipeline are additional context and user query used by LLMs to respond to the user?**

**Options:**
a) Evaluation  
b) Ingestion  
c) Retrieval  
d) Generation ✅

**Explanation:**  
The **Generation** phase is where the LLM uses the retrieved context and the user’s query to generate a response. Retrieval happens before this, and ingestion is the data preparation phase.

---

### **03. What is the primary reason why diffusion models are difficult to apply to text generation tasks?**

**Options:**
a) Because text is not categorical  
b) Because text representation is categorical, unlike images ✅  
c) Because diffusion models can only produce images  
d) Because text generation does not require complex models

**Explanation:**  
Diffusion models work well with continuous data like images. Text is **categorical**, making it harder to model with diffusion techniques due to discrete tokenization.

---

### **04. What happens when this line of code is executed?**  
`embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)`

**Options:**
a) It processes and configures the OCI profile settings for the inference session.  
b) It initializes a pretrained OCI Generative AI model for use in the session.  
c) It sends a request to the OCI Generative AI service to generate an embedding for the input text. ✅  
d) It initiates a connection to OCI and authenticates using the user’s credentials.

**Explanation:**  
This line sends a request to OCI GenAI to **embed the input text**, converting it into a vector representation for semantic search or other downstream tasks.

---

### **05. In an OCI Generative AI chat model, which of these parameter settings is most likely to induce hallucinations and factually incorrect information?**

**Options:**
a) `temperature = 0.9`, `top_p = 0.8`, and `frequency_penalty = 0.1` ✅  
b) `temperature = 0.2`, `top_p = 0.6`, and `frequency_penalty = 0.8`  
c) `temperature = 0.0`, `top_p = 0.7`, and `frequency_penalty = 1.0`  
d) `temperature = 0.5`, `top_p = 0.9`, and `frequency_penalty = 0.5`

**Explanation:**  
High **temperature** and **top_p** values increase randomness, which can lead to **hallucinations**—responses that are fluent but factually incorrect.

---

### **06. When activating content moderation in OCI Generative AI Agents, which of these can you specify?**

**Options:**
a) The maximum file size for input data  
b) The threshold for language complexity in responses  
c) The type of vector search used for retrieval  
d) Whether moderation applies to user prompts, generated responses, or both ✅

**Explanation:**  
OCI GenAI allows you to configure **content moderation** to apply to **user inputs**, **model outputs**, or **both**, helping enforce safety and compliance.

---

### **07. How can you affect the probability distribution over the vocabulary of a Large Language Model (LLM)?**

**Options:**
a) By adjusting the token size during the training phase  
b) By restricting the vocabulary used in the model  
c) By using techniques like prompting and training ✅  
d) By modifying the model’s training data

**Explanation:**  
**Prompting and training** influence how the model selects words by shaping the probability distribution over its vocabulary during inference.

---

### **08. You want to build an LLM application that can connect application components easily and allow for component replacement in a declarative manner. What approach would you take?**

**Options:**
a) Use LangChain Expression Language (LCEL). ✅  
b) Use prompts.  
c) Use agents.  
d) Use Python classes like LLMChain.

**Explanation:**  
**LCEL** allows you to declaratively define chains and swap components easily, making it ideal for modular and maintainable LLM applications.

---

### **09. What must be done before you can delete a knowledge base in Generative AI Agents?**

**Options:**
a) Disconnect the database tool connection.  
b) Delete the data sources and agents using that knowledge base. ✅  
c) Reassign the knowledge base to a different agent.  
d) Archive the knowledge base for future use.

**Explanation:**  
You must **delete all agents and data sources** that depend on the knowledge base before it can be removed, ensuring no active dependencies exist.

---

### **10. Which phase of the RAG pipeline includes loading, splitting, and embedding of documents?**

**Options:**
a) Evaluation  
b) Generation  
c) Retrieval  
d) Ingestion ✅

**Explanation:**  
The **Ingestion** phase prepares the data by **loading**, **chunking**, and **embedding** documents into a vector store for later retrieval.

---


When does a chain typically interact with memory in a run within the LangChain framework?
A. After user input but before chain execution, and again after core logic but before output
B. Only after the output has been generated
C. Before user input and after chain execution
D. Continuously throughout the entire chain execution process

✅ **Correct Answer: A. After user input but before chain execution, and again after core logic but before output**

---

### 🔍 Explanation:

In the **LangChain framework**, **memory** is used to **track and store information across interactions**—like previous user inputs, model outputs, or conversation history. A chain typically interacts with memory at two key points during a run:

1. **After receiving user input but before executing the chain logic**  
   - This allows the chain to **retrieve relevant context** (e.g., previous messages or variables) and incorporate it into the current run.

2. **After the core logic is executed but before returning the output**  
   - This is when the chain **updates the memory** with new information (e.g., the latest user query and model response), ensuring continuity in future interactions.

This pattern enables **stateful conversations**, which are essential for chatbots, agents, and multi-turn applications.

---

### ❌ Why the other options are incorrect:

- **B. Only after the output has been generated** – Memory is also accessed before execution.
- **C. Before user input and after chain execution** – Memory isn't typically accessed before user input.
- **D. Continuously throughout the entire chain execution process** – Memory access is **event-driven**, not continuous.

Would you like a visual diagram or code example showing how memory is integrated in a LangChain chain?



How does the utilization of T-Few transformer layers contribute to the efficiency of the fine-tuning process?
A. By allowing updates across all layers of the model
B. By excluding transformer layers from the fine-tuning process entirely
C. By incorporating additional layers to the base model
D. By restricting updates to only a specific group of transformer layers


✅ **Correct Answer: D. By restricting updates to only a specific group of transformer layers**

---

### 🔍 Explanation:

**T-Few (Task-aware Few-shot fine-tuning)** is designed to make fine-tuning **efficient and scalable** by:

- **Restricting updates** to only a **subset of transformer layers** rather than updating the entire model.
- This selective tuning reduces:
  - **Computational cost**
  - **Memory usage**
  - **Risk of overfitting**, especially in few-shot scenarios

By focusing on **task-relevant layers**, T-Few achieves strong performance with minimal data and resources.

---

### ❌ Why the other options are incorrect:

- **A. Updates across all layers** – This is typical of full fine-tuning, not T-Few.
- **B. Excludes transformer layers entirely** – Transformer layers are still involved; only some are updated.
- **C. Adds new layers** – T-Few doesn’t add layers; it fine-tunes selectively within the existing architecture.




### 🔍 What Is Learning Rate?

The **learning rate** is a key hyperparameter in machine learning and deep learning that controls **how much the model's weights are updated** during training in response to the error it sees.

---

### ✅ In Simple Terms:
- It determines **how fast or slow** a model learns.
- A **high learning rate** means **larger updates** to the model weights.
- A **low learning rate** means **smaller, more gradual updates**.

---

### 📈 Why It Matters:
- If the learning rate is **too high**, the model may **overshoot** the optimal solution and fail to converge.
- If it's **too low**, training becomes **very slow** and may get stuck in a suboptimal state.

---

### 🧠 Example:
Imagine you're trying to find the lowest point in a valley (the optimal model). The learning rate controls how big your steps are:
- **Big steps** (high learning rate) might make you jump over the lowest point.
- **Small steps** (low learning rate) help you move carefully but slowly toward the bottom.

---

### 🔧 Common Learning Rate Strategies:
- **Constant**: Fixed value throughout training.
- **Decay**: Gradually reduces over time.
- **Warmup**: Starts small and increases gradually (see `learning_rate_warmup_steps`).
- **Adaptive**: Adjusts based on performance (used in optimizers like Adam).

### 🔍 What Are `learning_rate_warmup_steps`?

**`learning_rate_warmup_steps`** is a hyperparameter used during the **training or fine-tuning** of machine learning models, especially **transformers** like those in **OCI Generative AI** or Hugging Face models.

---

### ✅ **Purpose of Warmup Steps**

During the early stages of training, models can be **unstable** if the learning rate is too high. Warmup steps help stabilize training by:

- **Gradually increasing the learning rate** from a small value to the target learning rate.
- Preventing large weight updates early on, which could destabilize the model.
- Allowing the optimizer to "ease into" training before applying full-strength updates.

---

### 📈 How It Works

If you set `learning_rate_warmup_steps = 500`, for example:

- For the **first 500 steps**, the learning rate increases linearly from 0 to the specified maximum.
- After that, the learning rate follows the main schedule (e.g., constant, cosine decay, etc.).

---

### 🧠 Why It’s Important

- Helps **prevent divergence** early in training.
- Improves **generalization** and **convergence stability**.
- Especially useful in **fine-tuning large models** where initial gradients can be volatile.

---
### 🔍 What Are `total_training_tokens`?

**`total_training_tokens`** refers to the **total number of tokens** processed during the **fine-tuning or training** of a language model.

---

### ✅ What Is a Token?

A **token** is a unit of text used by language models. It can be:
- A word
- A subword
- A character
- Or even punctuation

For example, the sentence:  
**"Generative AI is powerful."**  
might be split into tokens like:  
`["Generative", "AI", "is", "powerful", "."]` — or even smaller units depending on the tokenizer.

---

### 📊 What Does `total_training_tokens` Measure?

It measures the **cumulative number of tokens** fed into the model across:
- All training examples
- All epochs
- All batches

### 🧠 Why It Matters:
- It helps estimate **training cost** and **duration**.
- It’s used to **monitor progress** and **control overfitting**.
- It’s often a key metric in **billing** for cloud-based model training (like in OCI GenAI).

---

### 🧮 Example Calculation:

If you have:
- 10,000 training examples
- Each example averages 100 tokens
- Trained over 3 epochs

Then:  
**`total_training_tokens = 10,000 × 100 × 3 = 3,000,000 tokens`**

---

### 🔍 What Is `early_stopping_patience`?

**`early_stopping_patience`** is a hyperparameter used in training and fine-tuning machine learning models to **control when early stopping should occur**.

---

### ✅ What Is Early Stopping?

**Early stopping** is a technique that **halts training** when the model’s performance on a **validation set stops improving**, helping to:
- Prevent **overfitting**
- Save **computational resources**
- Improve **generalization**

---

### 🧠 What Does `early_stopping_patience` Do?

It defines **how many consecutive training steps or epochs** the model is allowed to continue **without improvement** before stopping.

#### Example:
If `early_stopping_patience = 3`, and the model’s validation loss doesn’t improve for **3 consecutive epochs**, training will stop early.

---

### 📊 Why It’s Useful:
- Helps avoid wasting time on training that no longer improves the model.
- Balances between giving the model a chance to improve and stopping before overfitting.

---

The question from the image was:

**"Which OCI networking component is essential to allow a Compute instance in a private subnet to call the Generative AI API without public internet exposure?"**

And the answer choices were:

- A. Internet Gateway  
- B. Service Gateway  
- C. NAT Gateway  

The correct answer is:

**B. Service Gateway**

### Explanation:
In Oracle Cloud Infrastructure (OCI), when you want a **Compute instance in a private subnet** to access **OCI services like the Generative AI API** **without exposing it to the public internet**, you use a **Service Gateway**.

Here’s a quick breakdown of the options:

- **Internet Gateway**: Allows resources in a **public subnet** to access the internet. Not suitable for private subnets.
- **NAT Gateway**: Allows **outbound internet traffic** from private subnets, but still goes through the public internet.
- **Service Gateway**: Enables private subnet resources to access **OCI public services** (like Object Storage, Generative AI API, etc.) **without using the internet**.

**Cold start latency** refers to the delay that occurs when a system or service is started for the first time or after a period of inactivity. It’s especially relevant in **cloud computing** and **serverless architectures**.

### In more detail:

- **In serverless functions (e.g., AWS Lambda, OCI Functions):**
  - When a function is invoked after being idle, the platform needs to **initialize the runtime environment**, **load the code**, and **establish network connections**.
  - This initialization time is called **cold start latency**.
  - Subsequent invocations (called **warm starts**) are much faster because the environment is already running.

- **In Generative AI APIs or ML models:**
  - Cold start latency can occur when a model is **loaded into memory** from disk or a remote location before it can serve requests.
  - This is common in **on-demand model serving** setups where models are not kept in memory to save resources.

### Why it matters:
- It affects **user experience**, especially for real-time applications.
- It can impact **performance SLAs**.
- Mitigation strategies include **pre-warming**, **provisioned concurrency**, or **keeping instances warm**.


Here is the question, answer, and explanation from the image:

---

### **Question:**
A fine-tuning job fails with an error related to **"catastrophic forgetting."** What does this imply about the training process?

### **Answer:**
**B. The model learned the new data but lost its original capabilities.**

---

### **Explanation:**
**Catastrophic forgetting** is a phenomenon in machine learning where a model, after being trained on new data, **forgets previously learned information**. This typically happens during **fine-tuning**, especially when the new data is significantly different from the original training data or when the training process doesn't preserve the original knowledge.

In this case, the model:
- **Successfully learned** the new data.
- But in doing so, **overwrote or lost** its prior knowledge.
- This leads to degraded performance on tasks it was previously good at.

To mitigate this, techniques like **regularization**, **rehearsal methods**, or **continual learning strategies** are often used.

Here is the full question with all the options, followed by the correct answer and explanation:

---

### **Question:**
Which OCI service is designed to orchestrate a multi-step ML workflow, including data prep, model fine-tuning, and evaluation?

**Options:**
- **A. OCI Functions**  
- **B. OCI Data Flow**  
- **C. OCI Data Science Pipelines**

---

### **Correct Answer:**
**C. OCI Data Science Pipelines**

---

### **Explanation:**
**OCI Data Science Pipelines** is specifically built to manage and automate **multi-step machine learning workflows**. It allows data scientists and ML engineers to define, schedule, and monitor steps such as:

- Data preprocessing  
- Model training and fine-tuning  
- Model evaluation and deployment  

This orchestration ensures reproducibility, scalability, and efficiency in ML operations.

#### Why not the others?
- **OCI Functions**: Best for running short-lived, event-driven serverless functions—not ideal for orchestrating complex ML workflows.
- **OCI Data Flow**: Designed for large-scale data processing using Apache Spark—not specifically tailored for ML pipeline orchestration.


Here is the full question with all the options, followed by the correct answer and explanation:

---

### **Question:**
When using the OCI Python SDK from a Compute instance, which signer object is used for authentication via **instance principal**?

**Options:**
- **A. `oci.auth.signers.InstancePrincipalsSecurityTokenSigner`**  
- **B. `oci.auth.signers.ConfigFileSigner`**  
- **C. `oci auth signers Instance Principals Security Token Signer`**

---

### **Correct Answer:**
**A. `oci.auth.signers.InstancePrincipalsSecurityTokenSigner`**

---

### **Explanation:**
When authenticating to Oracle Cloud Infrastructure (OCI) from a **Compute instance using instance principals**, the correct signer to use in the **OCI Python SDK** is:

```python
oci.auth.signers.InstancePrincipalsSecurityTokenSigner
```

This signer:
- Uses the **instance's identity and dynamic group membership** to authenticate.
- Does **not require a config file or API keys**, making it ideal for secure, in-cloud automation.

#### Why not the others?
- **B. `ConfigFileSigner`**: Used when authenticating with a local config file (e.g., `~/.oci/config`), not for instance principals.
- **C.** is a **malformed version** of option A and not a valid Python object.

Here is the full question with all the options, followed by the correct answer and explanation:

---

### **Question:**
What is the most direct way to increase the **concurrent request capacity** of a **Dedicated AI Cluster**?

**Options:**
- **A. Increase the `max_output_tokens` parameter in API calls.**  
- **B. Change the cluster's VM shape to one with more vCPUs.**  
- **C. Increase the `unit_count` of the cluster.**

---

### **Correct Answer:**
**C. Increase the `unit_count` of the cluster.**

---

### **Explanation:**
The **`unit_count`** in a Dedicated AI Cluster determines the **number of compute units** allocated to the cluster. Increasing this value:

- Adds more **parallel processing capacity**,  
- Directly increases the number of **concurrent requests** the cluster can handle,  
- Is the most **scalable and straightforward** way to boost throughput.

#### Why not the others?
- **A. `max_output_tokens`**: This controls the **length of the response**, not concurrency. Increasing it may actually increase latency.
- **B. Changing VM shape**: While it may improve performance per request, it doesn't directly scale **concurrent request capacity** like increasing `unit_count` does.


Here is the full question with all the options, followed by the correct answer and explanation:

---

### **Question:**
For **semantic search** across documents in **English, Spanish, and French**, which model type is most appropriate?

**Options:**
- **A. `cohere.command`**  
- **B. `cohere.embed-english-v3.0`**  
- **C. `cohere.embed-multilingual-v3.0`**

---

### **Correct Answer:**
**C. `cohere.embed-multilingual-v3.0`**

---

### **Explanation:**
The **`cohere.embed-multilingual-v3.0`** model is specifically designed for **multilingual semantic understanding**, making it ideal for tasks like **semantic search** across documents written in **multiple languages**, including English, Spanish, and French.

#### Why not the others?
- **A. `cohere.command`**: This is a **text generation model**, not optimized for embedding or semantic search tasks.
- **B. `cohere.embed-english-v3.0`**: This model is optimized for **English-only** embeddings, so it would not perform well on Spanish or French documents.


Here is the full question with all the options, followed by the correct answer and explanation:

---

### **Question:**
After fine-tuning, where are your **custom models stored and managed as versioned artifacts** within OCI?

**Options:**
- **A. In a service-managed Object Storage bucket**  
- **B. In the OCI Model Catalog of your compartment**  
- **C. In OCI Container Registry**

---

### **Correct Answer:**
**B. In the OCI Model Catalog of your compartment**

---

### **Explanation:**
In Oracle Cloud Infrastructure (OCI), after you fine-tune a model, the resulting **custom model** is stored and versioned in the **OCI Model Catalog**. This catalog:

- Acts as a **central repository** for managing ML models,  
- Supports **versioning**, **metadata tracking**, and **deployment integration**,  
- Enables easy reuse and governance of models across teams and projects.

#### Why not the others?
- **A. Object Storage**: While models may be temporarily stored here during training, it's not the official versioned management system.
- **C. OCI Container Registry**: Used for storing container images, not ML models.

Here is the full question with all the options, followed by the correct answer and explanation:

---

### **Question:**
What is OCI's **data privacy commitment** regarding **prompts sent to pre-trained base model endpoints**?

**Options:**
- **A. Prompts are used to improve the base models for all customers.**  
- **B. Prompts are stored for 30 days for troubleshooting purposes.**  
- **C. Prompts are never used to train or improve the base models.**

---

### **Correct Answer:**
**C. Prompts are never used to train or improve the base models.**

---

### **Explanation:**
Oracle Cloud Infrastructure (OCI) maintains a strong commitment to **data privacy and customer isolation**. For **pre-trained base model endpoints**, OCI ensures that:

- **Customer prompts and data are not used** to train or improve the underlying models.
- This guarantees that **your data remains private and isolated**, and is not shared across tenants or used to influence model behavior for others.

#### Why not the others?
- **A.** This would violate OCI’s privacy guarantees.
- **B.** OCI does not retain prompts for training or troubleshooting unless explicitly configured by the customer.


Here is the full question with all the options, followed by the correct answer and explanation:

---

### **Question:**
What is the **primary billing metric** for an **OCI Generative AI fine-tuning job**?

**Options:**
- **A. The number of tokens in the training dataset**  
- **B. The number of hours the fine-tuning job is active**  
- **C. The number of epochs specified for training**

---

### **Correct Answer:**
**B. The number of hours the fine-tuning job is active**

---

### **Explanation:**
OCI bills **Generative AI fine-tuning jobs** primarily based on the **duration** the job runs, measured in **hours**. This includes the time spent on:

- Allocating and using compute resources  
- Processing the training data  
- Running the training loop (regardless of the number of epochs)

#### Why not the others?
- **A. Number of tokens**: While token count affects training time, it’s not the direct billing metric.
- **C. Number of epochs**: This influences training duration, but billing is still based on **actual runtime**, not the number of training cycles.



 **context length includes both the input and the output tokens**.

### 🔁 Here's how it works:
- The **total number of tokens** in a single interaction — including:
  - **System prompt**
  - **User input**
  - **Model output**
- Must **not exceed** the model’s maximum context length.

---

### 📌 Example (with an 8,000-token context limit):
- System prompt: 500 tokens  
- User input: 1,000 tokens  
- Model output: **can be up to 6,500 tokens**

If the combined input already uses 7,500 tokens, the model can only generate **up to 500 tokens** in response.

---

### Why this matters:
- If your prompt is too long, the model may **truncate earlier parts** or **cut off the response**.
- For long documents or conversations, you may need to **summarize**, **chunk**, or use **retrieval-augmented generation (RAG)** to stay within limits.



Great question! Let's break it down clearly.

---

### 🔢 **What is Cumulative Probability?**

In the context of **Top-p sampling** (also called **nucleus sampling**), **cumulative probability** refers to the **running total of probabilities** of the most likely tokens, starting from the top.

---

### 🧠 **Top-p Sampling Explained with an Example**

Imagine a language model is trying to predict the next word, and it gives the following probabilities for the top tokens:

| Token     | Probability |
|-----------|-------------|
| "car"     | 0.30        |
| "truck"   | 0.25        |
| "bike"    | 0.15        |
| "train"   | 0.10        |
| "plane"   | 0.08        |
| "boat"    | 0.07        |
| "scooter" | 0.05        |

Now, suppose **Top-p = 0.80**.

We start adding tokens from the top until the **cumulative probability ≥ 0.80**:

- "car" → 0.30  
- "truck" → 0.30 + 0.25 = 0.55  
- "bike" → 0.55 + 0.15 = 0.70  
- "train" → 0.70 + 0.10 = **0.80** ✅

So, the model will **randomly select the next token from this set**:  
**{ "car", "truck", "bike", "train" }**

Tokens like "plane", "boat", and "scooter" are **excluded** because they fall outside the top-p cumulative threshold.

---

### 🔁 Summary:

- **Top-k**: Fixed number of top tokens (e.g., top 5).
- **Top-p**: Variable number of tokens whose **combined probability ≥ p** (e.g., 0.9).




### ✅ **A. Preamble**

---

### 🔍 Explanation:

- The **preamble** is a field used to provide **context, instructions, or guidance** to the model about how it should behave or respond during a conversation.
- In this case, if you want the model to respond **in the tone of a pirate**, the preamble is where you'd specify that instruction (e.g., "Respond like a pirate in all replies").

---

### ❌ Why the other options are incorrect:

- **B. Temperature**: Controls randomness/creativity in responses, not style or tone.
- **C. Seed**: Used for reproducibility of outputs, not for setting tone or instructions.
- **D. Truncate**: Refers to how long responses are or how input is shortened, not about style.



