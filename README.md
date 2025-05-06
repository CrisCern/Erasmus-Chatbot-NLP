# Erasmus-Chatbot-NLP

**Intelligent chatbot for Erasmus student support, powered by NLP, sentiment analysis, topic modeling, and retrieval-augmented generation.**

> Created by: **Cristian Cernicchiaro** & **Enrico Castori**  
> Course: *Web Analytics and Text Analysis* – University of Cagliari, 2025

---

##  Project Objectives

- Help Erasmus students find clear answers to common questions
- Analyze real support group messages to identify recurring themes
- Extract relevant entities and emotions using NLP techniques
- Build a chatbot using a Retrieval-Augmented Generation (RAG) pipeline

---

##  Main Components

- **Named Entity Recognition (NER)** with spaCy and custom EntityRuler
- **Sentiment & Emotion Analysis** using HuggingFace models
- **Topic Modeling** via BERTopic
- **RAG Architecture** combining dense/sparse retrieval and LLM-based response
- **Simple CLI Interface** for user interaction and testing

---

## ️ Technologies Used

- Python 3.10+
- spaCy + EntityRuler
- HuggingFace Transformers
- BERTopic
- Haystack (retrieval + pipelines)
- bge-large, BM25, CrossEncoder, Zephyr-7b-beta
- Pandas, Scikit-learn, Matplotlib

---


---

## 💬 How It Works

1. User asks a question (e.g., "Where do I upload the OLA?")
2. The query is embedded using `bge-large`
3. A hybrid retriever (BM25 + Dense) selects relevant documents
4. A prompt is constructed combining context and user input
5. The `Zephyr-7b-beta` model generates a tailored response
6. The chatbot can be tested directly via the CLI

---

##  Evaluation

- BLEU and ROUGE scores for automatic evaluation
- Manual quality check on selected conversations
- Identified limitations and improvement suggestions included

---

##  Authors

- **Cristian Cernicchiaro**
- **Enrico Castori**

Academic project – University of Cagliari – Spring 2025  
Developed for the course *Web Analytics and Text Analysis*

---

##  License

This project was developed exclusively for academic purposes.  
Any form of reuse, redistribution, or modification is **strictly prohibited** without prior written consent.  
All rights reserved.  

© Cristian Cernicchiaro & Enrico Castori, 2025
