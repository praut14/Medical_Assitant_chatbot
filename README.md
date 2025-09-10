# Medical Assistant QA — README (Dataset-specific)

This README is tailored to the supplied dataset **`mle_screening_dataset.csv`** and the accompanying Python/Colab code that performs RAG-style Q&A over CSV rows.

## Dataset Summary
- **File**: `mle_screening_dataset.csv`
- **Shape**: 16406 rows × 2 columns
- **Columns**: question, answer
- **Numeric columns** (for stats/aggregations): —
- **Text columns** (used for semantic retrieval): question, answer
- **Common ID fields** (best-effort guess): —
- **Potential score/metric fields** (best-effort guess): —
- **Potential label/ground-truth fields** (best-effort guess): answer

*Top missingness (fraction of NaN):*
        missing_frac
answer           0.0

## Approach (Concise)
1. **Row-to-Document Conversion**: Each CSV row is serialized into a compact text string (`col=value; ...`) so it can be retrieved as a “document.”  
2. **Retrieval**:  
   - *Default fast path*: **TF‑IDF** similarity to fetch top‑k relevant rows for a natural question.  
   - *RAG path*: **Sentence‑Transformers** embeddings + **FAISS** (or **Pinecone**) for semantic search.  
3. **Answer Synthesis**:  
   - *Local*: a small seq2seq model (e.g., FLAN‑T5) summarizes the retrieved rows.  
   - *Deterministic tools*: when questions clearly ask for min/max/avg over a numeric column, a small helper computes the value directly from the DataFrame.  
4. **(Optional) Quality Evaluation**: **RAGAS** metrics (faithfulness, answer relevancy, context precision/recall) to spot hallucinations and retrieval issues.

## Assumptions
- The CSV rows contain enough self‑contained facts to answer most assignment questions without web search.  
- Row‑level chunking is sufficient (no need for multi‑row joins beyond top‑k retrieval).  
- Basic normalization (lowercasing, trimming) suffices; medical abbreviations and synonyms are handled implicitly by embeddings.  
- For numeric queries, direct computation from the DataFrame is the source of truth; the generator should cite row IDs/values.

## Model Performance (Strengths & Weaknesses)
**Strengths**
- Fast, explainable Q&A: answers are grounded in **retrieved rows**, which you can display as evidence.  
- Handles both **keyword** and **semantic** matches (TF‑IDF or embeddings).  
- Numeric questions (e.g., “highest/lowest/average of a metric”) return **deterministic** results via pandas helpers.

**Weaknesses**
- If important information is spread across **multiple rows**, a small generator may miss cross‑row reasoning unless explicitly prompted.  
- Retrieval quality depends on how informative each **row serialization** is (very wide tables may truncate values).  
- Medical synonyms/abbreviations may require **custom normalization** or a domain embedding model.

## Potential Improvements / Extensions
- **Domain‑aware chunking**: add clinical ontologies/sections to the row text (e.g., symptoms, diagnosis, medications, ICD‑10 tags).  
- **Re‑ranking**: add a **cross‑encoder** to re‑rank the top‑k for higher precision before generating answers.  
- **Tool use & aggregation**: expose pandas/SQL tools for multi‑row joins and calculations (totals, group‑bys).  
- **Better embeddings**: try **biomedical sentence embeddings** (e.g., BioClinicalBERT sentence embeddings) for improved recall on medical terminology.  
- **RAGAS‑driven loops**: collect a small eval set of real questions and iterate on prompts/chunking until **faithfulness** and **context metrics** improve.  
- **Guardrails**: enforce citation of row IDs and require “insufficient information” when context is missing.

## Example Questions This Dataset Supports
- “Which row has the **highest** or **lowest** value of *\<metric column\>*?”  
- “List the **top 3 rows** most relevant to *\<condition/symptom/term\>* and summarize why.”  
- “Compute the **average** of *\<numeric column\>* for transparency and cite the supporting rows.”

---

> **How to run**: Use the provided Colab cells to (a) load this CSV, (b) build TF‑IDF or FAISS/Pinecone retrieval, and (c) call `rag_answer("your question")`. For grading, run the **RAGAS** block on a small set of questions; export scores as `ragas_scores.csv`.
