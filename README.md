# 🩺 ClinicaSense AI

## Agentic AI Clinical Decision Support System

ClinicaSense AI is a demonstration prototype that combines **Agentic AI, Retrieval-Augmented Generation (RAG), vector search, clinical safety screening, evidence quality checks, LLM reasoning, and output guardrails**.

The project is designed to demonstrate how an AI-powered clinical decision-support workflow can be orchestrated using **LangGraph**, **Google Gemini**, **ChromaDB**, and **Streamlit**.

> ⚠️ **Important:** ClinicaSense AI is a prototype for educational and demonstration purposes only. It is **not a medical diagnostic system**, does not provide professional medical advice, and must not be used as a substitute for qualified healthcare professionals or emergency services.

---

## 📌 Project Objective

The objective of ClinicaSense AI is to demonstrate an end-to-end Agentic AI workflow for clinical decision support.

The system accepts a clinical case and performs:

1. Initial safety screening
2. High-risk case detection
3. Intent classification
4. Clinical domain classification
5. Domain-aware RAG retrieval
6. Evidence quality evaluation
7. Evidence-grounded prompt construction
8. LLM-based clinical reasoning
9. Output safety validation
10. Final clinical decision-support response

The architecture follows a **Safety First → Evidence First → LLM Reasoning** approach.

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      User / User     │
                         │    Clinical Case     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Streamlit UI     │
                         │       main.py        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Clinical Agent     │
                         │  clinical_agent.py   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      LangGraph       │
                         │   Agentic Workflow   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Safety Screening    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    HIGH RISK               NORMAL
                         │                     │
                         ▼                     ▼
                ┌────────────────┐    ┌──────────────────┐
                │ Safety Response│    │  Intent Router   │
                └───────┬────────┘    └────────┬─────────┘
                        │                      │
                        │                      ▼
                        │             ┌──────────────────┐
                        │             │ Clinical Domain  │
                        │             │     Router       │
                        │             └────────┬─────────┘
                        │                      │
                        │                      ▼
                        │             ┌──────────────────┐
                        │             │    Medical RAG   │
                        │             └────────┬─────────┘
                        │                      │
                        │                      ▼
                        │             ┌──────────────────┐
                        │             │    ChromaDB      │
                        │             │  Vector Search    │
                        │             └────────┬─────────┘
                        │                      │
                        │                      ▼
                        │             ┌──────────────────┐
                        │             │ Evidence Quality │
                        │             │      Gate        │
                        │             └────────┬─────────┘
                        │                      │
                        │              ┌───────┴────────┐
                        │              │                │
                        │        INSUFFICIENT        SUFFICIENT
                        │              │                │
                        │              ▼                ▼
                        │      ┌───────────────┐  ┌──────────────┐
                        │      │ Safe Response │  │ Prompt Build │
                        │      └───────┬───────┘  └──────┬───────┘
                        │              │                 │
                        │              │                 ▼
                        │              │          ┌──────────────┐
                        │              │          │ Gemini LLM   │
                        │              │          └──────┬───────┘
                        │              │                 │
                        │              │                 ▼
                        │              │          ┌──────────────┐
                        │              │          │   Output     │
                        │              │          │  Guardrail   │
                        │              │          └──────┬───────┘
                        │              │                 │
                        └──────────────┴─────────────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ Final Response    │
                              │ + Evidence        │
                              │ + Safety Status   │
                              └──────────────────┘
```

---

# 🔄 End-to-End Workflow

## 1. Clinical Case Input

The user enters a clinical case through the Streamlit interface.

Example:

```text
58-year-old male with a history of hypertension and
repeatedly elevated blood pressure readings.
```

The request is passed to:

```python
run_clinical_agent(clinical_case)
```

---

## 2. Safety Screening

The first LangGraph node executes a safety check.

The current prototype uses rule-based high-risk symptom detection.

Examples of high-risk terms include:

```text
chest pain
difficulty breathing
shortness of breath
severe bleeding
unconscious
stroke
seizure
suicidal
```

If high-risk symptoms are detected, the workflow is immediately routed to a safety response.

```text
Clinical Case
     │
     ▼
Safety Screening
     │
     ├── HIGH RISK ──► Safety Response ──► END
     │
     └── NORMAL ─────► Continue Workflow
```

This prevents normal RAG and LLM reasoning from being executed for potentially urgent cases.

---

## 3. Intent Routing

Normal cases are classified as:

```text
CLINICAL_QUESTION
```

High-risk cases are classified as:

```text
HIGH_RISK
```

The intent router controls the next step in the LangGraph workflow.

---

## 4. Clinical Domain Classification

For normal clinical questions, Gemini is used to classify the clinical domain.

Current supported domains:

```text
CARDIOVASCULAR
HYPERTENSION
DIABETES
RESPIRATORY
MEDICATION_SAFETY
GENERAL_CLINICAL
```

Example:

```text
Input:
Patient has repeatedly elevated blood pressure.

Domain:
HYPERTENSION
```

---

## 5. RAG Retrieval

The clinical case is converted into an embedding using the configured Hugging Face embedding model.

The embedding is searched against ChromaDB.

The current RAG flow is:

```text
Clinical Case
      │
      ▼
Embedding Model
      │
      ▼
Query Embedding
      │
      ▼
ChromaDB
      │
      ▼
Domain-Aware Vector Search
      │
      ▼
Relevant Clinical Evidence
```

The vector search can filter documents by:

```text
clinical_domain
```

For example:

```text
HYPERTENSION
```

will prioritize documents with:

```text
domain = HYPERTENSION
```

---

## 6. Evidence Quality Gate

Retrieved evidence is evaluated before sending the request to the LLM.

The workflow has two paths:

```text
Retrieved Evidence
       │
       ▼
Evidence Quality Gate
       │
       ├── Insufficient ──► Safe Insufficient-Evidence Response
       │
       └── Sufficient ────► Continue to LLM
```

This helps reduce unsupported or hallucinated answers.

---

## 7. Prompt Construction

If sufficient evidence is available, the system creates an evidence-grounded prompt containing:

- System safety instructions
- Safety screening result
- Clinical domain
- Retrieved clinical evidence
- User clinical case

The prompt instructs Gemini to:

- Avoid diagnosis claims
- Avoid unsupported treatment decisions
- Clearly communicate uncertainty
- Use retrieved evidence
- Avoid inventing citations or medical evidence
- Highlight safety considerations
- Recommend professional evaluation where appropriate

---

## 8. Gemini Clinical Reasoning

The configured Gemini model generates the final clinical decision-support response.

The LLM is used after:

```text
Safety Screening
+
Domain Routing
+
RAG Retrieval
+
Evidence Quality Check
```

This means the LLM is not operating as an unrestricted chatbot.

---

## 9. Output Guardrail

The generated response is checked for potentially unsafe phrases.

Example phrases currently checked include:

```text
you definitely have
you are diagnosed with
take this medication
stop your medication
```

The system returns:

```text
output_safe = True
```

or:

```text
output_safe = False
```

This provides a basic post-generation safety validation layer.

---

# 🧰 Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| LLM | Google Gemini |
| Embeddings | Hugging Face Embedding Model |
| Vector Database | ChromaDB |
| RAG | Custom RAG Pipeline |
| State Management | Python TypedDict |
| Safety | Rule-Based Safety Guardrails |
| Output Validation | Custom Output Guardrail |
| Knowledge Base | Local TXT Documents |
| Environment | Python Virtual Environment |
| Development Platform | macOS |

---

# 📁 Project Structure

```text
clinical-sense-ai/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── config.py
│   │
│   ├── agent/
│   │   └── clinical_agent.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   ├── guardrails/
│   │   └── safety.py
│   │
│   ├── llm/
│   │   └── gemini_client.py
│   │
│   ├── prompts/
│   │   ├── clinical_prompt.py
│   │   └── router_prompt.py
│   │
│   └── rag/
│       ├── embeddings.py
│       ├── index_documents.py
│       ├── retriever.py
│       ├── test_retriever.py
│       └── vector_store.py
│
├── knowledge_base/
│   ├── cardiovascular_demo.txt
│   ├── hypertension_demo.txt
│   ├── diabetes_demo.txt
│   ├── respiratory_demo.txt
│   └── medication_safety_demo.txt
│
├── chroma_db/
│
├── .env
├── requirements.txt
└── README.md
```

---

# 🚀 Setup

## Prerequisites

Install:

- Python 3.10+
- pip
- Git
- Google Gemini API key

---

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd clinical-sense-ai
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file.

Example:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Use the exact environment variable expected by `app/llm/gemini_client.py`.

> Never commit `.env` or API keys to Git.

---

# 📚 Build the RAG Knowledge Base

Whenever knowledge-base documents are added or modified, rebuild the ChromaDB index.

Remove the existing local vector database:

```bash
rm -rf chroma_db
```

Run indexing:

```bash
python -m app.rag.index_documents
```

Expected output:

```text
Loading medical knowledge...
Loaded documents
Generating embeddings...
Adding documents to ChromaDB...
Medical knowledge indexed successfully.
```

---

# 🔎 Test the Retriever

Run:

```bash
python -m app.rag.test_retriever
```

This verifies that the embedding and ChromaDB retrieval pipeline is working.

Expected output includes:

```text
Retrieved Evidence:
...
Sources:
...
```

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app/main.py
```

Open the local Streamlit URL displayed in the terminal.

---

# 🧪 Test Cases

The following test cases can be used for a project demonstration.

## TC01 — High-Risk Chest Pain

### Input

```text
58-year-old male with severe chest pain and shortness of breath.
```

### Expected

```text
Intent: HIGH_RISK
```

Expected flow:

```text
Safety Screening
      ↓
HIGH_RISK
      ↓
High-Risk Safety Response
      ↓
END
```

Expected behavior:

- High-risk alert displayed
- Urgent professional evaluation recommended
- Normal RAG workflow skipped
- Normal LLM clinical reasoning skipped

---

## TC02 — Stroke Symptoms

### Input

```text
Patient suddenly developed weakness on one side of the body
and difficulty speaking.
```

### Expected

```text
HIGH_RISK
```

The system should route directly to the safety response.

---

## TC03 — Hypertension

### Input

```text
Patient has a history of hypertension and repeatedly elevated
blood pressure readings.
```

### Expected

```text
Intent: CLINICAL_QUESTION
Domain: HYPERTENSION
```

Expected workflow:

```text
Safety
  ↓
Intent
  ↓
HYPERTENSION
  ↓
RAG
  ↓
Evidence Quality
  ↓
Prompt
  ↓
Gemini
  ↓
Output Guardrail
```

---

## TC04 — Diabetes

### Input

```text
Patient with diabetes has elevated blood glucose levels
and is being evaluated for long-term risk factors.
```

### Expected

```text
Domain: DIABETES
```

The RAG layer should retrieve diabetes-related evidence.

---

## TC05 — Respiratory

### Input

```text
Patient has a history of asthma and recurrent wheezing.
```

### Expected

```text
Domain: RESPIRATORY
```

---

## TC06 — Medication Safety

### Input

```text
Patient is taking multiple medications and there is concern
about possible medication interactions.
```

### Expected

```text
Domain: MEDICATION_SAFETY
```

---

## TC07 — General Clinical

### Input

```text
Patient presents with a general clinical concern that does
not clearly belong to a supported domain.
```

### Expected

```text
Domain: GENERAL_CLINICAL
```

---

## TC08 — Insufficient Evidence

Ask a question that is outside the current demo knowledge base.

### Expected

```text
Evidence Quality: INSUFFICIENT
```

The system should return an insufficient-evidence response instead of generating unsupported clinical claims.

---

## TC09 — Output Guardrail

For testing, temporarily mock or force an LLM response containing:

```text
You definitely have hypertension.
```

### Expected

```text
output_safe = False
```

The response should be flagged for potentially unsafe clinical language.

---

## TC10 — Empty Input

Submit the form without entering a clinical case.

### Expected

```text
Please enter a clinical case.
```

No clinical agent workflow should execute.

---

# 📊 Test Case Summary

| ID | Test Scenario | Expected Result |
|---|---|---|
| TC01 | Chest pain + shortness of breath | HIGH_RISK |
| TC02 | Stroke symptoms | HIGH_RISK |
| TC03 | Hypertension | HYPERTENSION |
| TC04 | Diabetes | DIABETES |
| TC05 | Asthma / wheezing | RESPIRATORY |
| TC06 | Medication interaction | MEDICATION_SAFETY |
| TC07 | Unknown domain | GENERAL_CLINICAL |
| TC08 | Unsupported question | INSUFFICIENT EVIDENCE |
| TC09 | Unsafe LLM phrase | Guardrail failure |
| TC10 | Empty input | Validation warning |

---

# 🛡️ Safety and Responsible AI

ClinicaSense AI is a prototype and has several safety limitations.

### Current Safety Features

- Rule-based high-risk symptom detection
- High-risk workflow bypass
- Evidence-grounded prompting
- Evidence quality gate
- Output unsafe-phrase detection
- Clinical disclaimer
- Explicit uncertainty handling

### Current Limitations

The current prototype:

- Does not diagnose patients
- Does not provide validated medical treatment recommendations
- Uses a limited demonstration knowledge base
- Uses keyword-based safety detection
- Uses prototype retrieval confidence scoring
- Has not undergone clinical validation
- Has not undergone regulatory approval
- Should not be used for real clinical decision-making

---

# 🔐 Security Considerations

For a production system, the following would be required:

- Secure API key management
- Secrets manager
- Authentication and authorization
- Role-based access control
- Encryption
- Audit logging
- PHI/PII protection
- Data retention policies
- HIPAA/GDPR or applicable regulatory compliance
- Secure deployment
- Model monitoring

The current demo should not be used with real patient-identifiable information.

---

# 🔮 Future Enhancements

Possible future improvements include:

### RAG

- Document chunking
- Hybrid keyword + vector search
- Re-ranking
- Better similarity thresholds
- Larger validated medical knowledge base
- Medical guideline integration

### AI

- Claude/OpenAI/local LLM support
- Model abstraction layer
- Structured LLM output
- Function/tool calling
- Multi-agent architecture

### Clinical

- Validated medical guidelines
- Evidence citations
- Source ranking
- Evidence provenance
- Human-in-the-loop review

### Enterprise

- Authentication
- RBAC
- Audit logs
- Observability
- Monitoring
- API layer
- Docker
- CI/CD
- Cloud deployment

### Healthcare Integration

Potential future integration with:

- FHIR
- EHR systems
- Clinical data platforms

---

# 🎯 Demo Presentation

A recommended live demonstration sequence:

### Demo 1 — High-Risk Routing

Input:

```text
58-year-old male with severe chest pain and shortness of breath.
```

Show:

```text
Safety Screening
      ↓
HIGH_RISK
      ↓
Safety Response
      ↓
RAG and normal LLM reasoning skipped
```

### Demo 2 — Normal Clinical RAG

Input:

```text
Patient has a history of hypertension and repeatedly elevated
blood pressure readings.
```

Show:

```text
Safety
 ↓
Clinical Question
 ↓
HYPERTENSION
 ↓
RAG
 ↓
Evidence Quality
 ↓
Gemini
 ↓
Output Guardrail
 ↓
Final Response
```

### Demo 3 — Insufficient Evidence

Ask something outside the knowledge base.

Show:

```text
Evidence Quality
      ↓
INSUFFICIENT
      ↓
Safe Response
```

This demonstrates that the system does not blindly ask the LLM to answer everything.

---

# 🏆 Key AI Engineering Concepts Demonstrated

This project demonstrates:

- Python application development
- LLM integration
- Google Gemini
- Prompt engineering
- Retrieval-Augmented Generation
- Embeddings
- Vector databases
- ChromaDB
- LangGraph
- Agentic workflows
- Conditional routing
- Domain classification
- Safety guardrails
- Evidence quality gating
- Output validation
- Streamlit UI
- State-based workflow orchestration

---

# 📌 Project Status

**Status:** Demo / Proof of Concept

**Architecture:** Agentic AI + RAG

**LLM:** Google Gemini

**Vector Database:** ChromaDB

**Workflow Framework:** LangGraph

**UI:** Streamlit

**Knowledge Base:** Local demonstration clinical documents

**Production Ready:** No

**Clinical Use:** No

---

# 👨‍💻 Author

**ClinicaSense AI**

An Agentic AI demonstration project showcasing the combination of:

```text
LLM
+
RAG
+
Vector Database
+
LangGraph
+
Safety Guardrails
+
Evidence Quality
+
Streamlit
```

---

## ⚠️ Disclaimer

This project is intended solely for educational, research, and demonstration purposes.

ClinicaSense AI does not provide medical diagnosis, treatment, or professional medical advice. It is not a substitute for consultation with qualified healthcare professionals. In an emergency or potentially life-threatening situation, users should contact appropriate emergency services or seek immediate professional medical attention.
