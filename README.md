#Commands:
#for Rag testing : python -m app.rag.index_documents
#for test Rag :app/rag/test_retriever.py

#Rebuild cromadb :python -m app.rag.index_documents

#Start streamlite : python -m streamlit run app/main.py
#Current Workflow :
                  ┌──────────────────┐
                  │  Clinical Input  │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ Safety Screening │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │  Intent Router   │
                  └──────┬─────┬─────┘
                         │     │
                   HIGH RISK   │ CLINICAL
                         │     │
                         ↓     ↓
                  Safety Response
                               ↓
                     Domain Router
                               ↓
                     Medical RAG
                               ↓
                  Evidence Quality Gate
                               ↓
                      Prompt Builder
                               ↓
                          Gemini
                               ↓
                     Output Guardrail
                               ↓
                       Final Response