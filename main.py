import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import json
import time

# --- IMPORT NEW AGENT ROLES ---
from agents.document_ingestor import DocumentIngestorAgent 
from agents.thesis_extractor import ThesisExtractorAgent 
from agents.insight_synthesizer import InsightSynthesizerAgent 


# Load environment variables
load_dotenv()
# Check for API Key presence (Mandatory for Agent 3)
if not os.getenv("OPENROUTER_API_KEY"):
    print("FATAL ERROR: OPENROUTER_API_KEY is not set in the .env file.")
    exit(1)


# --- STATE DEFINITION ---
class AgentState(TypedDict):
    """Represents the state of our graph for academic analysis (Module 2 MVP)."""
    file_path: str
    original_content: str
    chunks: list
    retriever: object
    thesis_data: dict # Holds keywords, hypotheses, and findings
    summary_output: dict

# --- AGENT NODES (Module 2 MVP Logic) ---

def run_document_ingestor(state: AgentState):
    print("\n--- 🧠 Running DocumentIngestorAgent (Step 1/3: Ingestion and RAG Setup) ---")
    file_path = state['file_path']
    
    try:
        agent = DocumentIngestorAgent(file_path)
        chunks, full_content = agent.process_document()
        retriever = agent.create_retriever(chunks)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"CRITICAL ERROR during document ingestion: {e}")
        return {"error": str(e)}

    return {
        "original_content": full_content,
        "chunks": chunks,
        "retriever": retriever,
    }

def run_thesis_extractor(state: AgentState):
    print("\n--- 📝 Running ThesisExtractorAgent (Step 2/3: Data Extraction) ---")
    
    # Graceful exit if Agent 1 failed
    if state.get('error') or not state.get('original_content'):
        print("Skipping Agent 2 due to previous failure.")
        return state
        
    original_content = state['original_content']
    agent = ThesisExtractorAgent()
    thesis_data = agent.extract_thesis_data(original_content)
    
    return {
        "thesis_data": thesis_data
    }

def run_insight_synthesizer(state: AgentState):
    print("\n--- 💡 Running InsightSynthesizerAgent (Step 3/3: Final Synthesis) ---")
    
    # Graceful exit if a previous agent failed
    if state.get('error') or not state.get('retriever'):
        print("Skipping Agent 3 due to previous failure.")
        return state

    original_content = state['original_content']
    thesis_data = state['thesis_data']
    retriever = state['retriever']
    
    agent = InsightSynthesizerAgent(retriever)
    summary_output = agent.generate_final_summary(original_content, thesis_data)
    
    return {
        "summary_output": summary_output
    }

# --- LANGGRAPH ORCHESTRATION ---

def create_graph():
    """Defines the sequential flow using LangGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("ingest_document", run_document_ingestor)
    workflow.add_node("extract_thesis", run_thesis_extractor)
    workflow.add_node("synthesize_insight", run_insight_synthesizer)

    workflow.add_edge(START, "ingest_document")
    workflow.add_edge("ingest_document", "extract_thesis")
    workflow.add_edge("extract_thesis", "synthesize_insight")
    workflow.add_edge("synthesize_insight", END)

    return workflow.compile()

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    
    # --- IMPORTANT: PLACE YOUR TEST FILE IN data/sample_docs/ ---
    TEST_FILE = "data/sample_docs/sample_paper.pdf" 
    
    if not os.path.exists(TEST_FILE):
         print(f"\nFATAL ERROR: Test file not found at '{TEST_FILE}'.")
         print("Please place a PDF or TXT file named 'sample_paper.pdf' there.")
         exit(1)

    app = create_graph()
    
    inputs = {"file_path": TEST_FILE}
    print(f"\n--- Starting Analysis of: {TEST_FILE} ---")
    start_time = time.time()
    
    final_state = app.invoke(inputs)

    # --- Print Final Results ---
    print("\n" + "="*50)
    print(f"        ✅ MODULE 2 MVP WORKFLOW COMPLETE ({time.time() - start_time:.2f}s) ✅")
    print("="*50)
    
    if final_state.get('error'):
        print(f"Workflow Halted: {final_state['error']}")
    
    # Display Extracted Thesis Data
    if final_state.get('thesis_data'):
        print("\n[STEP 2 RESULTS] Extracted Thesis Data:")
        print(json.dumps(final_state['thesis_data'], indent=2))
        
    # Display Final Synthesis
    if final_state.get('summary_output'):
        print("\n[STEP 3 RESULTS] Final Insight Synthesis:")
        # Summary output is a dictionary, use json.dumps for clean printing
        print(json.dumps(final_state['summary_output'], indent=2))