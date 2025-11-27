import streamlit as st
import json
import os
import time
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Import Agents and Utility
from agents.document_ingestor import DocumentIngestorAgent 
from agents.thesis_extractor import ThesisExtractorAgent 
from agents.insight_synthesizer import InsightSynthesizerAgent 
from tools.file_processor import save_uploaded_file # NEW UTILITY

# Load environment variables
load_dotenv()

# --- STATE DEFINITION (Same as main.py) ---
class AgentState(TypedDict):
    file_path: str
    original_content: str
    chunks: list
    retriever: object
    thesis_data: dict
    summary_output: dict

# --- AGENT NODES (Adapted from main.py, simplified for Streamlit) ---

# Note: These functions must match the structure of the nodes in your main.py/LangGraph setup
def run_document_ingestor(state: AgentState):
    st.info("Agent 1: Ingesting document and creating RAG index...")
    file_path = state['file_path']
    
    try:
        agent = DocumentIngestorAgent(file_path)
        chunks, full_content = agent.process_document()
        retriever = agent.create_retriever(chunks)
    except FileNotFoundError as e:
        st.error(f"Error: File not found. {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Critical Error during document ingestion: {e}")
        return {"error": str(e)}

    st.success("Analysis Complete. RAG index created.")
    return {
        "original_content": full_content,
        "chunks": chunks,
        "retriever": retriever,
    }

def run_thesis_extractor(state: AgentState):
    st.info("Agent 2: Extracting Thesis and Key Findings...")
    
    if state.get('error') or not state.get('original_content'):
        return state
        
    original_content = state['original_content']
    agent = ThesisExtractorAgent()
    thesis_data = agent.extract_thesis_data(original_content)
    
    st.success("Thesis extraction complete.")
    return {
        "thesis_data": thesis_data
    }

def run_insight_synthesizer(state: AgentState):
    st.info("Agent 3: Synthesizing Final Academic Summary...")
    
    if state.get('error') or not state.get('retriever'):
        return state

    original_content = state['original_content']
    thesis_data = state['thesis_data']
    retriever = state['retriever']
    
    with st.spinner('Invoking LLM to generate summary...'):
        agent = InsightSynthesizerAgent(retriever)
        summary_output = agent.generate_final_summary(original_content, thesis_data)
    
    st.success("Synthesis Complete.")
    return {
        "summary_output": summary_output
    }

# --- LANGGRAPH ORCHESTRATION (Same as main.py) ---

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

# --- STREAMLIT UI DEFINITION ---
def main():
    st.set_page_config(page_title="Academia Analyzer Agent", layout="wide")
    
    st.title("🎓 Academia Analyzer Agent")
    st.subheader("Automated Research Paper Synthesis (Module 2 MVP)")
    st.markdown("---")

    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("FATAL ERROR: OPENROUTER_API_KEY is not set. Please check your .env file.")
        return

    st.sidebar.markdown("### System Architecture")
    st.sidebar.markdown("* **Orchestration:** LangGraph (3 Agents)")
    st.sidebar.markdown("* **LLM:** GPT-3.5-Turbo (via OpenRouter)")
    st.sidebar.markdown("* **RAG:** FAISS + MiniLM Embeddings")

    st.header("1. Upload Research Document")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, or Markdown File",
        type=["pdf", "txt", "md"],
        help="Upload the paper you want the agents to analyze and summarize."
    )
    
    if st.button("🚀 Start Analysis", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a file to begin the analysis.")
            return

        # 1. Save the file locally using the utility tool
        file_path = save_uploaded_file(uploaded_file)
        
        if file_path is None:
            st.error("Could not save file locally. Check permissions.")
            return

        st.subheader("2. Agent Workflow Execution")
        
        app = create_graph()
        inputs = {"file_path": file_path}
        
        try:
            # 2. Run the LangGraph pipeline
            final_state = app.invoke(inputs)

            st.markdown("---")
            st.subheader("3. Final Insight Summary")
            
            summary_output = final_state.get('summary_output', {})
            thesis_data = final_state.get('thesis_data', {})
            
            if summary_output.get('error'):
                 st.error(f"Synthesis failed: {summary_output['error']}")
            else:
                 st.success("Analysis Complete!")
                 
                 st.markdown("### **🌟 Core Synthesis**")
                 st.markdown(f"**Novel Title:** {summary_output.get('novel_title', 'N/A')}")
                 st.markdown(f"**Executive Summary:** {summary_output.get('executive_summary', 'N/A')}")

                 st.markdown("### **📝 Extracted Findings**")
                 st.markdown(f"**Primary Hypothesis:** {thesis_data.get('primary_hypothesis', 'N/A')}")
                 st.markdown(f"**Key Findings:** {summary_output.get('key_findings', 'N/A')}")

                 st.markdown("### **💡 Discussion Points**")
                 st.markdown("* " + "\n* ".join(summary_output.get('discussion_points', [])))
            
        except Exception as e:
            st.error(f"An unexpected error occurred during workflow: {e}")
        
        # Clean up the saved file after analysis
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    main()