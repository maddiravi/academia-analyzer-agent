import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# --- Pydantic Schema for Structured Output ---
class InsightSummary(BaseModel):
    """Structured output model for the final academic summary."""
    novel_title: str = Field(description="A concise, attention-grabbing title for the paper summary.")
    executive_summary: str = Field(description="A one-paragraph summary detailing the motivation, method, and conclusion.")
    discussion_points: list[str] = Field(description="3-5 critical discussion points or future research directions.")

class InsightSynthesizerAgent:
    """Agent 3: Uses RAG to generate structured academic insights."""
    def __init__(self, retriever):
        # Using ChatOpenAI compatible with OpenRouter
        self.llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",  
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3 # Higher temperature for synthesis/creativity
        )
        self.retriever = retriever

    def generate_final_summary(self, original_content: str, thesis_data: dict):
        """Generates structured insights based on RAG and thesis data."""
        
        # RAG retrieval based on synthesis task
        retrieved_docs = self.retriever.invoke("synthesize the main argument, methodology, and key results of the document")
        retrieved_context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt_template = PromptTemplate(
            template="""You are an expert academic synthesizer. Your task is to analyze the core thesis data and the detailed context, and generate a highly structured, objective, and concise summary. 

            Use the RETRIEVED CONTEXT to ensure the summary is grounded in the document's facts.

            CORE THESIS DATA (From Agent 2): {thesis_data}
            RETRIEVED CONTEXT (Detailed passages from paper): {context}

            Generate the required structured summary, focusing on novelty and academic rigor.
            
            {format_instructions}
            """,
            input_variables=["context", "thesis_data"],
            partial_variables={"format_instructions": JsonOutputParser(pydantic_object=InsightSummary).get_format_instructions()},
        )

        full_prompt = prompt_template.invoke({"context": retrieved_context, "thesis_data": thesis_data})

        parser = JsonOutputParser(pydantic_object=InsightSummary)
        
        try:
            print("Invoking LLM for synthesis...")
            # We use invoke and parse the raw string output
            response = self.llm.invoke(full_prompt.text)
            parsed_data = parser.parse(response.content)
            print("Synthesis complete.")
            return parsed_data
        
        except Exception as e:
            print(f"LLM Final Synthesis Failed: {e}")
            return {"error": str(e)}