import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# --- Pydantic Schema for Structured Output ---
class ExtractedThesis(BaseModel):
    """Structured output model for primary academic components."""
    primary_hypothesis: str = Field(description="The main hypothesis, claim, or objective of the paper.")
    methodology_keywords: list[str] = Field(description="5-7 technical keywords representing the methods used (e.g., 'Transformer-XL', 'Monte Carlo Simulation').")
    key_findings: str = Field(description="The single most important conclusion or result achieved.")

class ThesisExtractorAgent:
    """Agent 2: Extracts core academic data using the LLM."""
    def __init__(self):
        # Using ChatOpenAI compatible with OpenRouter
        self.llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",  
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1  # Low temperature for factual extraction
        )

    def extract_thesis_data(self, content: str):
        """Uses the LLM to parse and extract structured academic data."""
        
        # We ensure the content is truncated to fit within the LLM's context window
        MAX_TOKENS = 5000  
        truncated_content = content[:MAX_TOKENS] 

        # The prompt guides the LLM to its task and defines the output format
        prompt_template = PromptTemplate(
            template="""You are an expert thesis extractor. Analyze the provided academic text and strictly extract the main hypothesis, the key findings, and the technical methodology used.

            TEXT SAMPLE: {content}

            {format_instructions}
            """,
            input_variables=["content"],
            partial_variables={"format_instructions": JsonOutputParser(pydantic_object=ExtractedThesis).get_format_instructions()},
        )

        # Chain the prompt and parser, then invoke the LLM
        parser = JsonOutputParser(pydantic_object=ExtractedThesis)
        
        full_prompt = prompt_template.invoke({"content": truncated_content})

        try:
            # We use invoke and parse the raw string output
            response = self.llm.invoke(full_prompt.text)
            
            # The LLM's response is a JSON string, which we load and return as a dict
            parsed_data = parser.parse(response.content)
            return parsed_data
        
        except Exception as e:
            print(f"LLM Thesis Extraction Failed: {e}")
            return {"error": str(e), "content": truncated_content}