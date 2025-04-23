import os
import json
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv
import google_llm_service
# Load environment variables
load_dotenv()

# Define the output schema for the guardrail
class CorrectQuestion(BaseModel):
    query: str

class Task:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def validate_pdf_path(self, pdf_path: str) -> str:
        """Validate if the PDF path exists and is correct."""
        if not os.path.isfile(pdf_path):
            raise ValueError(f"The provided path '{pdf_path}' does not exist or is not a valid file.")
        return pdf_path

    def input(self, query: str, pdf_path: str) -> dict:
        """Function to query Gemini and return the response in JSON format."""
        try:
            print(f"Received query: {query}")
            print(f"Received PDF path: {pdf_path}")

            # Generate a better prompt for grammatical correction
            prompt = f"""You are an English expert. Your task is to ensure the given question is grammatically correct with no spelling mistakes.
                         Do this without changing the intent of the question.
                         Question: {query}
                         Please return the corrected question."""
            
            # Query Gemini
            response = self.client.models.generate_content(
                model="gemini-1.5-pro-latest",
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CorrectQuestion,
                ),
            )
            

            
            # Parse the response
            output = json.loads(response.text)
            print(f"Corrected query: {output['query']}")

            pdf_text = ""
            # Validate and add PDF path only if provided
            if pdf_path.strip():  # Check if pdf_path is not empty
                validated_pdf_path = self.validate_pdf_path(pdf_path)
                output['pdf_path'] = validated_pdf_path
                print(f"Validated PDF path: {validated_pdf_path}")
                pdf_text = google_llm_service.GoogleLLM().getResults(pdf_path)
            else:
                print("No PDF path provided. Skipping validation.")
            
            output["query"] = pdf_text + "\n" + query
            return output
        
        except ValueError as e:
            print(f"Error: {e}")
            return {"error": str(e), "query": query}


