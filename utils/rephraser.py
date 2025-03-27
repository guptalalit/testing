import os
import json
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define the output schema for the rephrased query
class RephrasedQueryOutput(BaseModel):
    rephrased_query: str  # The rephrased query

def rephrase_query(original_query: str, reason: str,answer:str) -> dict:
    """
    Generate a rephrased query based on the original query and the reason for validation failure.

    Args:
        original_query (str): The original query.
        reason (str): The reason why the previous answer was incorrect.

    Returns:
        dict: A dictionary containing the rephrased query.
    """
    # Construct the prompt for rephrasing
    prompt = f"""You are a highly skilled query rephraser specializing in enhancing the precision and relevancy of questions based on specific criteria. Please follow these instructions carefully:

                Original Query: {original_query} - This is the question that requires rephrasing.
                Reason for Incorrect Answer: {reason} - This indicates why the previous answer was deemed incorrect, focusing on faithfulness and answer relevancy according to RAG (Retrieval-Augmented Generation) metrics.
                Requirements: Your rephrased query should:
                Maintain the original intent of the question.
                Improve clarity and specificity.
                Address the identified shortcomings related to faithfulness and relevancy.
                Output Format:

                Rephrased Query:
                Please ensure that the rephrased query is concise, clear, and directly aligned with the provided reason for the previous answer's inadequacy.
                """

    # Query Gemini for rephrased query
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RephrasedQueryOutput,
        ),
    )

    # Parse the response
    rephrased_result = json.loads(response.text)
    return rephrased_result


