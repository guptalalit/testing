import os
from typing import Dict, Optional
from dotenv import load_dotenv
# from validate.rephraser import rephrase_query
from utils.rephraser import rephrase_query
from tools.toolRAG import RAG
from validate.validateLLM import validate_answer
from memory.memory import Memory
import google.generativeai as genai
from openai import OpenAI
import json
import utils.google_llm_services as googleserv


class ModelProvider:
    def __init__(self):
        load_dotenv()
        self.provider = os.getenv("MODEL_PROVIDER").lower()
        print("Provider: ",self.provider)
        self._validate_config()
        self._initialize_provider()

    def _validate_config(self):
        if self.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is required in .env file")
        if self.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required in .env file")
        if self.provider not in ["gemini", "openai"]:
            raise ValueError(f"Unsupported model provider: {self.provider}")

    def _initialize_provider(self):
        if self.provider == "gemini":
            
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model =genai.GenerativeModel('gemini-1.5-pro-latest')
        else:
            
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str) -> str:
        if self.provider == "gemini":
            response = self.model.generate_content(prompt)
            print(response.text)
            return response.text
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content 

class Planner:
    def __init__(self):
        self.num_attempts = 3
        self.model_provider = ModelProvider()
        self.rag = RAG()

    def sequence(self, parser,application_name) -> str:
        counter = 0
        query = parser['query']
        print(f"Initial Query: {query}")

        while counter < self.num_attempts:
            print(f"\nAttempt {counter + 1}:")
            
            try:
                # Get context from RAG
                print("app: ", application_name, parser['sessionid'], query)
                context = self.rag.rag(query, parser['sessionid'],application_name)
                
                # Get previous conversation
                memory = Memory(application_name)
                prev_conversation = memory.search_conversations(query=query, k=3)
                if parser["context"] != "":
                    report_context_1 = googleserv.GoogleLLM().getResults(parser["context"])
                    report_context = "Report Pdf:" + report_context_1
                else:
                    report_context = ""
                # report_context = parser["context"]
                print("prev conversation: ",prev_conversation, report_context, parser["output_expectation"])

                prompt = f"""
                You are an intelligent assistant designed to answer questions based strictly on the provided context. Use only the information available in the Context and Previous Conversation to formulate your answer. Do not use any external knowledge or assumptions.

                Answer the given Question using the most appropriate format based on the Output Expectation.

                If a Report Context is provided, consider it as a reference document or supplementary data that may assist in forming your response.

                Instructions:

                Focus strictly on the Context, Previous Conversation and Report Context.

                Ensure the response matches the required Output Expectation exactly.

                Do not provide explanations or additional commentary unless asked.

                Question:
                {query}

                Report Context:
                {report_context}

                Context:
                {context}

                Previous Conversation:
                {prev_conversation}

                Output Expectation:
                {parser["output_expectation"]}

                Format your response strictly as:
                {parser["output_expectation"]}

                """

                # Generate prompt
                # prompt = f"""Provide the answer for the given question based on the context only.
                #     Question: {query}
                #     {report_context}
                #     Context: {context}
                #     Previous Conversation: {prev_conversation}
                #     Format of the answer should be: {parser["output_expectation"]}
                    
                #     Answer: """
                #Output Format: {parser['output_expectation']}
                # Generate answer
                answer = str(self.model_provider.generate(prompt))
                print("answer: ",answer)
                # Validate answer
                if answer:
                    validation_result = validate_answer(query, answer, context, parser['output_expectation'])
                    print(f"Validation Result: {validation_result}")

                    if validation_result.get("is_correct"):
                        print("Answer is correct. Returning the answer.")
                        return answer
                    
                    # Rephrase query if invalid
                    reason = validation_result.get("reason", "No reason provided.")
                    print(f"Answer is incorrect. Reason: {reason}")
                    rephrased_result = rephrase_query(query, reason, answer=answer)
                    query = rephrased_result["rephrased_query"]
                    print(f"Rephrased Query: {query}")

            except Exception as e:
                print(f"Error during processing: {str(e)}")

            counter += 1

        print("No valid answer found after all attempts.")
        return "I don't have an answer."