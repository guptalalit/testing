import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy
load_dotenv()
# Load environment variables
def validate_answer(
    question: str,
    answer: str,
    context: List[str],
    output: str,
    threshold1: float = 0.4,
    threshold2: float = 0.8
) -> Dict[str, Any]:
    """
    Validate a single question-answer pair using RAGAS metrics.

    Args:
        question (str): The question to validate.
        answer (str): The answer to validate.
        context (List[str]): The context chunks for the question.
        threshold (float): Threshold for passing validation (default: 0.7).

    Returns:
        Dict[str, Any]: A dictionary containing the validation result with keys:
                        - is_correct (bool): Whether the answer is correct.
                        - reason (str): The reason for the validation result.
                        - faithfulness_score (float): The faithfulness score.
                        - answer_relevancy_score (float): The answer relevancy score.
    """

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set it in the .env file.")

    # Initialize OpenAI components
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)  # Use a valid OpenAI model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Wrap LLM and embeddings for Ragas
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)


    # Prepare dataset for RAGAS
    print("RAGAS, context: ", context, "\noutput: ", output, "answer: ", answer)
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [context],
    }
    # print("data: ", data)

    dataset = Dataset.from_dict(data)

    # Define evaluation metrics
    metrics = [Faithfulness(), AnswerRelevancy()]

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # Extract scores
    faithfulness_score = result["faithfulness"][0]
    answer_relevancy_score = result["answer_relevancy"][0]

    # Determine if the answer is correct
    is_correct = (
        faithfulness_score >= threshold1 or answer_relevancy_score >= threshold2
    )

    # Generate reason for validation
    reasons = []
    if faithfulness_score < threshold1:
        reasons.append(f"The answer is not faithful to the context (score: {faithfulness_score:.2f}).")
    if answer_relevancy_score < threshold2:
        reasons.append(f"The answer is not relevant to the question (score: {answer_relevancy_score:.2f}).")
    reason = " ".join(reasons) if reasons else "The answer is correct and relevant."

    # Return the validation result
    return {
        "is_correct": is_correct,
        "reason": reason,
        "faithfulness_score": faithfulness_score,
        "answer_relevancy_score": answer_relevancy_score
    }