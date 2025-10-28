from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse import Evaluation
from langfuse.experiment import ExperimentItem

from main import rag_bot

load_dotenv()
langfuse = get_client()


def rag_task(*, item: ExperimentItem, **kwargs):
  """Task function that runs the full RAG pipeline."""
  question = item.input["question"]  # type: ignore
  result = rag_bot(question)
  
  return {
    "answer": result["answer"],
    "documents": result["documents"]
  }


# Answer Relevance Evaluation
class AnswerRelevanceGrade(TypedDict):
  explanation: Annotated[str, ..., "Explain your reasoning for the score"]
  score: Annotated[int, ..., "Rate the relevance of the answer to the question from 1 to 10"]


answer_relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
  AnswerRelevanceGrade, method="json_schema", strict=True
)

answer_relevance_instructions = """You are evaluating the relevance of an answer to a question.
You will be given a QUESTION, an ANSWER, and an EXPECTED OUTPUT.

Here is the grade criteria to follow:
(1) The ANSWER should directly address the QUESTION
(2) The ANSWER should be similar in content and scope to the EXPECTED OUTPUT
(3) The ANSWER should not contain significant irrelevant information
(4) It's acceptable if the ANSWER provides additional helpful context as long as it addresses the core question

You should return a score from 1 to 10, where:
- 1-3: The answer is mostly irrelevant or doesn't address the question
- 4-6: The answer partially addresses the question but misses key points
- 7-8: The answer addresses the question well with minor gaps
- 9-10: The answer fully and accurately addresses the question
"""


def answer_relevance_evaluator(*, input, output, expected_output, metadata, **kwargs):
  """Evaluates how relevant the generated answer is to the question."""
  result = answer_relevance_llm.invoke(
    answer_relevance_instructions
    + "\n\nQUESTION: "
    + input["question"]
    + "\n\nANSWER: "
    + output["answer"]
    + "\n\nEXPECTED OUTPUT: "
    + expected_output["answer"]
  )

  return Evaluation(
    name="answer_relevance",
    value=result["score"],
    comment=result.get("explanation", "")
  )


# Faithfulness Evaluation
class FaithfulnessGrade(TypedDict):
  explanation: Annotated[str, ..., "Explain your reasoning for the score"]
  score: Annotated[int, ..., "Rate the faithfulness of the answer to the source documents from 1 to 10"]


faithfulness_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
  FaithfulnessGrade, method="json_schema", strict=True
)

faithfulness_instructions = """You are evaluating the faithfulness of an answer to the source documents.
You will be given an ANSWER and the DOCUMENTS that were used to generate it.

Here is the grade criteria to follow:
(1) The ANSWER should only contain information that can be verified from the DOCUMENTS
(2) The ANSWER should not hallucinate or make up facts not present in the DOCUMENTS
(3) The ANSWER should not contradict information in the DOCUMENTS
(4) It's acceptable for the ANSWER to say "I don't know" if the DOCUMENTS don't contain the information

You should return a score from 1 to 10, where:
- 1-3: The answer contains significant hallucinations or contradictions
- 4-6: The answer is partially grounded but contains some unverified claims
- 7-8: The answer is mostly faithful with minor unsupported details
- 9-10: The answer is fully grounded in the source documents
"""


def faithfulness_evaluator(*, input, output, expected_output, metadata, **kwargs):
  """Evaluates how faithful the generated answer is to the source documents."""
  result = faithfulness_llm.invoke(
    faithfulness_instructions
    + "\n\nANSWER: "
    + output["answer"]
    + "\n\nDOCUMENTS: "
    + "\n\n".join(doc.page_content for doc in output["documents"])
  )

  return Evaluation(
    name="faithfulness",
    value=result["score"],
    comment=result.get("explanation", "")
  )


if __name__ == "__main__":
  print("Fetching dataset")
  dataset = langfuse.get_dataset(name="rag_bot_evals")

  print("Running answer evaluation experiment")
  dataset.run_experiment(
    name="Answer Quality: Relevance and Faithfulness",
    task=rag_task,
    evaluators=[answer_relevance_evaluator, faithfulness_evaluator],
  )

  print("Experiment run successfully")
  langfuse.flush()
