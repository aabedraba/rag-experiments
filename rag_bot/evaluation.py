from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse import Evaluation

from main import RagBotResponse, rag_bot

load_dotenv()
langfuse = get_client()


def rag_bot_task(*, item, **kwargs):
  question = item.input["question"]
  response = rag_bot(question)

  return response


class RetrievalRelevanceGrade(TypedDict):
  explanation: Annotated[str, ..., "Explain your reasoning for the score"]
  relevant: Annotated[
    list[int, int, int],
    ...,
    "Three chunks are provided, please rate the relevance of each chunk to the question from 1 to 10",
  ]


retrieval_relevance_llm = ChatOpenAI(model="gpt-5", temperature=0).with_structured_output(
  RetrievalRelevanceGrade, method="json_schema", strict=True
)

retrieval_relevance_instructions = """You are evaluating the relevance of a set of chunks to a question. 
You will be given a QUESTION, an EXPECTED OUTPUT, the ANSWER and a set of three DOCUMENTS provided with the answer.

Here is the grade criteria to follow:
(1) You goal is to identify DOCUMENTS that are completely unrelated to the QUESTION
(2) It is OK if the facts have SOME information that is unrelated as long as itis close to the EXPECTED OUTPUT

You should return a list of three numbers, one for each chunk, indicating the relevance of the chunk to the question.
"""


# Define evaluation functions
def relevant_chunks_evaluator(*, input, output: RagBotResponse, expected_output, metadata, **kwargs):
  retrieval_relevance_result = retrieval_relevance_llm.invoke(
    retrieval_relevance_instructions
    + "\n\nQUESTION: "
    + input["question"]
    + "\n\nEXPECTED OUTPUT: "
    + expected_output["answer"]
    + "\n\nANSWER: "
    + output["answer"]
    + "\n\nDOCUMENTS: "
    + "\n\n".join(doc.page_content for doc in output["documents"])
  )

  return Evaluation(
    name="retrieval_relevance",
    value=sum(retrieval_relevance_result["relevant"]) / len(retrieval_relevance_result["relevant"]),
  )

print("Fetching dataset")
dataset = langfuse.get_dataset(name="rag_bot_evals")
print("Running experiments...")
dataset.run_experiment(name="Multi-metric Evaluation", task=rag_bot_task, evaluators=[relevant_chunks_evaluator])

print("Experiment run successfully")
langfuse.flush()
