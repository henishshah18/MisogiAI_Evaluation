from langchain_text_splitters import RecursiveCharacterTextSplitter
# import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain import hub
from langchain_chroma import Chroma

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

with open("rag_test.txt") as f:
    rag_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    separators=[
        "\n\n",
        "\n"]
)
texts = text_splitter.create_documents([rag_text])

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key = OPENAI_API_KEY)

vector_store = Chroma("test_rag", embeddings)

# index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )

vector_store.add_documents(documents=texts)



llm = OpenAI(model="gpt-4o-mini",
    temperature=0,
    max_retries=2,
    api_key=OPENAI_API_KEY)

prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: str
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# response = graph.invoke({"question": "What are Partner plank band row?"})
# print(response["answer"])