from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who speaks like a pirate",
        ),
        ("human", "{text}"),
    ]
)
_model = ChatOpenAI()

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model



### TODO: update pyproject.toml


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
#from langchain.vectorstores import PathwayVectorStore

# If you have a running Pathway Vectorstore instance you can connect to it via client. If not, you can run Vectorstore as follows.

# Example for document loading (from local folders), splitting, and creating vectorstore with Pathway

"""
import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# Load
# Track data in two locations
# See pathway documentation for more options including Google Drive, s3, Dropbox, etc.
docs1 = pw.io.fs.read('./sample_documents/doc1.txt', format='binary', mode='streaming_with_deletions', with_metadata=True)
docs2 = pw.io.fs.read('./sample_documents/doc2.docx', format='binary', mode='streaming_with_deletions', with_metadata=True)
# Split
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 10,
    chunk_overlap  = 5,
    length_function = len,
    is_separator_regex = False,
)
# Embed
embeddings_model = OpenAIEmbeddings(openai_api_key="sk-...")
# Launch VectorDB
vector_server = pw_vs.PathwayVectorServer(
    host="127.0.0.1", port="8765",
    embedder=embeddings_model,
    splitter=text_splitter
)
vector_server.run_server(docs1, docs2, threaded=True)
# Initalize client
client = pw_vs.PathwayVectorClient(
    host="127.0.0.1", port="8765",
)
retriever = client.as_retriever()
"""


from experimental.janek.langchain.vectorstores import pathway as pw_vs

client = pw_vs.PathwayVectorClient(
    host="127.0.0.1", port="8765",
)

retriever = client.as_retriever()


# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI()
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)