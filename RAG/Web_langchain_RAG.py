import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# RecursieCharaterTextSplitter : 공통적인 구분 기호를 사용하여 문서를 재귀적으로 분할함.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# vectorstore.as_retriever : 벡터 저장소의 유사성 검색 기능을 사용해서 검색을 용이하게 함.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")      # LangChain prompt hub에 체크인된 RAG 프롬프트 사용
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# chain을 정의하여 구성 요소를 연결하고 투명한 방식으로 rag-chain을 자동으로 추적함.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is Task Decomposition?")
print(result)