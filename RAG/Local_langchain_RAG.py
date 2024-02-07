from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

llm = ChatOpenAI(temperatrue=0.1)

cache_dir = LocalFileStore("./.cache/practice/")

# 문서 제공 시 문서를 분할해서 필요한 부분만 전달하면 정확도 감소와 비용 소모를 줄일 수 있다.
# 문맥의 의미를 손상시키지 않는 선에서 적절한 크기로 분할하는 것이 중요하다.
# CharacterTextSplitter : 사용자가 지정한 문자를 기준으로 문서를 분할한다.
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator='\n',         # 해당 문자 기준으로 문서 분할
    chunk_size=600,         # 분할된 한 문서의 최대 chunk 크기를 지정
    chunk_overlap=100,      # 문서 분할 시 앞뒤 문서의 100자를 중복으로 추가하여 생성. 문맥상 적절하지 않은 부분에서 문서 분할 문제 해결
)
# UnstructedFileLoader는 text files, powerpoints, html, pdfs, images 등 여러 가지 형식의 파일 지원에 편리함
loader = UnstructuredFileLoader("./files/운수 좋은날.txt")
# load_and_split : 파일 로딩과 동시에 분할 진행. splitter 파라미터로 전달하고, 분할된 문서를 반환한다.
docs = loader.load_and_split(text_splitter=splitter)

# Embedding : text에 적절한 점수를 의미별로 부여하는 방식이고, 자연어를 vector로 변환하는 작업
# Embedding 된 문서는 vectorstore에 저장됨.
# Retriever에서 쿼리와 연관성이 높은 문서들을 vectorstore 로부터 찾아오고, 문서를 LLM에 전달할 프롬프트에 포함시켜서 정확도가 높은 답변을 기대함.
embeddings = OpenAIEmbeddings()

# Cache Memory를 사용해서 임베딩을 효율적으로 처리함.
# CacheBackedEmbeddings : Embedding 객체와 캐시가 저장되어 있는 위치를 파라미터로 전달. embedding 객체가 호출될 일이 있으면, cached_embeddings를 사용
# 이미 캐시되어 있다면 저장된 캐시를 사용, 그렇지 않다면 embedding을 진행하여 캐시를 생성함.
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embeddings)
retriver = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. 
            Answer questions using only the following context. 
            If you don't know the answer just say you don't know, don't make it up:
            \n\n
            {context}",
            """
        ),
        ("human", "{question}"),
    ]
)

# Stuff : 관련 문서를 모두 prompt에 채워 넣어 전달
# Map reduce : 각 문서를 요약하고, 요약된 문서를 기반으로 최종 요약본을 만들어냄. 문서 요약에서 속도가 느림
# Refine : 문서들을 순회하며 중간 답변을 생성, 이것을 반복하면서 답변을 정제함. 양질의 답변을 만들어 낼 수 있고, 속도와 비용면에서 단점이 있지만, 결과물이 뛰어남.
# Map re-rank : 각 문서에 대해 각 답변을 만들어 내고, 점수를 부여한다. 가장 높은 점수의 답변을 최종 답변으로 설정함.

# Stuff 방식으로 chain을 구성
chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),         # 사용자의 질문이 {question}에 그대로 들어가게 됨.
    }
    | prompt
    | llm
)

# 전달된 쿼리를 retriever에 전달하고, 반환된 문서를 {context}에 넣어준다.
result = chain.invoke("김첨지는 학생을 어디로 데려다 주었나?")
print(result)