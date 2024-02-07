## langchain

- langchain은 언어 모델을 이용한 개발을 촉진하고 간소화하는 목적으로 만들어진 프레임워크
- 언어 모델과 상호작용하는 애플리케이션을 빠르고 쉽게 구축할 수 있도록 설계되었다.
- 주요목적 : 언어 모델을 활용하여 다양한 도메인과 상황에서의 문제를 해결할 수 있도록 지원하는 것


### 주요기능

1. 언어 모델 통합 : 여러 언어 모델을 쉽게 통합하고 사용할 수 있도록 해서 다양한 소스의 언어모델을 쉽게 실험하고 최적의 결과를 낼 수 있다.
2. toolchain & 유틸리티 : 텍스트 생성, 요약, 질문 답변 등과 같은 언어 모델을 활용한 작업 도구와 유틸리티가 제공된다.
3. 확장성 : template과 chain을 이용해 필요에 따라 자신만의 기능을 추가하거나 기존 기능을 수정해서 사용할 수 있어 확장성이 좋다.

하지만, 이러한 프레임워크도 전문적인 지식이 필요한 task에 대해서 유의미한 답변 제공이 힘들 수 있다. 
이러한 한계를 극복하기 위해 RAG 기법을 사용하게 되는데, 이 과정에서 관련 문서를 chunk 단위로 쪼개거나, embedding, vector화, 쿼리 등을 사용하며 질문에 대해 유사한 chunk를 검색해서 활용하는 방식을 선택한다.


### langchain module

1. Model I/O
- 언어 모델과의 직접적인 input과 output을 관리한다.
- 특정 언어 모델에 요청을 보내고, 모델의 예측 또는 응답을 받아 처리하는 역할을 한다.
- 언어 모델을 선택, 모델에 전송할 텍스트 정의, 모델의 응답을 애플리케이션 내에서 사용할 수 있도록 한다.

2. Data Connection
- 애플리케이션에 필요한 특정 데이터 소스와의 연결을 관리한다.
- 외부 API, 데이터베이스, 파일시스템 등의 다양한 데이터 소스를 포함한다.
- 모듈을 통해 애플리케이션에 필요한 데이터를 검색, 저장 또는 업데이트 할 수 있다.

3. Chains-Call 시퀀스 구성
- 여러 단계 또는 작업을 연속적으로 실행하는 로직을 정의한다.
- 복잡한 작업을 단순화하고, 재사용 가능한 프로세스 chain을 만드는 데 유용하다.
- 특정 작업을 위해 필요한 단계들을 chain으로 구성하여 chain 실행을 통해 작업을 자동화할 수 있다.

4. Agents
- chain이 high-level 지침에 따라 사용할 도구를 선택하도록 한다.

5. Memory
- 애플리케이션의 상태 또는 컨텍스트를 저장하고, chain 실행 사이에 정보를 유지한다.
- 메모리 모듈을 통해서 사용자 세션, 대화기록, 사용자 선호도 등의 정보를 저장하고 접근할 수 있다.

6. Callbacks
- chain의 중간 단계를 기록하거나 스트리밍 하는 역할을 한다.
- 중간 단계에서 로그를 기록하거나, 조건부 로직을 실행시킬 수 있다.

7. UI/UX Components
- 사용자의 입력을 받아서 처리하고, 결과를 사용자에게 표시하는데 사용한다.
- template 기능을 사용하여 사용자가 원하는 형태의 작업을 할 수 있다.

8. Security & Authentication
- 보안 및 인증 모듈, 사용자의 신원확인, 데이터 접근을 제어한다.

9. Intergration Tools
- 외부 서비스나 애플리케이션과의 통합을 용이하게 하는 도구로 API연동, 웹훅 설정을 지원한다.


### langchain 활용
1. ChatPromptTemplate
```
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
```

Prompt의 Template을 제시해서, 이 형식에 맞게 ChatGPT가 답변할 수 있도록 format을 만들어 줄 수 있다.

2. ChatOpenAI
```
from langchain.chat_models import ChatOpenAI

api_key = "..."

llm = ChatOpenAI(openai_api_key=api_key, temperatrue=0.1)
llm.predict("instruction~~~")
```

이 기능을 통해 openAI API를 효율적으로 가져올 수 있다.

3. OutputParser
```
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

이 기능을 통해 chain 내에서 결과물을 효율적으로 parsing 할 수 있다.



### langchain을 이용한 RAG 구현

RAG는 전문적인 도메인과 아직 학습하지 않은 데이터에 대해 LLM이 가지는 hallucination을 막기 위해 외부에서 검색한 정보를 활용해서 정확하고 사실적인 응답을 생성할 수 있도록 LLM의 능력을 보완하고 강화하는 기법이다.



1. Local 문서 기반 RAG 구현
```
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

llm = ChatOpenAI(temperatrue=0.1)

cache_dir = LocalFileStore("로컬 문서 경로")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator='\n',         
    chunk_size=600,        
    chunk_overlap=100,      
)

loader = UnstructuredFileLoader("txt 문서 경로")

docs = loader.load_and_split(text_splitter=splitter)
```

ChatOpenAI(), LocalFileStore(), CharaterTextSplitter(), UnstructuredFileLoader() 기능을 사용했다.

- LocalFileStore() : 로컬에 저장된 파일 불러오기.
- CharacterTextSplitter() : 사용자가 지정한 문자를 기준으로 문서를 분할한다.
- UnstructuredFileLoader() : txt, pdf 등과 같은 구조화되지 않은 파일 로드, 텍스트 추출이나 파일 내용을 분석하기 위한 전처리 단계에서 많이 사용.
- load_and_split() : 파일 로딩과 동시에 분할 진행. splitter 파라미터로 전달하고, 분할된 문서를 반환한다.

이 단계까지가 RAG 구현시에 모델 - 문서 load - chunk 단위 split 까지이다. 문서 제공시 문서를 분할해서 필요한 부분만 전달하게 되면, 정확도 감소와 비용 소모를 줄일 수 있다. 하지만, 문맥의 의미를 손상시키지 않는 선에서 적절한 크기로 분할하는 것이 중요하다.



2. 웹 기반 문서 RAG 구현
```
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("웹 문서 경로",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

WebBaseLoader(), RecursiveCharacterTextSplitter() 기능을 사용했다.

- WebBaseLoader : Web에서 문서 경로를 받아오는 역할
- RecursiveCharacterTextSplitter : 공통적인 구분 기호를 사용하여 문서를 재귀적으로 분할한다.

두 작업은 초반부에 문서를 load 하는 과정과 split 하는 과정에서 차이가 있었다. 하지만, 결국 문서를 load하고 splitter를 사용하여 문서를 분할하고 나면, 그 다음 작업은 동일하다.



3. 동일작업
- embedding을 통해 vectorstore에 저장(이때, cache Memory를 사용하여 효율적으로 임베딩 처리를 할 수도 있다.)
```
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embeddings)
retriver = vectorstore.as_retriever()
```

OpenAIEmbeddings(), CacheBackedEmbeddings(), vectorstore에서 retriever 실시하는 기능을 사용했다.

- OpenAIEmbeddings()를 이용한 Embedding : text에 적절한 점수를 의미별로 부여하는 방식이고, 자연어를 vector로 변환하는 작업
- Embedding 된 문서는 vectorstore에 저장됨.
- Retriever에서 쿼리와 연관성이 높은 문서들을 vectorstore 로부터 찾아오고, 문서를 LLM에 전달할 프롬프트에 포함시켜서 정확도가 높은 답변을 기대함.
- CacheBackedEmbeddings() : Embedding 객체와 캐시가 저장되어 있는 위치를 파라미터로 전달. embedding 객체가 호출될 일이 있으면, cached_embeddings를 사용.
- 이미 캐시되어 있다면 저장된 캐시를 사용, 그렇지 않다면 embedding을 진행하여 캐시를 생성함.


```
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

chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),         # 사용자의 질문이 {question}에 그대로 들어가게 됨.
    }
    | prompt
    | llm
)
```

vectorstore 작업이 끝나면, prompt의 template, chain 형성 작업을 실시한다.

```
result = chain.invoke("김첨지는 학생을 어디로 데려다 주었나?")
print(result)
```

invoke() 함수나, predict() 함수를 통해서 쿼리를 전달한다. 전달된 쿼리를 retriever에 전달하고, 변환된 문서를 template의 {context}에 담아준다.
