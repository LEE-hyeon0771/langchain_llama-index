# langchain vs llama-index

두 프레임워크 모두 전체적인 구조는 유사하다. 하지만 적절하게 사용환경에 따라 선택하는 것이 중요하다.

langchain에서 query 부분이 발전되고, vectorstore를 만드는 방식이 간편하게 발전된 것이 llama-index인데, 다양한 자료들과 예시들, chain을 통한 확장성 때문에 langchain의 범용성이 아직 더 우세한 편이다.

## model API-key

### langchain
```
from langchain.chat_models import ChatOpenAI

api_key = "..."

llm = ChatOpenAI(openai_api_key=api_key, temperatrue=0.1)
llm.predict("instruction~~~")
```

### llama-index
```
import os

# openAI API key
os.environ['OPENAI_API_KEY'] = 'YOUR OPENAI API KEY'
```

문법의 차이가 조금 있다. langchain은 ChatOpenAI()를 사용하고, llama-index는 os에서 API-key를 받아온다.


## 전체적인 구성의 차이

모델 설정 - vector store에 저장 - query 질문이라는 전체적인 구성은 유사하다.

### langchain
1. 모델 api 선정

2. document 불러오기(LocalFileStore)

3. 파일 load

4. 사용자가 적절히 chunk 단위로 parsing(CharacterTextSplitter)

5. 임베딩을 통해서 vector store 저장(OpenAIEmbeddings(), CacheBackedEmbeddings, FAISS, vectorstore.as_retriever())

6. 템플릿과 chain 방식을 선정해서 chain을 구성

7. 쿼리 전달(invoke(), predict())


### llama-index
1. api key

2. 데이터 문서 다운로드

3. 데이터 문서 load(SimpleDirectoryReader)

4. model 선정

5. chunk 단위로 parsing(꼭 필요한 작업은 아님)(SimpleNodeParse)

6. vector store의 vector index 정의(VectorStoreIndex)

7. QueryEngine 빌드 및 Query 시작(as_query_engine())


이렇게 langchain의 경우에는 chunk 단위로 적절히 parsing 하는 작업을 통해서 메모리의 효율성을 보장할 수 있지만, llama-index에서는 parsing 없이도 VectorStoreIndex()를 사용해서 vectorstore에 적절히 벡터를 보관함으로써 인덱스의 효율성을 보장하고, 쿼리엔진으로 쿼리 자체에 집중하는 모습을 보인다.
langchain은 llama-index가 인덱스에 초점을 맞추는 것과는 다르게 template과 chain을 사용해서 사용자가 원하는 방식으로 구성을 다르게 만들 수 있는 것에 조금 더 집중하고 있는 모습을 볼 수 있다.



## Vector store 저장 모습
### langchain
```
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator='\n',         # 해당 문자 기준으로 문서 분할
    chunk_size=600,         # 분할된 한 문서의 최대 chunk 크기를 지정
    chunk_overlap=100,      # 문서 분할 시 앞뒤 문서의 100자를 중복으로 추가하여 생성. 문맥상 적절하지 않은 부분에서 문서 분할 문제 해결
)

loader = UnstructuredFileLoader("./files/운수 좋은날.txt")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embeddings)
retriver = vectorstore.as_retriever()
```

### llama-index
```
# node index를 chunk 단위로 parsing
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documnets)

# vectorstore의 vector index를 정의
vector_index = VectorStoreIndex(nodes)
```

langchain의 chunk 단위 split, embedding, vectorstore 저장과 Llama-index의 split, vectorstore 저장 모습을 보면 랭체인에서 라마인덱스로 발전하면서 어느정도로 vectorstore에서 인덱스에 힘을 주었는지를 알 수 있다.


## langchain의 prompt template과 chain 기능

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
ChatPromptTemplate을 사용해서 prompt의 template을 만들고 {context} 부분에 쿼리질문에 응답한 response를 넣어줄 수 있다. 이 기능을 통해서 사용자가 원하는 다양한 포맷을 구성할 수 있다.

### langchain의 chain 방식
1. Stuff 방식 : 데이터 수집, 준비, 초기 처리 단계를 지칭하며, 분석이나 처리를 위해 필요한 입력 데이터를 구성하는 과정으로 관련 문서를 모두 prompt에 넣어서 전달한다.
```
chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),         # 사용자의 질문이 {question}에 그대로 들어가게 됨.
    }
    | prompt
    | llm
)
```

2. Map Reduce : 대규모 데이터 세트를 처리하고 생성하기 위한 프로그래밍 모델이다. Map 단계에서 데이터 세트를 key-value 쌍으로 매핑하고, "Reduce" 단계에서는 매핑된 데이터를 요약, 합계, 필터링하는 연산을 수행한다. 이 방식으로 병렬처리와 분산시스템에서의 데이터 처리가 가능하다.
각 문서를 요약하고, 요약된 문서를 기반으로 최종 요약본을 만들어냄. 문서 요약에서 속도가 느림.
```
def reduce_function(data_points):
    # 여러 데이터 포인트의 합계를 계산
    sum_result = sum(data_points)
    return sum_result

chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),  # 사용자의 질문을 그대로 전달
    }
    | map_function                         # 데이터에 적용할 매핑 함수
    | reduce_function                      # 매핑된 데이터를 집계하거나 요약하는 리듀스 함수
    | llm  
)
```

3. Refine : 문서들을 순회하며 중간 답변을 생성, 이것을 반복하면서 답변을 정제함. 양질의 답변을 만들어 낼 수 있고, 속도와 비용면에서 단점이 있지만, 결과물이 뛰어나다. 노이즈 제거, 데이터 포맷 조정, 누락된 값 처리 등을 포함할 수 있고 최종 데이터가 분석 목적이나 사용자 요구에 부합하도록 만든다.
```
def refine_function(raw_data):
    # 불필요한 문자 제거 및 대문자로 변환
    refined_data = raw_data.replace("\n", " ").strip().upper()
    return refined_data

chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | refine_function                      # LLM의 출력을 정제하거나 변환
)
```

4. Map re-rank : 각 문서에 대해 각 답변을 만들어 내고, 결과 데이터 세트의 재정렬이나 중요도에 따른 재순위 부여를 수행한다. 가장 높은 점수의 답변을 최종 답변으로 설정함. 이 과정은 최종 결과의 정확성이나 관련성을 높이기 위해서 사용된다.
```
def re_rank_function(data_items):
    # 데이터 아이템을 중요도에 따라 재순위화
    ranked_items = sorted(data_items, key=lambda x: x['importance'], reverse=True)
    return ranked_items

chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),
    }
    | map_function  
    | re_rank_function                      # 매핑된 결과를 재순위화
    | llm
)
```

## Query

### langchain
```
result = chain.invoke("김첨지는 학생을 어디로 데려다 주었나?")
```

보는 것과 마찬가지로 invoke() 함수를 사용하거나, predict() 함수를 사용해서 생성되는 response를 template의 {context}에 담아 출력한다.

### llama-index
```
# QueryEngine을 빌드하고 Query를 시작
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("What did the author do growing up?")
```

langchain과는 다르게 llama-index의 경우는 쿼리 엔진을 이용해서 쿼리를 작동시키는 방식을 사용한다.

이렇게 랭체인과 라마인덱스를 비교해보았는데, 전체적인 구성차이는 큰 차이가 없다. 하지만, 서비스의 특성상 집중해야 하는 것이 무엇이냐에 따라 적절하게 프레임워크를 선택해야 할 것이다.

단, 랭체인이 선호되는 이유는 개발자 커뮤니티에서 다양한 예제들이 널려 있다는 것이다. 또한, 프로젝트 초기 단계에서 방향성을 설정하는 데 있어서 유연성과 확장성이 좋아 적절하게 모델을 조작할 수 있다는 것이 엄청난 장점이라고 생각된다.

