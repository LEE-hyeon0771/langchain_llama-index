## llama-index
- 데이터를 수집, 구조화 및 액세스 하는데 초점을 맞추고 있는 프레임워크
- 이전에는 chunk 단위로 parsing 하는 작업을 어떻게 parsing 하는 것이 효율적인 것일지를 고민했다면 llama-index는 parsing 없이도 손쉽게 vector store에 저장시켜주는 기능을 가지고 있다.
- 인덱스 쿼리에 최적화되어 있어서, 데이터를 수집하고 벡터화시키고 쉽게 쿼리단계까지 진행시킬 수 있다.
- 쿼리 엔진이 작동하게 되면 벡터화된 데이터에서 가장 관련성이 높은 정보와 비교하고 LLM에 context로 적용시킬 수 있어서 외부 데이터를 가지고 오고, 그 데이터를 이용해서 학습시키지 않은 정보에 대한 response를 적절하게 제공할 수 있는 RAG 기법에 최적화된 프레임워크이다.



### index 유형
1. List index
- 리스트처럼 노드를 순차적으로 정렬시킨다.
- 입력 데이터를 노드로 chunking 한 후 선형 방식으로 정렬되어 순차적 or 키워드나 임베딩을 통해 쿼리할 수 있다.
- 순차적 쿼리가 필요할 때 유용하고, LLM의 토큰 제한을 초과하더라도 각 노드에서 텍스트를 스마트하게 쿼리하고 목록을 탐색하면서 답변을 구체화할 수 있다.

2. Vecotr store index
- 노드는 로컬 또는 특수 벡터 데이터베이스에 저장된 벡터 임베딩으로 변환된다.
- 쿼리를 받으면 top_k 노드를 가져와서 전달한다.
- 벡터 검색을 통한 의미 유사성에 의존하는 텍스트 비교 작업에서 유리하다.

3. Tree index
- 긴 텍스트 chunk를 쿼리하는데 효율적이고 다양한 텍스트 segment에서 정보를 추출하는데 유리하다.

4. Keyword indexing
- 쿼리를 수행하면 쿼리에서 키워드가 추출되고 매핑된 노드에 포커싱된다.
- 명확한 사용자 쿼리(집중해야 할 키워드)가 있다면 키워드 indexing을 사용하는 것이 유리하다.



## RAG 구현 활용

### Llama index의 기본 RAG pipline
1. API key
2. 데이터 문서 다운로드
3. 데이터 문서 load
4. model 선정
5. chunk 단위로 parsing(꼭 필요한 작업은 아님)
6. vector store의 vecotr index 정의(VectorStoreIndex)
7. QueryEngine 빌드 및 Query 시작

```
import nest_asyncio
nest_asyncio.apply()
import asyncio
from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.evaluation import RetrieverEvaluator
from llama_index.llms import OpenAI

import os
import pandas as pd

# openAI API key
os.environ['OPENAI_API_KEY'] = 'YOUR OPENAI API KEY'

# data load
documnets = SimpleDirectoryReader("./data/paul_graham/").load_data()

# openAI model 선정
llm = OpenAI(model="gpt-4")
```

- os.environ['OPENAI_API_KEY']를 이용해서 API key 받아오기
- SimpleDirectoryReader() : 데이터 로드
- OpenAI() : openAI model을 선정


```
# node index를 chunk 단위로 parsing
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documnets)

# vectorstore의 vector index를 정의
vector_index = VectorStoreIndex(nodes)
```

- SimpleNodeParser : node index를 chunk 단위로 parsing 한다.
- VectorStoreIndex() : vectorstore의 vector index를 임베딩 과정 없이 손쉽게 정의한다.


```
# QueryEngine을 빌드하고 Query를 시작
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("What did the author do growing up?")

print(response_vector.response)
```

- QueryEngine을 이용해서 빌드하고 Query를 시작한다.

```
qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=2
)
```

- generate_question_context_pairs : 검색 및 응답 평가의 RAG 시스템 평가에 사용할 수 있는 질문 및 컨텍스트 쌍 형성


## llama-index의 평가 방식

기본적인 RAG 구현 pipeline은 위 code를 baseline으로 잡고 구현하면 된다. 지금부터는 다양한 llama-index의 평가방식을 알아보려고 한다.

### Retriever 평가

적중률 : 검색된 상위 k개 문서 내에서 정답이 있는 쿼리의 비율을 계산. 시스템이 상위 몇 가지 추측 내에서 얼마나 자주 맞는지를 판단.
MRR : 각 쿼리에 대해 MRR은 가장 높은 위치에 있는 관련 문서의 순위를 확인하고 시스템의 정확도를 평가.
모든 쿼리에서 순위의 역수의 평균임. 따라서 첫 번째 관련 문서가 최상위 결과인 경우 역수 순위는 1이고, 두 번째는 역수 순위가 1/2 등임.

```
async def evaluate_retriever(dataset):
    retriever = vector_index.as_retriever(similarity_top_k=2)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(dataset)
    return eval_results

loop = asyncio.get_event_loop()
results = loop.run_until_complete(evaluate_retriever(qa_dataset))
print(results)

# 평가 결과를 테이블 형식으로 표현
def display_results(name, eval_results):
    
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
        
    full_df = pd.DataFrame(metric_dicts)
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    
    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rage": [hit_rate], "MRR": [mrr]}
    )
    return metric_df

display_results("OpenAI Embedding Retriever", results)
```

### Response 평가

FaithfulnessEvalutor : 쿼리 엔진의 응답이 원본 노드와 일치하는지 여부 측정, 응답이 환각인지 여부를 측정하는데 유용함.

```
queries = list(qa_dataset.queries.values())

gpt35 = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_gpt35 = ServiceContext.from_defaults(llm=gpt35)
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

# service context를 만들어서 query에 대한 응답을 생성하고 쿼리엔진을 실행
vector_index = VectorStoreIndex(nodes, service_context = service_context_gpt35)
query_engine = vector_index.as_query_engine()

# gpt4로 FaithfulnessEvaluator 평가를 진행
from llama_index.evaluation import FaithfulnessEvaluator
faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)

response_vector = query_engine.query(queries[0])
eval_result = faithfulness_gpt4.evaluate_response(response=response_vector)

print(eval_result.passing)
```

### Relevance 평가

RelevancyEvaluator : 응답 및 원본 노드(검색된 context)가 쿼리와 일치하는지 측정하는데 유용함. 응답이 실제로 쿼리에 응답하는 것인지 확인.

```
from llama_index.evaluation import RelevancyEvaluator
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)

query = queries[10]
response_vector = query_engine.query(query)
eval_result = relevancy_gpt4.evaluate_response(
    query=query, response=response_vector
)

print(eval_result.passing)       # 평가 pass 시 True
print(eval_result.feedback)
```

### Batch 평가

BatchEvalRunner 사용해서

```
from llama_index.evaluation import BatchEvalRunner
batch_eval_queries = queries[:10]

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=0,
)

async def evaluate_queries(queries):
    eval_results = await runner.aevaluate_queries(
        query_engine, queries=queries
    ) 
    return eval_results

loop = asyncio.get_event_loop()
batch_results = loop.run_until_complete(evaluate_queries(batch_eval_queries))
print(batch_results)
```


