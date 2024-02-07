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

'''
llama_index를 활용한 RAG pipeline
'''
# openAI API key
os.environ['OPENAI_API_KEY'] = 'YOUR OPENAI API KEY'

# data load
documnets = SimpleDirectoryReader("./data/paul_graham/").load_data()

# openAI model 선정
llm = OpenAI(model="gpt-4")

# node index를 chunk 단위로 parsing
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documnets)

# vectorstore의 vector index를 정의
vector_index = VectorStoreIndex(nodes)

# QueryEngine을 빌드하고 Query를 시작
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("What did the author do growing up?")

print(response_vector.response)

# 검색 및 응답 평가의 RAG 시스템 평가에 사용할 수 있는 질문 및 컨텍스트 쌍 형성(generate_question_context_pairs를 사용하면 쉽게 형성 가능)
qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=2
)

'''
Retriever 평가
적중률 : 검색된 상위 k개 문서 내에서 정답이 있는 쿼리의 비율을 계산. 시스템이 상위 몇 가지 추측 내에서 얼마나 자주 맞는지를 판단.
MRR : 각 쿼리에 대해 MRR은 가장 높은 위치에 있는 관련 문서의 순위를 확인하고 시스템의 정확도를 평가.
모든 쿼리에서 순위의 역수의 평균임. 따라서 첫 번째 관련 문서가 최상위 결과인 경우 역수 순위는 1이고, 두 번째는 역수 순위가 1/2 등임.
'''
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


'''
Response 평가
FaithfulnessEvalutor : 쿼리 엔진의 응답이 원본 노드와 일치하는지 여부 측정, 응답이 환각인지 여부를 측정하는데 유용함.
'''
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

'''
Relevance 평가
RelevancyEvaluator : 응답 및 원본 노드(검색된 context)가 쿼리와 일치하는지 측정하는데 유용함.
응답이 실제로 쿼리에 응답하는 것인지 확인하는데 유용함.
'''
from llama_index.evaluation import RelevancyEvaluator
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)

query = queries[10]
response_vector = query_engine.query(query)
eval_result = relevancy_gpt4.evaluate_response(
    query=query, response=response_vector
)

print(eval_result.passing)       # 평가 pass 시 True
print(eval_result.feedback)

'''
배치 평가
BatchEvalRunner 사용
'''
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

faithfulness_score = sum(result.passing for result in evaluate_queries['faithfulness']) / len(evaluate_queries['faithfulness'])
print(faithfulness_score)

relevancy_score = sum(result.passing for result in evaluate_queries['relevancy']) / len(evaluate_queries['relevancy'])
print(relevancy_score)