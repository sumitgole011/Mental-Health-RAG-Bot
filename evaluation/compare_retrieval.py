"""Run paired old MMR, old similarity and candidate hybrid on frozen questions."""
import sys,json,pathlib,hashlib
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from retrieval import HybridRetriever,load_chunks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu'})
old=[]
for path in sorted((root/'Doc').glob('*.txt')):
 old.extend(RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True).create_documents([path.read_text()],metadatas=[{'source':path.name}]))
store=Chroma.from_documents(old,emb,collection_name='paired_retrieval_comparison')
candidate=HybridRetriever(load_chunks(root/'Doc'),emb)
methods={'old_mmr':store.as_retriever(search_type='mmr',search_kwargs={'k':2}), 'old_similarity':store.as_retriever(search_type='similarity',search_kwargs={'k':2}),'hybrid':candidate}
out={'source_commit':'87e6b10aee23120dd3088dbd9d78ddc9751c859f','scope':'Retrieval only; explicit UTF-8 loader adaptation; anchor containment proxy, not answer accuracy.','sets':{}}
for file in ['questions.json','fresh_questions.json']:
 qs=json.loads((root/'evaluation'/file).read_text());rs={}
 for name,method in methods.items():
  rows=[]
  for q in qs:
   docs=method.invoke(q['question']);hits=[];retrieved=[]
   for i,d in enumerate(docs,1):
    a=d.metadata['start_index'];b=a+len(d.page_content)
    if d.metadata['source']==q['source'] and a<=q['evidence_start'] and b>=q['evidence_end']:hits.append(i)
    retrieved.append({'source':d.metadata['source'],'start':a,'end':b})
   rows.append({'id':q['id'],'hit_at_2':bool(hits),'reciprocal_rank':1/min(hits) if hits else 0,'retrieved':retrieved})
  rs[name]={'hits':sum(x['hit_at_2'] for x in rows),'total':len(rows),'hit_rate':sum(x['hit_at_2'] for x in rows)/len(rows),'mrr_at_2':sum(x['reciprocal_rank'] for x in rows)/len(rows),'results':rows}
 out['sets'][file]={'sha256':hashlib.sha256((root/'evaluation'/file).read_bytes()).hexdigest(),'methods':rs}
(root/'evaluation/comparison_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps({s:{k:{a:b for a,b in v.items() if a!='results'} for k,v in r['methods'].items()} for s,r in out['sets'].items()},indent=2))
