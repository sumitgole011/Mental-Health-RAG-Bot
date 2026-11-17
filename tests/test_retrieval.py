import tempfile,unittest
from pathlib import Path
import numpy as np
from retrieval import load_chunks,HybridRetriever

class TestRetrieval(unittest.TestCase):
 def test_ingestion_preserves_offsets_and_excludes_page_fragments(self):
  with tempfile.TemporaryDirectory() as d:
   text='12\n\n'+('Relevant evidence about a particular subject. '*12)
   Path(d,'source.txt').write_text(text)
   docs=load_chunks(d)
   self.assertTrue(docs)
   for doc in docs:
    self.assertEqual(text[doc.metadata['start_index']:doc.metadata['end_index']],doc.page_content)
    self.assertGreater(len(doc.page_content),100)
 def test_empty_corpus_fails_clearly(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):load_chunks(d)
 def test_hybrid_ranks_matching_document_and_handles_empty_query(self):
  from langchain_core.documents import Document
  class Embeddings:
   def embed_documents(self,texts):return [[1.,0.],[0.,1.]]
   def embed_query(self,text):return [0.,1.]
  docs=[Document(page_content='apples orchards fruit harvest'),Document(page_content='breathing relaxation exercise calm')]
  retriever=HybridRetriever(docs,Embeddings(),k=1)
  self.assertEqual(retriever.invoke('breathing exercise')[0],docs[1])
  self.assertEqual(retriever.invoke(''),[])
