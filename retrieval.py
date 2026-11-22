"""Deterministic text ingestion and hybrid retrieval shared by app and evaluation."""
from pathlib import Path
import re
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer


def load_chunks(directory):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks = []
    for path in sorted(Path(directory).glob('*.txt')):
        text = path.read_text(encoding='utf-8')
        for doc in splitter.create_documents([text], metadatas=[{'source': path.name}]):
            # Drop page labels and fragments that cannot provide meaningful context.
            if len(re.findall(r'[A-Za-z]', doc.page_content)) < 100:
                continue
            doc.metadata['end_index'] = doc.metadata['start_index'] + len(doc.page_content)
            chunks.append(doc)
    if not chunks:
        raise ValueError('No usable UTF-8 knowledge documents found.')
    return chunks


def search_query(query):
    query = re.sub(r'(?i)\b(according to (the )?(booklet|book|guide)|in the (booklet|book|guide)|does the booklet|the booklet|LEAP quick-start guide)\b', ' ', query)
    return re.sub(r'\s+', ' ', query).strip()


class HybridRetriever:
    """Fuse lexical and normalized semantic rankings, with bounded top-k output.

    No retrieval score is an answer-confidence or clinical-quality probability.
    """
    def __init__(self, documents, embeddings, k=2):
        if k < 1:
            raise ValueError('k must be positive')
        self.documents = documents
        self.embeddings = embeddings
        self.k = k
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.lexical = self.vectorizer.fit_transform(d.page_content for d in documents)
        self.vectors = np.asarray(embeddings.embed_documents([d.page_content for d in documents]), dtype=np.float32)
        self.vectors /= np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)

    def invoke(self, query):
        if not isinstance(query, str) or not query.strip():
            return []
        query = search_query(query)
        if not query:
            return []
        lexical = (self.lexical @ self.vectorizer.transform([query]).T).toarray().ravel()
        vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        semantic = self.vectors @ vector
        # Reciprocal rank fusion avoids comparing incompatible raw score scales.
        fused = np.zeros(len(self.documents))
        for scores in (lexical, semantic):
            order = np.argsort(-scores, kind='stable')[:20]
            for rank, index in enumerate(order, 1):
                if scores[index] > 0:
                    fused[index] += 1 / (60 + rank)
        order = np.argsort(-fused, kind='stable')[:self.k]
        return [self.documents[int(i)] for i in order if fused[i] > 0]
