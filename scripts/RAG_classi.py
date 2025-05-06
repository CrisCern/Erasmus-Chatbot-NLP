import pandas as pd
import torch
import warnings
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.utils import ComponentDevice
import rich

warnings.filterwarnings(action="ignore")
pd.set_option("display.float_format", '{:.2f}'.format)

class WhatsAppRAG:
    def __init__(self, csv_path, device="cuda:0"):
        self.csv_path = csv_path
        self.device = device
        self.df = None
        self.documents = None
        self.document_store = None
        self.indexing_pipeline = None
        self.retrieval_pipeline = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df["timestamp"] = pd.to_datetime(self.df["start_time"], format="%Y-%m-%d %H:%M:%S")

    def create_documents(self):
        self.documents = [
            Document(content=row["documento"], meta={"date": row["timestamp"], "sender": row["partecipanti"]})
            for _, row in self.df.iterrows()
        ]

    def build_indexing_pipeline(self):
        self.document_store = InMemoryDocumentStore(
            bm25_tokenization_regex=r"(?u)\\b\\w\\w+\\b",
            bm25_algorithm="BM25L",
            embedding_similarity_function="cosine"
        )

        cleaner = DocumentCleaner(remove_empty_lines=True, remove_extra_whitespaces=True)
        embedder = SentenceTransformersDocumentEmbedder(
            model="BAAI/bge-large-en-v1.5",
            device=ComponentDevice.from_str(self.device),
            batch_size=32,
            progress_bar=True,
            normalize_embeddings=False,
            meta_fields_to_embed=["Timestamp", "sender"],
            embedding_separator='\n'
        )
        embedder.warm_up()

        writer = DocumentWriter(document_store=self.document_store, policy=DuplicatePolicy.OVERWRITE)

        pipeline = Pipeline()
        pipeline.add_component("cleaner", cleaner)
        pipeline.add_component("embedder", embedder)
        pipeline.add_component("writer", writer)
        pipeline.connect("cleaner", "embedder")
        pipeline.connect("embedder", "writer")
        pipeline.run(data={"cleaner": {"documents": self.documents}})
        self.indexing_pipeline = pipeline

    def build_retrieval_pipeline(self):
        text_embedder = SentenceTransformersTextEmbedder(
            model="BAAI/bge-large-en-v1.5",
            device=ComponentDevice.from_str(self.device),
            batch_size=32,
            progress_bar=True,
            normalize_embeddings=False
        )

        bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store, top_k=10)
        embedding_retriever = InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=10)
        joiner = DocumentJoiner(join_mode='concatenate', sort_by_score=True)

        ranker = TransformersSimilarityRanker(model='BAAI/bge-reranker-base', top_k=5, scale_score=True)
        ranker.warm_up()

        prompt_template = """<|system|>Data questo prompt, risponderai esclusivamente in italiano. Utilizzando le informazioni contenute nel contesto, fornisci una risposta esauriente alla domanda.
Se la risposta non può essere dedotta dal contesto, non fornire una risposta.</s>
<|user|>
Context:
  {% for doc in documents %}
  name: {{ doc.meta['name'] }} content: {{ doc.content }}
  {% endfor %};
Question: {{query}}
</s>
<|assistant|>
"""
        prompt_builder = PromptBuilder(template=prompt_template)

        hf_pipeline = {
            "device_map": "auto",

            # Rimuovi quantizzazione se usi solo CPU
            # oppure usa un modello più piccolo

            "model_kwargs": {

                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16
            }
        }

        llm = HuggingFaceLocalGenerator(
            model="HuggingFaceH4/zephyr-7b-beta",
            # model="tiiuae/falcon-7b",",
            #model="meta-llama/Llama-2-7b",
            task="text-generation",
            generation_kwargs={"max_new_tokens": 386},
            huggingface_pipeline_kwargs=hf_pipeline
        )
        llm.warm_up()

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("bm25_retriever", bm25_retriever)
        pipeline.add_component("embedding_retriever", embedding_retriever)
        pipeline.add_component("document_joiner", joiner)
        pipeline.add_component("ranker", ranker)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", llm)

        pipeline.connect("text_embedder", "embedding_retriever")
        pipeline.connect("bm25_retriever", "document_joiner")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("document_joiner", "ranker")
        pipeline.connect("ranker.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.prompt")

        self.retrieval_pipeline = pipeline

    def ask_question(self, query):
        if not query.strip():
            print("⚠️ La query è vuota. Inserisci una domanda valida.")
            return {}

        data = {
            "text_embedder": {"text": query},
            "bm25_retriever": {"query": query},
            "ranker": {"query": query},
            "prompt_builder": {"query": query}
        }

        result = self.retrieval_pipeline.run(
            data=data,
            include_outputs_from={"ranker", "prompt_builder", "llm"}
        )
        return result


if __name__ == "__main__":
    rag = WhatsAppRAG(csv_path="documenti_completi.csv")
    rag.load_and_prepare_data()
    rag.create_documents()
    rag.build_indexing_pipeline()
    rag.build_retrieval_pipeline()


    questions = [
        "Vorrei avere informazioni sulla scadenza dell'invio della modulistica",
        "Studio economia, a chi devo mandare la modulistica?",
        "Quando si schiudono le uova di fenicottero?"
    ]

    for q in questions:
        response = rag.ask_question(q)
        rich.print(response["llm"]['replies'][0])