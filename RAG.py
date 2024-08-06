import os
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def _pdf_reader(file_path):

    reader = PdfReader(file_path)
    content = {
        f"page_{i+1}": page.extract_text() for i, page in enumerate(reader.pages)
    }
    return content

class RAGRetriever():
    def __init__(
            self,
            embedding_model_ckpt: str = "all-mpnet-base-v2",
            source_dir: str = "sources_data",
            chunk_size: int = 100,
            overlaps: int = 2,
    ):
        self.embedding_model_ckpt = embedding_model_ckpt
        self.source_dir = source_dir
        self.chunk_size = chunk_size # in terms of # of words
        self.overlaps = overlaps # in terms of # of sentences

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _pgsl_connection(self):
        connection = psycopg2.connect(
            database=os.getenv("psql_db"),
            user=os.getenv("psql_username"),
            password=os.getenv("psql_pwd"),
            host="localhost",
            port=5432,
        )
        connection.set_client_encoding("UTF8")
        return connection

    def _parse_data_in_chunks(self):
        data = []

        chunk = []
        chunk_char_count = 0
        begin_page = "page_1"

        for file_path in os.listdir(self.source_dir):
            #read in contents
            if ".pdf" in file_path:
                file_content = _pdf_reader(
                    os.path.join(self.source_dir, file_path)
                )

            for page, content in file_content.items():
                content = content.replace("\n", " ")
                sentences = iter(content.split(". "))

                while True:
                    try:
                        sentence = next(sentences)
                    except StopIteration:
                        break

                    chunk_char_count += sentence.count(" ")
                    chunk.append(sentence)
                    if chunk_char_count > self.chunk_size:
                        end_page = page
                        data.append(
                            {
                                "end_file": file_path,
                                "pages": f"{begin_page}-{end_page}",
                                "content": ". ".join(chunk) + ". ",
                            }
                        )

                        begin_page = page
                        chunk = chunk[-self.overlaps:]
                        chunk_char_count = sum([s.count(" ") for s in chunk])

        return data

    def embedding(self, parsed_data: list | str):
        if isinstance(parsed_data, str):
            parsed_data = [
                {
                    "end_file": "NaN",
                    "pages": "NaN",
                    "content": parsed_data,
                }
            ]

        embedding_model = SentenceTransformer(
            self.embedding_model_ckpt,
            device=self.device,
        )

        for chunk in parsed_data:
            chunk["embedding_model"] = self.embedding_model_ckpt
            chunk["embedding"] = embedding_model.encode(chunk["content"])
        return parsed_data

    def _db_ingestion(self, embedding_data: list):
        conn = self._pgsl_connection()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)

        create_table = f"""
        CREATE TABLE IF NOT EXISTS rag (
                    id bigserial primary key, 
                    end_file text,
                    pages text,
                    content text,
                    embedding_model text,
                    embedding vector({len(embedding_data[0]["embedding"])})
                    );
                    """

        cur.execute(create_table)
        conn.commit()

        data = [
            (
                str(row["end_file"]),
                str(row["pages"]),
                str(row["content"]),
                str(row["embedding_model"]),
                row["embedding"].tolist(),
            )
            for row in embedding_data
        ]

        execute_values(
            cur,
            "INSERT INTO rag (end_file, pages, content, embedding_model, embedding) VALUES %s",
            data,
        )
        conn.commit()

        indexing = "CREATE INDEX ON rag USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction = 64)"
        cur.execute(indexing)
        conn.commit()


    def _ann_search(self, embedding_data: list, top_k: int):
        conn = self._pgsl_connection()
        cur = conn.cursor()

        search_query = f"""
        BEGIN;
        SET LOCAL hnsw.ef_search = 100;
        SELECT end_file, pages, embedding_model, content 
        FROM rag 
        ORDER BY embedding <=> '{embedding_data[0]["embedding"].tolist()}' 
        LIMIT {top_k};
"""

        cur.execute(search_query)
        results = cur.fetchall()

        info = [row[:-1] for row in results]
        contents = [row[-1] for row in results]
        return info, contents

    def _prompt_engineering(self, contents: list, query: str):
        prompt = "Please use the provided information to answer question."
        for i, info in enumerate(contents):
            prompt = f"{prompt}\n Information {i+1}: {info}"

        prompt = f"{prompt}\n\n Question: {query}\n Answer:"
        return prompt


    def process_reference_data(self):
        parsed_data = self._parse_data_in_chunks()
        embedding_data = self.embedding(parsed_data)
        self._db_ingestion(embedding_data)


    def process_prompt(self, prompt_query: str, top_k: int):
        query_embedding = self.embedding(prompt_query)
        retrieved_info, retrieved_contents = self._ann_search(
            query_embedding,
            top_k,
        )
        prompt = self._prompt_engineering(
            retrieved_contents,
            prompt_query,
        )

        return prompt


class RAGGenerator():
    def __init__(self, model_ckpt, quantization_config):
        self.model_ckpt = model_ckpt
        self.quantization_config = quantization_config

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_ckpt
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_ckpt,
            quantization_config=self.quantization_config,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def generate(self, prompt: str, temperature: float =0.1):

        token_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        response_ids = self.llm_model.generate(
            **token_ids,
            temperature=temperature,
            max_new_tokens=1000,
        )
        response = self.tokenizer.decode(response_ids[0])
        return response


def rag_pipeline(
        prompt_query,
        top_k_context,
        llm_model_ckpt,
        llm_generate_temperature,
        quantization_4_bits=True,
):


    rag_retriever = RAGRetriever()
    prompt = rag_retriever.process_prompt(prompt_query, top_k=top_k_context)


    if quantization_4_bits:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        quantization_config = None

    RAG_generator = RAGGenerator(llm_model_ckpt, quantization_config)
    response = RAG_generator.generate(prompt, temperature=llm_generate_temperature)
    print(response)
    return response

if __name__=="__main__":

    """
    #prepare rag data
    rag_agent = RAGRetriever(embedding_model_ckpt="all-mpnet-base-v2", source_dir="sources_data", chunk_size=100, overlaps=2)
    rag_agent.process_reference_data()
    """

    ################################################################################

    prompt_query = "What's the direction of s2 mode?"
    top_k_context = 3

    llm_model_ckpt = "Meta-Llama-3.1-8B"
    llm_generate_temperature = 0.1
    quantization_4_bits = True

    rag_pipeline(prompt_query, top_k_context, llm_model_ckpt, llm_generate_temperature, quantization_4_bits)
    ################################################################################


