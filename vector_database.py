from tqdm import tqdm

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def generate_database(file_path: str, embeddings_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                        chunk_size: int = 1024, save_database: bool = True, local_path: str = 'articles_indexed'):
    print("Loading database:")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    try:
        vector_db = FAISS.load_local(local_path, embeddings)
        print('Vector database loaded successfully')
    except Exception as e:
        vector_db = None
        print('Vector database could not be loaded, creating new embeddings:')
        try:
            loader = CSVLoader(file_path=file_path, source_column='Text')
            data = loader.load()
        except Exception as e:
            print("Data cannot be loaded")
            return e
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=24
        )
        chunked_documents = text_splitter.split_documents(data)
        chunked_documents = chunked_documents[:2]
        with tqdm(total=len(chunked_documents), desc="Indexing documents") as pbar:
            for doc in chunked_documents:
                if vector_db:
                    vector_db.add_documents([doc])
                else:
                    vector_db = FAISS.from_documents([doc], HuggingFaceEmbeddings(model_name=embeddings_model_name))
                pbar.update(1)
        if save_database:
            vector_db.save_local(local_path)
    return vector_db
