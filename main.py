from vector_database import generate_database
# from langchain_community.vectorstores import FAISS

class RetrievalModel:
    def __init__(self, database_path: str):
        self.vector_database = generate_database(database_path)


if __name__ == '__main__':
    vector_database = generate_database('medium.csv')
    print(vector_database)
