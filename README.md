# Article Retrieval System

This is a Python project that implements a retrieval-augmented generation (RAG) system using pre-trained language models, vector databases, and the LangChain library.
This project is a recruitment task for Nokia - Machine Learning Summer Trainee.

## Features

- Utilizes pre-trained language models text generation. The project enables to use any sufficiently large model, however I used Mistral-7B-Instruct-v0.2 and TinyLlama-1.1B.
- Supports creating vector databases from CSV files and loading existing vector database. It is not necessary to create vector database each time.
- Retrieves relevant context chunks from documents from the vector database using similarity search.
- Generates answers to questions based on the retrieved context and the provided query.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ErykMikolajek/article-retrieval-system.git
```
2. Install the required dependencies:
 ```bash
 pip install -r requirements.txt
 ```
3. Run ```model.py```

## Usage
Prepare your text data in a CSV file with a column named 'Text' containing the text content.
Create a instance of ```RetrievalModel```, pass the name of the model you want to use. Write a query and then call ```get_answer()``` on the model.
```Python
model = RetrievalModel(name)
query = "How does word2vec work?"
answer = model.get_answer(query)
print(answer['text'])
```

This will load the pre-trained language model and the vector database. The system will retrieve relevant context from documents from the vector database, and use the pre-trained model to generate an answer based on the context and the question.

## Configuration
You can modify the following parameters in the retrieval_model.py:
- model_name: The name of the pre-trained language model to use.
- database_path: The path to the directory where the csv file is stored (default: 'medium.csv').
- max_tokens: The maximum number of tokens to generate for the answer (default: 1000).

