from vector_database import generate_database

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain

import warnings
warnings.simplefilter('ignore')


class RetrievalModel:
    def __init__(self, model_name, database_path='medium.csv', max_tokens=1000):
        print("Loading model:")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.database_path = database_path
        self.vector_db = generate_database(self.database_path)
        self.max_tokens = max_tokens

        self.text_generation_pipeline = self.setup_text_generation_pipeline()
        self.mistral_llm = self.setup_mistral_llm()
        self.prompt = self.setup_prompt_template()
        self.llm_chain = self.setup_llm_chain()

    def setup_text_generation_pipeline(self):
        return pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=self.max_tokens
        )

    def setup_mistral_llm(self):
        return HuggingFacePipeline(pipeline=self.text_generation_pipeline)

    def setup_prompt_template(self):
        prompt_template = """
        [INST] 
        Answer the question based on the following context:
        {% for doc in context %}
        {{ doc.page_content }}
        {% endfor %}
        [/INST]

        Question:
        {{question}}
        """
        return PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
            template_format="jinja2"
        )

    def setup_llm_chain(self):
        return LLMChain(llm=self.mistral_llm, prompt=self.prompt)

    def get_answer(self, local_query):
        print("Generating response:")
        if self.vector_db:
            retriever = self.vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 3}
            )
            rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | self.llm_chain
            )
            response = rag_chain.invoke(local_query)
        else:
            response = self.llm_chain.run(local_query)

        return response


if __name__ == '__main__':
    name = '../tiny_llama'
    # name = '../mistral_model'
    model = RetrievalModel(name, max_tokens=1000)
    query = "How does word2vec work?"
    answer = model.get_answer(query)
    print("Question:", answer["question"])
    print(answer['text'])




