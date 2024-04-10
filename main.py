from vector_database import generate_database

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain

from typing import Any
from uuid import UUID
from tqdm.auto import tqdm
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

import warnings
warnings.simplefilter('ignore')


class BatchCallback(BaseCallbackHandler):
    def __init__(self, total: int):
        super().__init__()
        self.count = 0
        self.progress_bar = tqdm(total=total, desc="Generating tokens")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any):
        self.count += 1
        self.progress_bar.update(1)


class RetrievalModel:
    def __init__(self, model_name, database_path='articles_indexed', max_tokens=100):
        print("Loading model:")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.database_path = database_path
        self.vector_db = self.load_vector_db()
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
        {context}
        [/INST]

        Question:
        {question} 
        """
        return PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

    def setup_llm_chain(self):
        return LLMChain(llm=self.mistral_llm, prompt=self.prompt)

    def load_vector_db(self):
        return generate_database(self.database_path)

    def get_answer(self, local_query):
        # cb = BatchCallback(len(local_query))
        print("Generating response:")
        if self.vector_db:
            retriever = self.vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 1}
            )
            rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | self.llm_chain
            )
            response = rag_chain.invoke(local_query)
        else:
            response = self.llm_chain.run(local_query)

        # cb.progress_bar.close()
        return response


if __name__ == '__main__':
    # name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    name = '../tiny_llama'
    # name = '../mistral_model'
    model = RetrievalModel(name, max_tokens=20)
    query = "Tell me about Transformers."
    answer = model.get_answer(query)
    print(answer)


