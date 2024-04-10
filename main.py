from vector_database import generate_database

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS


class RetrievalModel:
    def __init__(self, model_name: str):
        # self.vector_database = generate_database(database_path)
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype="float16",
        #     bnb_4bit_use_double_quant=False,
        # )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=300
        )

        mistral_llm = HuggingFacePipeline(pipeline=self.text_generation_pipeline)
        self.prompt_template = """
        [INST] 
        Answer the question based on the following context:
        {context}
        [/INST]

        Question:
        {question} 
         """

        # Create prompt from prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template,
        )
    def invoke(self, query: str):
        pass

if __name__ == '__main__':
    vector_database = generate_database('medium.csv')

    model = RetrievalModel('../mistral_model')
    # print(vector_database)
