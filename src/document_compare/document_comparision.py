import sys
from dotenv import load_dotenv
import pandas as pd

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import SummaryResponse
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader

from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

class DocumentComapareLLM:
    def __init__(self):
        load_dotenv()
        self.log=CustomLogger().get_logger(__name__)
        self.loader=ModelLoader()
        self.llm=self.loader.load_llm()
        self.parser=JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser=OutputFixingParser.from_llm(parser=self.parser,llm=self.llm)
        self.prompt=PROMPT_REGISTRY["document_comparison"]
        self.chain=self.prompt|self.llm|self.parser
        self.log.info("DocumentComparision LLM initialized with model and parser")

    def compare_documents(self,combined_docs):
        try:
            inputs={
                "combined_docs":combined_docs,
                "format_instruction":self.parser.get_format_instructions()
            }
            self.log.info("Starting document comparison",inputs=inputs)
            response=self.chain.invoke(inputs)
            self.log.info("Document comparision completed",response=response)
            return self.format_response(response)

        except Exception as e:
            self.log.error(f"Error in compare documents: {e}")
            raise DocumentPortalException("An error occured while performing comparing the docs.",sys) from e

    def format_response(self,response:list[dict]):
        try:
            df=pd.DataFrame(response)
            self.log.info("Response formatted into DataFrame",dataframe=df)
            return df
        except Exception as e:
            self.log.error(f"Error in compare documents: {e}")
            raise DocumentPortalException("An error occured in format response.",sys) from e


    
