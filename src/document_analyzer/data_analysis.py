import sys

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import *
from model.models import *

from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser


class DocumentAnalyzer:

    def __init__(self):
        self.log=CustomLogger().get_logger(__name__)
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()

            self.parser=JsonOutputParser(pydantic_object=MetaData)
            self.fixing_parser=OutputFixingParser.from_llm(parser=self.parser,llm=self.llm)

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer:{e}")
            raise DocumentPortalException("Error in DocumentAnalyzer Initialization", sys)

    def analyze_document(self):
        pass
    
da=DocumentAnalyzer()