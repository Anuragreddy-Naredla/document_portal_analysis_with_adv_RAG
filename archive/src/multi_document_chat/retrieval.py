
import sys
import os
from operator import itemgetter
from typing import List,Optional

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from logger import GLOBAL_LOGGER as log

class ConversationalRag:

    def __init__(self,session_id:str,retriever=None):
        self.session_id=session_id
        try:
            self.llm=ModelLoader().load_llm()
            self.contextualize_prompt=PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt=PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            if retriever is None:
                raise ValueError("Retriever cannot be None")
            self.retriever=retriever
            self._build_lcel_chain()
            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationRAG",error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG",sys)

    def load_retriever_from_faiss(self,index_path:str):
        try:
            embeddings=ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found:{index_path}")
            
            vectorstore=FAISS.load_local(index_path,embeddings,allow_dangerous_deserialization=True)
            self.retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})
            log.info("FAISS retriever loaded successfully",index_path=index_path,session_id=self.session_id)
            return self.retriever


        except Exception as e:
            log.error("Failed to load retriever from faiss database",error=str(e))
            raise DocumentPortalException("Error in load_retriever_from_faiss function",sys)

    def invoke(self,user_input,chat_history:Optional[List[BaseMessage]]=None):
        try:
            chat_history = chat_history or []
            payload={"input":user_input,"chat_history":chat_history}
            answer=self.chain.invoke(payload)
            if not answer:
                log.warning("No answer generated", user_input=user_input,session_id=self.session_id)
                return "No answe generated"
            log.info("Chain invoked successfully", session_id=self.session_id,user_input=user_input,answer_preview=answer[:150])
            return answer
        except Exception as e:
            log.error("Failed to invoke in retriever file",error=str(e))
            raise DocumentPortalException("Error in invoke function",sys)

    def load_llm(self):
        try:
            self.llm=ModelLoader().load_llm()
            if not self.llm:
                raise ValueError("LLM could not be loaded.")
            log.info("LLM Loaded successfully", session_id=self.session_id)
            return self.llm
        except Exception as e:
            log.error("Failed to load llm",error=str(e))
            raise DocumentPortalException("Error in load_llm function",sys)


    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def _build_lcel_chain(self):
        try:
            question_rewriter=(
                {"input":itemgetter("input"), "chat_history":itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            retriever_docs = question_rewriter| self.retriever | self._format_docs
            self.chain=(
                {
                    "context":retriever_docs,
                    "input":itemgetter("input"),
                    "chat_history":itemgetter("chat_history"),
                }
                |self.qa_prompt
                |self.llm
                |StrOutputParser()
            )
            log.info("LCEL built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build the lcel chain",error=str(e))
            raise DocumentPortalException("Error in _build_lcel_chain function",sys)




