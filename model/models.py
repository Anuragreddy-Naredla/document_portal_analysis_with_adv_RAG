# pydantic is used for validation.
from pydantic import BaseModel,Field,RootModel
from typing import Optional,List,Dict,Any,Union

class MetaData(BaseModel):
    Summary:List[str]=Field(default_factory=list,description="Summary of the document")
    Title: str
    Author: str
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int,str]
    SentimentTone: str

class ChangeFormat(BaseModel):
    Page: str
    changes: str

class SummaryResponse(RootModel[List[ChangeFormat]]):
    pass
