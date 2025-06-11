from typing import Optional
from pydantic import BaseModel


class DatabasePostInsert(BaseModel):
    date: str
    raw_text: str
    clean_text: Optional[str] = None


class DatabasePost(DatabasePostInsert):
    id: str
