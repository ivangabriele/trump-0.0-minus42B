from typing import Optional
from pydantic import BaseModel


class DatabasePost(BaseModel):
    id: str
    date: str
    raw_text: str
    clean_text: Optional[str] = None
