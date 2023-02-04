import pandas as pd
from pydantic import BaseModel


class Chapter(BaseModel):
    book_name: str
    chapter_num: int
    verses_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


    def __str__(self) -> str:
        return f"{self.book_name} {self.chapter_num}"