import pandas as pd
from pydantic import BaseModel


class Chapter(BaseModel):
    book_num: int
    book_name: str
    chapter_num: int
    verses_df: pd.DataFrame
