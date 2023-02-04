import urllib

import pandas as pd
from pydantic import BaseModel


class Chapter(BaseModel):
    book_name: str
    chapter_num: int
    verses_df: pd.DataFrame
    highlight_verses_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return f"{self.book_name} {self.chapter_num}"

    def get_formatted_text(self):

        # Construct chapter text
        texts = []
        for _, row in self.verses_df.iterrows():
            text = row["text"]
            if text in self.highlight_verses_df["text"].tolist():
                text = f"**:green[{text}]**"
            text = f"<sup>{row['verse']}</sup> {text}"
            texts.append(text)
        chapter_text = " ".join(texts)
        return chapter_text

    def get_biblegateway_url(self, version="NIV"):
        return f"https://www.biblegateway.com/passage/?search={urllib.parse.quote(self.book_name)}+{self.chapter_num}&version={version}"
        
