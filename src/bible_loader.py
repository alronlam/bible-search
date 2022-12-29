import pandas as pd
import streamlit as st
from loguru import logger


@st.cache()
def load_bible(metadata_csv, verses_csv):
    # There is one constant metadata file (metadata_csv),
    #   and another csv file containing the actual verses in the specified version (bible_csv).
    metadata_df = pd.read_csv(metadata_csv)
    verses_df = pd.read_csv(verses_csv, escapechar="\\")
    df = pd.merge(verses_df, metadata_df, on="b")
    df = df.fillna("")  # Some verses are blank in some versions

    df = df[["n", "c", "v", "t_x"]]

    # The data sources used have this convention in the columns.
    # Renaming them here for ease of remembrance.
    col_rename = {"n": "book", "c": "chapter", "v": "verse", "t_x": "text"}
    df = df.rename(columns=col_rename)

    # Create a human-friendly string of specifying a verse (e.g. Genesis 1:1)
    df["source"] = df.apply(
        lambda row: f"{row['book']} {row['chapter']}:{row['verse']}", axis=1
    )

    logger.info(
        f"Successfully loaded Bible DF with {len(df):,} rows. Columns: {df.columns.tolist()}"
    )

    return df
