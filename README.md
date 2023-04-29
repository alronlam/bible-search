# Bible Search

Try the app here: https://huggingface.co/spaces/alronlam/bible-search

![App Screenshot](/static/app_screenshot.PNG)


What does the Bible say about this or that? This tool was created to help find Biblical verses for certain topics. The web app was built using [Streamlit](streamlit.io), and uses [semantic similarity](https://github.com/UKPLab/sentence-transformers) and some re-ranking logic under the hood to perform the search and highlight relevant verses in the context of chapters.

# Local Dev

One-time Setup:
```
make conda-env
make setup
```

Activate the conda env:
```
conda activate ./env
```

Run the streamlit app:
```
bash run_streamlit.sh
```

# Data
The CSV file of Bible verses were taken from this [website](https://my-bible-study.appspot.com/). The file is also included in the repo, so there is no need to manually download. But you can refer to this website if you want to use a different version.
