# Bible Search
What does the Bible say about this or that? This tool was created to help find Biblical verses for certain topics. For example, searching for **"salvation"** should show you all relevant verses, such as [Ephesians 2:8-9](https://www.biblegateway.com/passage/?search=Ephesians+2%3A8-9&version=NIV) or [John 3:16](https://www.biblegateway.com/passage/?search=john+3%3A16&version=NIV). 

The web app was built using [Streamlit](streamlit.io), and uses [semantic similarity](https://github.com/UKPLab/sentence-transformers) under the hood to perform the search. This project is a work in progress, and we are currently working to deploy a version that is publicly accessible. Stay tuned!

# Bible Versions
We're working on adding more Bible versions to choose from. The data currently used is obtainable from the [Bible corpus on Kaggle by Oswin](https://www.kaggle.com/oswinrh/bible). 

# Local Set-up
To run the app locally:
* Download and unzip the data from the [Bible corpus on Kaggle by Oswin](https://www.kaggle.com/oswinrh/bible) to a local `data` folder in the root directory.
* Install libraries by running `pip install -r requirements.txt`.
* Finally, run `streamlit run app.py` and you're good to go! You should see the app  on your browser at http://localhost:8501/.
