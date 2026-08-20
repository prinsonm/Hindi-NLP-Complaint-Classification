import pandas as pd
import re
import unicodedata
import stanza


# --------------------------------------------------
# 1. Initialize Hindi NLP pipeline
# --------------------------------------------------

nlp = stanza.Pipeline(
    lang="hi",
    processors="tokenize,pos,lemma",
    use_gpu=False
)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv("Hindi_Dataset.csv")

print(df.head())


# --------------------------------------------------
# 3. Hindi stopwords
# --------------------------------------------------

stop_words = set([
    "मैं", "मेरा", "मेरी", "मेरे",
    "हम", "हमारा", "हमारी",
    "आप", "आपका", "आपकी",
    "यह", "वह", "ये", "वे",
    "एक", "और", "या",
    "का", "के", "की",
    "को", "से", "में", "पर",
    "है", "हैं", "था", "थी", "थे",
    "हो", "रहा", "रही", "रहे",
    "कर", "करना", "किया",
    "लिए", "द्वारा",
    "भी", "तो", "तक"
])


# --------------------------------------------------
# 4. Preprocessing function
# --------------------------------------------------

def preprocess(text):

    # Convert to string
    text = str(text)


    # --------------------------------------------------
    # Unicode Normalization
    # --------------------------------------------------

    text = unicodedata.normalize("NFKC", text)


    # --------------------------------------------------
    # Filtration
    # --------------------------------------------------

    # Remove URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    # Remove email addresses
    text = re.sub(
        r'\S+@\S+',
        ' ',
        text
    )

    # Remove HTML tags
    text = re.sub(
        r'<.*?>',
        ' ',
        text
    )


    # --------------------------------------------------
    # Script Validation / Filtering
    # Keep only Devanagari characters and spaces
    # --------------------------------------------------

    text = re.sub(
        r'[^\u0900-\u097F\s]',
        ' ',
        text
    )


    # --------------------------------------------------
    # Remove digits
    # --------------------------------------------------

    text = re.sub(
        r'\d+',
        ' ',
        text
    )


    # --------------------------------------------------
    # Remove extra whitespace
    # --------------------------------------------------

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()


    # --------------------------------------------------
    # Tokenization + Lemmatization
    # --------------------------------------------------

    doc = nlp(text)

    words = []

    for sentence in doc.sentences:

        for word in sentence.words:

            lemma = word.lemma

            if lemma:
                words.append(lemma)


    # --------------------------------------------------
    # Stopword Removal
    # --------------------------------------------------

    words = [
        word
        for word in words
        if word not in stop_words
    ]


    # --------------------------------------------------
    # Final text
    # --------------------------------------------------

    return " ".join(words)


# --------------------------------------------------
# 5. Apply preprocessing
# --------------------------------------------------

df["Cleaned_Complaint"] = df["complaint"].apply(
    preprocess
)


# --------------------------------------------------
# 6. Save preprocessed dataset
# --------------------------------------------------

df.to_csv(
    "Hindi_Preprocessed.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nPreprocessing completed!")
print(df[["complaint", "Cleaned_Complaint"]].head())