import yaml
from joblib import load
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from helpers import preprocess_single_text, load_mapping

text = "ki_erstattung_test_topf2.txt Details Activity ki_erstattung_test_topf2.txt Sharing Info. Who has access M General Info. System properties Type Text Size 496 bytes Storage used 496 bytes Location testcases Owner Marc Bachmann Modified Dec 15, 2021 by Marc Bachmann Opened 6:32 PM by me Created Dec 15, 2021 Description. No description Download permissions. Viewers can download  From:  Marijke Holtkamp <m.etzrodtgweb.de> To:  tierarztrechnung@barmenia.de Subject  Tierarztrechungen Sent Thu, 21 Oct 2021 14:28:46+0200 IMG 2798.JPG IMG_2799.JPG Sehr geehrte Damen und Herren,  anbei sende ich Ihnen die Tierarztrechnung unserer Hündin Clara Tari mit der bitte um Erstattung: KreisSparkasse Köln DE 74 3705 0299 1152 0271 47 BIC COKSDE33xxX Vielen Dank.! Mit freundlichem Gruß  Marijke Holtkamp"

stopwords_locale = 'german'

stemmer = SnowballStemmer(stopwords_locale)

stop_words = set(stopwords.words(stopwords_locale))

with open('../dataset/stopwords.yaml', 'r') as f:
    curated_stop_words = yaml.safe_load(f)

text = preprocess_single_text(text, stop_words=stop_words, curated_stop_words=curated_stop_words, stemming=True, stemmer=stemmer)

mapping_dict = load_mapping(mapping_file='../dataset/mapping.yaml')

# load the saved pipleine model
for filename in ["../trained_models/model_logreg.sav", "../trained_models/model_sgd.sav"]:
    pipeline = load(filename)

    # predict on the text
    json_result = {}
    for cls, prob in zip(pipeline.classes_.tolist(), pipeline.predict_proba([text]).tolist().pop()):
        json_result[mapping_dict[cls]] = prob

    print(filename, '\n', json_result)