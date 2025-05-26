import re
import spacy

# Import ekphrasis components
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# Text Preprocessing
# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize ekphrasis text processor
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=[
        "url",
        "email",
        "percent",
        "money",
        "phone",
        "user",
        "time",
        "date",
        "number",
    ],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
    segmenter="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,
    spell_correction=True,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons],
)

def clean_html(text):
    """Remove HTML tags from text"""
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", text)
    return cleantext

def preprocess_text(text):
    """
    Preprocess text using ekphrasis and additional cleaning
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Preprocessed text as a string
    """
    # First clean any HTML
    cleaned_html_text = clean_html(text)

    # Process with ekphrasis
    word_list = text_processor.pre_process_doc(cleaned_html_text)
    
    # Remove annotation markers
    remove_words = [
        "<allcaps>", "</allcaps>", 
        "<hashtag>", "</hashtag>", 
        "<elongated>", "</elongated>",
        "<emphasis>", "</emphasis>", 
        "<repeated>", "</repeated>"
    ]
    word_list = [word for word in word_list if word not in remove_words]
    
    # Remove angle brackets from annotated words
    processed_text = " ".join(word_list)
    processed_text = re.sub(r"[<\*>]", "", processed_text)
    
    # Use spaCy for additional normalization (without tokenization)
    doc = nlp(processed_text)
    # Create a normalized string, preserving meaningful whitespace
    normalized_text = " ".join([token.text for token in doc])
    
    return normalized_text