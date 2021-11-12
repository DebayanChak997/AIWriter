from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
from keytotext import pipeline as pipe
import torch

device = 'cpu'
KEYWORDS = []

def Models_Download(SEED=False):

    if SEED == False:
        model_name_1 = "EleutherAI/gpt-neo-125M"
        model_name_2 = "google/pegasus-large"
        NLP = pipe("k2t-base")

        model_1 = GPTNeoForCausalLM.from_pretrained(model_name_1)
        tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name_1)

        model_2 = PegasusForConditionalGeneration.from_pretrained(model_name_2).to(device)
        tokenizer_2 = PegasusTokenizer.from_pretrained(model_name_2)

    return model_1, tokenizer_1, model_2, tokenizer_2, NLP


set_seed(42)

def genText(title="", cat="", phrase="", sub_category="", keywords=""):

    global KEYWORDS
    for key in keywords:
        KEYWORDS.append(key)
    
    model_1, tokenizer_1, model_2, tokenizer_2, nlp_model = Models_Download(SEED=False)
    
    text_from_key = nlp_model(KEYWORDS)
    abstract = str(text_from_key) + phrase
    print(abstract)
    seed_sentence = (title + abstract + cat + sub_category)

    batch = tokenizer_2(seed_sentence, truncation=True,
                        padding='longest',
                        return_tensors="pt").to(device)

    translated = model_2.generate(**batch)

    tgt_text = tokenizer_2.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print(tgt_text)

    input_ids = tokenizer_1(tgt_text,
                            return_tensors="pt",
                            add_special_tokens=True,
                            is_split_into_words=True,
                            max_length=512).input_ids

    gen_tokens = model_1.generate(input_ids,
                                  do_sample=True,
                                  temperature=0.65, #def user fun(creative)(0-1.5)
                                  max_length=350,
                                  repitition_penalty=1.25,
                                  top_k=65,
                                  top_p=1.5, #def user fun(0-1)
                                  num_return_sequences=2,
                                  no_repeat_ngram_size=3,
                                  vocab_size=50257,
                                  num_layers=24,
                                  num_head=16,
                                  intermediate_size=8192,
                                  max_position_embeddings=2048,
                                  layer_norm_epsilon=1e-5)

    gen_text = tokenizer_1.batch_decode(gen_tokens)[0]

    print(gen_text)

    return str(gen_text)


def GPT2MainFunc(X_title, X_sub_category, X_category, X_phrase, x_keyword):

    return genText(title=X_title, cat=X_category, phrase=X_phrase, sub_category=X_sub_category, keywords=x_keyword)
    
