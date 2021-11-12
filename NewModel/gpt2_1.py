from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
import torch

import warnings
warnings.filterwarnings('ignore')

set_seed(69)

title = "Data Science"
cat = "Deep Learning "
phrase = '''Deep learning is part of a broader family of machine learning methods based on artificial \
neural networks with representation learning. Learning '''


seed_sentence = (title + cat + phrase)

model_name_1 = "EleutherAI/gpt-neo-1.3B"
model_name_2 = "google/pegasus-xsum"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_1 = GPTNeoForCausalLM.from_pretrained(model_name_1)
tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name_1)

model_2 = PegasusForConditionalGeneration.from_pretrained(model_name_2).to(device)
tokenizer_2 = PegasusTokenizer.from_pretrained(model_name_2)

batch = tokenizer_2(phrase, truncation=True,padding='longest',return_tensors="pt").to(device)

translated = model_2.generate(**batch)

tgt_text = tokenizer_2.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(tgt_text)

input_ids = tokenizer_1(tgt_text,
                            return_tensors="pt",
                            add_special_tokens=True,
                            is_split_into_words=False).input_ids

gen_tokens = model_1.generate(input_ids,
                                  do_sample=True,
                                  temperature=0.75,
                                  max_length=500,
                                  repitition_penalty=2.25,
                                  top_k=75,
                                  top_p=1.75,
                                  num_return_sequences=2,
                                  no_repeat_ngram_size=3)

gen_text = tokenizer_1.batch_decode(gen_tokens)[0]

print(gen_text)

with open('Blog.txt', 'w') as fp:
    fp.write(gen_text)

