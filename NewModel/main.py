import torch.cuda
import transformers
from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# title = "Technology"
# category = "Cloud Computing"
# subcategory = "Cloud Computing and Machine Learning"
# phrase = "Artificial intelligence is the game changer in innovation along with cloud computing"

# input_seed_1 = title + category + phrase

input_seed_2_demo = '''The start-up was founded by Demis Hassabis, Shane Legg and Mustafa Suleyman in 2010.[13][14] Hassabis and Legg first met at University College London's Gatsby Computational Neuroscience Unit.[15]

During one of the interviews, Demis Hassabis said that the start-up began working on artificial intelligence technology by teaching it how to play old games from the seventies and eighties, which are relatively primitive compared to the ones that are available today. Some of those games included Breakout, Pong and Space Invaders. AI was introduced to one game at a time, without any prior knowledge of its rules. After spending some time on learning the game, AI would eventually become an expert in it. “The cognitive processes which the AI goes through are said to be very like those of a human who had never seen the game would use to understand and attempt to master it.”[16] The goal of the founders is to create a general-purpose AI that can be useful and effective for almost anything.

Major venture capital firms Horizons Ventures and Founders Fund invested in the company,[17] as well as entrepreneurs Scott Banister,[18] Peter Thiel,[19] and Elon Musk.[20] Jaan Tallinn was an early investor and an adviser to the company.[21] On 26 January 2014, Google announced the company had acquired DeepMind for $500 million,[22][23][24][25][26][27] and that it had agreed to take over DeepMind Technologies. The sale to Google took place after Facebook reportedly ended negotiations with DeepMind Technologies in 2013.[28] The company was afterwards renamed Google DeepMind and kept that name for about two years.[2]

In 2014, DeepMind received the "Company of the Year" award from Cambridge Computer Laboratory.[29]

In September 2015, DeepMind and the Royal Free NHS Trust signed their initial Information Sharing Agreement (ISA) to co-develop a clinical task management app, Streams.[30]'''

model_name = "google/pegasus-xsum"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)

model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

batch = tokenizer(input_seed_2_demo, truncation=True, padding='longest', return_tensors="pt").to(device)

translated = model.generate(**batch)

tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(tgt_text)

generator1 = pipeline(task="text-generation", model="EleutherAI/gpt-neo-1.3B", use_fast=True)

text_gen = generator1(tgt_text, do_sample = True, min_length=50, max_length=500, top_k = 60,\
                      top_p = 1.0,temperature = 0.75,number_generated_tokens = 500,repetition_penalty = 1.75,\
                      repetition_penalty_range = 300,repetition_penalty_slope = 3.33,number_show_last_actions = 15)

print(text_gen[0])
with open('gen_1.txt', 'w') as fp:
    fp.write(text_gen[0])




