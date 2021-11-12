# ####################Two Blog Generation (08/10/21) ##########################################

# from numpy import maximum
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# from transformers import set_seed
# import torch


# def Models_Download(SEED=False, DEVICE='cpu'):

#     device = DEVICE

#     if SEED == False:

#         model_name_1 = "EleutherAI/gpt-neo-125M"
#         model_name_2 = "google/pegasus-large"

#         model_1 = GPTNeoForCausalLM.from_pretrained(model_name_1).to(device)
#         tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name_1)

#         model_2 = PegasusForConditionalGeneration.from_pretrained(model_name_2).to(device)
#         tokenizer_2 = PegasusTokenizer.from_pretrained(model_name_2)

        
#         return model_1, tokenizer_1, model_2, tokenizer_2
    
# set_seed(69)

# def genText(title="", cat="", phrase="", sub_category="", DEVICE="cpu", top_p=0.8, top_k=40, temperature=0.75, repitition_penalty=2.15):

#     device = DEVICE
#     temp = temperature
#     topP = top_p
#     topK = top_k
#     repPenal = repitition_penalty

#     seed_sentence = (title + phrase + cat + sub_category)

#     model_1, tokenizer_1, model_2, tokenizer_2 = Models_Download(SEED=False)

#     batch = tokenizer_2(seed_sentence, truncation=True,
#                         return_tensors="pt").to(device)

#     translated = model_2.generate(**batch)

#     tgt_text = tokenizer_2.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

#     #print(tgt_text)

#     input_ids = tokenizer_1(tgt_text,
#                             return_tensors="pt",
#                             add_special_tokens=True,
#                             is_split_into_words=True,
#                             max_length=600, truncation=True).input_ids

#     gen_tokens = model_1.generate(input_ids,
#                                   do_sample=True,
#                                   temperature=temp,
#                                   max_length=600,
#                                   min_length=500,
#                                   repitition_penalty=repPenal,
#                                   top_k=topK,
#                                   top_p=topP,
#                                   num_return_sequences=3,
#                                   no_repeat_ngram_size=6,
#                                   vocab_size=50256,
#                                   num_layers=24,
#                                   num_head=24,
#                                   intermediate_size=8192,
#                                   max_position_embeddings=2048,
#                                   layer_norm_epsilon=1e-5, truncation=True)

#     gen_text = tokenizer_1.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

#     # print(gen_text)

#     return str(gen_text)


# def GPT2MainFunc(X_title, X_sub_category, X_category, X_phrase, DEVICE="cpu"):

#     device = DEVICE
#     TOP_P = 0.85
#     TOP_K = 75
#     TEMP = 0.75
#     REP_PENAL = 1.75
#     GEN_TOKEN = 3
#     counter = 1
#     temp1 = []
#     for i in range(GEN_TOKEN):
#         first_out_put_without_preprocessing = genText(title=X_title,cat=X_category,phrase=X_phrase,sub_category=X_sub_category, 
#                                                   DEVICE=device, top_p=TOP_P, top_k=TOP_K, temperature=TEMP, repitition_penalty=REP_PENAL)
#         temp1.append(" The AI-Transformer Generated Output {} is : \n\n {} \n\n ".format(counter, first_out_put_without_preprocessing))
#         counter = counter + 1
#         if counter == GEN_TOKEN :
#             print("Done")
#             break
#         else :
#             continue
#     def final_preprocess_function(DEVICE='cpu', delim=".", randomS=0):
    
#         delim = delim
#         randomS = randomS
#         temp2 = []
#         for i in range(len(temp1)):
#             string = temp1[i]
#             delimiter = delim
#             string_list = string.rsplit(delimiter,maxsplit = 1)
#             string_list = string_list[randomS]+delim
#             temp2.append(string_list)
#         return temp2
               
#     return final_preprocess_function()


#     ########################## Two Blog Gen End(08/10/21) ##########################




################################ Smooth Working Code 10/8/21 ##########################################
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
import torch

DEVICE_USAGE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def Models_Download(SEED=False, DEVICE=DEVICE_USAGE):

    device = DEVICE

    if SEED == False:

        model_name_1 = "EleutherAI/gpt-neo-125M"
        model_name_2 = "google/pegasus-large"

        model_1 = GPTNeoForCausalLM.from_pretrained(model_name_1).to(device)
        tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name_1)

        model_2 = PegasusForConditionalGeneration.from_pretrained(model_name_2).to(device)
        tokenizer_2 = PegasusTokenizer.from_pretrained(model_name_2)

        return model_1, tokenizer_1, model_2, tokenizer_2

set_seed(69)

def genText(title="", cat="", phrase="", sub_category="", DEVICE=DEVICE_USAGE,
            temp_1=0.7, top_p_1=0.95, top_k_1=65, r_penalty_1=1.75):

    device = DEVICE
    TEMP = 0.75
    TOP_P = 0.95
    TOP_K = 40
    REP_PENALTY = 2.15

    seed_sentence = (title + phrase + cat + sub_category)

    model_1, tokenizer_1, model_2, tokenizer_2 = Models_Download(SEED=False)

    batch = tokenizer_2(seed_sentence, truncation=True,
                        return_tensors="pt").to(device)

    translated = model_2.generate(**batch)

    tgt_text = tokenizer_2.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print(tgt_text)

    input_ids = tokenizer_1(tgt_text,
                            return_tensors="pt",
                            add_special_tokens=True,
                            is_split_into_words=True,
                            max_length=600, truncation=True).input_ids

    gen_tokens = model_1.generate(input_ids,
                                  do_sample=True,
                                  temperature=0.8,
                                  max_length=800,
                                  min_length=700,
                                  repitition_penalty=2.25,
                                  top_k=90,
                                  top_p=0.8,
                                  num_return_sequences=3,
                                  no_repeat_ngram_size=4,
                                  vocab_size=50256,
                                  num_layers=24,
                                  num_head=24,
                                  intermediate_size=8192,
                                  max_position_embeddings=2048,
                                  layer_norm_epsilon=1e-5, truncation=True)

    gen_text = tokenizer_1.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    print(gen_text)

    return str(gen_text)

def GPT2MainFunc(X_title, X_sub_category, X_category, X_phrase, DEVICE=DEVICE_USAGE):

    device = DEVICE
    X_T = X_title
    X_S_C = X_sub_category
    X_C = X_category
    X_P = X_phrase

    first_out_put_without_preprocessing = genText(title=X_T, cat=X_C, phrase=X_P, sub_category=X_S_C, DEVICE=device)

    def final_preprocess_function(seed=first_out_put_without_preprocessing, delim=".", randomS=0):

        randomS = randomS

        string = seed

        delimiter = delim

        string_list = string.rsplit(delimiter,maxsplit = 1)

        return (string_list[randomS]+delim)

    return final_preprocess_function()

################################## Smooth code End  ##########################





    
# # # ######################################
# # # # SLIDER CODE TO BE UPDATED AND INTEGRATED LATER


# # # '''
# # # from numpy import maximum
# # # from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# # # from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# # # from transformers import set_seed
# # # import torch


# # # def Models_Download(SEED=False, DEVICE='cpu'):

# # #     device = DEVICE

# # #     if SEED == False:

# # #         model_name_1 = "EleutherAI/gpt-neo-125M"
# # #         model_name_2 = "google/pegasus-large"

# # #         model_1 = GPTNeoForCausalLM.from_pretrained(model_name_1).to(device)
# # #         tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name_1)

# # #         model_2 = PegasusForConditionalGeneration.from_pretrained(model_name_2).to(device)
# # #         tokenizer_2 = PegasusTokenizer.from_pretrained(model_name_2)


# # #         return model_1, tokenizer_1, model_2, tokenizer_2

# # # set_seed(69)

# # # def genText(title="", cat="", phrase="", sub_category="", DEVICE="cpu", top_p=0, top_k=40, temperature=0.75, repitition_penalty=1.25):

# # #     device = DEVICE
# # #     temp = temperature
# # #     topP = top_p
# # #     topK = top_k
# # #     repPenal = repitition_penalty

# # #     seed_sentence = (title + phrase + cat + sub_category)

# # #     model_1, tokenizer_1, model_2, tokenizer_2 = Models_Download(SEED=False)

# # #     batch = tokenizer_2(seed_sentence, truncation=True,
# # #                         return_tensors="pt").to(device)

# # #     translated = model_2.generate(**batch)

# # #     tgt_text = tokenizer_2.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

# # #     #print(tgt_text)

# # #     input_ids = tokenizer_1(tgt_text,
# # #                             return_tensors="pt",
# # #                             add_special_tokens=True,
# # #                             is_split_into_words=True,
# # #                             max_length=600, truncation=True).input_ids

# # #     gen_tokens = model_1.generate(input_ids,
# # #                                   do_sample=True,
# # #                                   temperature=temp,
# # #                                   max_length=750,
# # #                                   min_length=450,
# # #                                   repitition_penalty=repPenal,
# # #                                   top_k=topK,
# # #                                   top_p=topP,
# # #                                   num_return_sequences=3,
# # #                                   no_repeat_ngram_size=6,
# # #                                   vocab_size=50256,
# # #                                   num_layers=24,
# # #                                   num_head=24,
# # #                                   intermediate_size=8192,
# # #                                   max_position_embeddings=2048,
# # #                                   layer_norm_epsilon=1e-5, truncation=True)

# # #     gen_text = tokenizer_1.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

# # #     # print(gen_text)

# # #     return str(gen_text)


# # # def GPT2MainFunc(X_title, X_sub_category, X_category, X_phrase, top_p=0, top_k=40, temperature=0.75, repitition_penalty=1.25 , gen_token = 2):

# # #     device = "cpu"
# # #     TOP_P = top_p
# # #     TOP_K = top_k
# # #     TEMP = temperature
# # #     REP_PENAL = repitition_penalty
# # #     GEN_TOKEN = gen_token
# # #     temp1 = []
# # #     for i in range(GEN_TOKEN):
# # #         first_out_put_without_preprocessing = genText(title=X_title,cat=X_category,phrase=X_phrase,sub_category=X_sub_category,
# # #                                                   DEVICE=device, top_p=TOP_P, top_k=TOP_K, temperature=TEMP, repitition_penalty=REP_PENAL)
# # #         temp1.append(first_out_put_without_preprocessing)

# # #     def final_preprocess_function(DEVICE='cpu', delim=".", randomS=0):

# # #         delim = delim
# # #         randomS = randomS
# # #         temp2 = []
# # #         for i in range(len(temp1)):
# # #             string = temp1[i]
# # #             delimiter = delim
# # #             string_list = string.rsplit(delimiter,maxsplit = 1)
# # #             string_list = string_list[randomS]+delim
# # #             temp2.append(string_list)
# # #         return temp2

# # #     return final_preprocess_function()
# # #   '''

# #########################################################################################
# #########################################################################################
# #################### DATED 07-10-2021 ###################################################
#
# from numpy import maximum
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# from transformers import set_seed
# import torch
#
# set_seed(69)
#
# def Models_Download(SEED=False, DEVICE='cpu'):
#
#     device = DEVICE
#
#     if SEED == False:
#
#         model_name_1 = "EleutherAI/gpt-neo-125M"
#         model_name_2 = "google/pegasus-large"
#
#         model_1 = GPTNeoForCausalLM.from_pretrained(model_name_1).to(device)
#         tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name_1)
#
#         model_2 = PegasusForConditionalGeneration.from_pretrained(model_name_2).to(device)
#         tokenizer_2 = PegasusTokenizer.from_pretrained(model_name_2)
#
#         return model_1, tokenizer_1, model_2, tokenizer_2
#
# def genText(title="", cat="", phrase="", sub_category="", DEVICE="cpu", top_p=0, top_k=40, temperature=0.75, repitition_penalty=1.25):
#
#     device = DEVICE
#     temp = temperature
#     topP = top_p
#     topK = top_k
#     repPenal = repitition_penalty
#
#     seed_sentence = (title + phrase + cat + sub_category)
#
#     model_1, tokenizer_1, model_2, tokenizer_2 = Models_Download(SEED=False)
#
#     batch = tokenizer_2(seed_sentence, truncation=True,
#                         return_tensors="pt").to(device)
#
#     translated = model_2.generate(**batch)
#
#     tgt_text = tokenizer_2.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#
#     print(tgt_text)
#
#     input_ids = tokenizer_1(tgt_text,
#                             return_tensors="pt",
#                             add_special_tokens=True,
#                             is_split_into_words=True,
#                             max_length=520, truncation=True).input_ids
#
#     gen_tokens = model_1.generate(input_ids,
#                                   do_sample=True,
#                                   temperature=temp,
#                                   max_length=850,
#                                   min_length=500,
#                                   repitition_penalty=repPenal,
#                                   top_k=topK,
#                                   top_p=topP,
#                                   num_return_sequences=5,
#                                   no_repeat_ngram_size=6,
#                                   vocab_size=50256,
#                                   num_layers=24,
#                                   num_head=36,
#                                   intermediate_size=8192,
#                                   max_position_embeddings=4096,
#                                   layer_norm_epsilon=1e-5, truncation=True)
#
#     gen_text = tokenizer_1.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
#
#     # print(gen_text)
#
#     return str(gen_text)
#
#
# def GPT2MainFunc(X_title, X_sub_category, X_category, X_phrase, DEVICE="cpu", top_p=0, top_k=40, temperature=0.75, repitition_penalty=1.25):
#
#     device = DEVICE
#     X_T = X_title
#     X_S_C = X_sub_category
#     X_C = X_category
#     X_P = X_phrase
#     TOP_P = top_p
#     TOP_K = top_k
#     TEMP = temperature
#     REP_PENAL = repitition_penalty
#
#     first_out_put_without_preprocessing = genText(title=X_title,cat=X_category,phrase=X_phrase,sub_category=X_sub_category,
#                                                   DEVICE=device, top_p=TOP_P, top_k=TOP_K, temperature=TEMP, repitition_penalty=REP_PENAL)
#
#     def final_preprocess_function(seed=first_out_put_without_preprocessing, DEVICE='cpu', delim=".", randomS=0):
#
#         delim = delim
#         randomS = randomS
#         device = DEVICE
#
#         string = seed
#
#         delimiter = delim
#
#         string_list = string.rsplit(delimiter,maxsplit = 1)
#
#         return (string_list[randomS]+delim)
#
#     return final_preprocess_function()
#
# def GPT2MultipleGenerationFunction(X_title_1="", X_sub_category_1="", X_category_1="", X_phrase_1="",
#                                    DEVICE="cpu", temp_1=0.7, top_p_1=0.95, top_k_1=65, r_penalty_1=1.75, num_gen=3):
#
#     output_list_gen = []
#
#     controller = num_gen
#
#     while (controller > 0) :
#
#         first_gen_output = GPT2MainFunc(X_title = X_title_1,
#                                         X_sub_category = X_sub_category_1,
#                                         X_category = X_category_1,
#                                         X_phrase = X_phrase_1,
#                                         top_p = top_p_1,
#                                         top_k = top_k_1,
#                                         temperature = temp_1,
#                                         repitition_penalty = r_penalty_1)
#
#         output_list_gen.append(first_gen_output)
#
#         controller = controller - 1
#
#         if controller == 0 :
#             break
#         else :
#             continue
#
#     def Preprocess(lst_gen = output_list_gen) :
#
#         lst3 = []
#         list_output = lst_gen
#         counter = 1
#         for item in list_output :
#             lst3.append("THE AI GENERATED TEXT OUTPUT {} is :  \n\n {} \n\n".format(counter, item))
#             counter = counter + 1
#
#         return lst3
#
#     return Preprocess()
#
# def genOut(list_of_generated_text) :
#
#     STRING = ""
#
#     for items in list_of_generated_text :
#         STRING = STRING + items
#
#     return STRING
