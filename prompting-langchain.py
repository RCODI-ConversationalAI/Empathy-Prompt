import langchain
import huggingface_hub
import datasets

from huggingface_hub import login
from datasets import load_dataset

import os
from dotenv import load_dotenv

# Use ChatHuggingFace to set up an open-source model chatbot
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from the environment variable
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

login(token=HF_TOKEN)  

zephyr_model = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=1000,
        do_sample=True,
        top_k=10,
        top_p=0.7,
        temperature=0.7,
        repetition_penalty=1.18,
    ),
)
zephyr_chat = ChatHuggingFace(
    llm=zephyr_model
)

dataset = load_dataset("RCODI/chatbot-conv")

data = dataset['train']

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

def prepare_prompt_zephyr(context):
    messages = [
        SystemMessage(content=context[0].content),
    ]
    for i in range(len(context)):
        if i%2 == 1:
            messages.append(HumanMessage(content=context[i].content))
        else:
            messages.append(AIMessage(content=context[i].content))
    return messages

def get_zephyr_response(context):
    responses = zephyr_chat.invoke(context)
    result = responses.content
    return result

def PromptandMetrics(row):
    context = row['Context']
    baseline = row['Baseline Response']
    if baseline is None:
        print('Empty baseline')
        baseline = 'Thank you.'
    messages = prepare_prompt_zephyr(context)
    response = get_zephyr_response(messages)
#   elif model == 'llama':
#     response = get_llama_response(context)
#   elif model == 'neural':
#     response = get_neural_response(context)

#   # text similarity
#   bleu_score = get_bleu_score(response, baseline)
#   rouge_score = get_rouge_score(response, baseline)
#   leven_distance = get_leven_distance(response, baseline)
#   text_similarity = (bleu_score + rouge_score + leven_distance)/3

#   # semantic similarity
#   cos_sim = get_cosine_similarity(response, baseline, semantic_model)

#   # factual consistency
#   nli_score = get_nli_result(response, baseline, nli_pipeline)

#   # sentiment label
#   response_sent_score = get_sentiment_score(response, sentiment_pipeline)
#   baseline_sent_score = get_sentiment_score(baseline, sentiment_pipeline)

#   # empathy rules
#   sia = SentimentIntensityAnalyzer()
#   response_length = count_sentences(response) + 1
#   response_person_form_score = score_person_form(response)/response_length
#   response_pronoun_score = score_pronouns(response)/response_length
#   response_tense_score = score_tense(response)/response_length
#   response_exclamations_score = score_exclamations(response)/response_length
#   response_stimulating_score = score_stimulating_dialogue(response)/response_length
#   response_acknowledging_score = score_acknowledging(response)/response_length
#   response_collective_reasoning_score = score_collective_reasoning(response)/response_length
#   response_imperative_socre = score_imperative_statements(response)/response_length
#   response_interim_q_score = score_interim_questioning(response)/response_length
#   response_caring_statement_score = score_caring_statements(response)
#   baseline_length = count_sentences(baseline) + 1
#   baseline_person_form_score = score_person_form(response)/baseline_length
#   baseline_pronoun_score = score_pronouns(response)/baseline_length
#   baseline_tense_score = score_tense(response)/baseline_length
#   baseline_exclamations_score = score_exclamations(response)/baseline_length
#   baseline_stimulating_score = score_stimulating_dialogue(response)/baseline_length
#   baseline_acknowledging_score = score_acknowledging(response)/baseline_length
#   baseline_collective_reasoning_score = score_collective_reasoning(response)/baseline_length
#   baseline_imperative_socre = score_imperative_statements(response)/baseline_length
#   baseline_interim_q_score = score_interim_questioning(response)/baseline_length
#   baseline_caring_statement_score = score_caring_statements(response)

#   # Return a dictionary with new columns
#   return {"response": response,
#           "text similarity": text_similarity,
#           "BLEU score": bleu_score,
#           "ROUGE score": rouge_score,
#           "LEVEN distance": leven_distance,
#           "semantic similarity": cos_sim,
#           "factual consistency": nli_score,
#           "LLM response length": response_length,
#           "LLM sentiment": response_sent_score,
#           "LLM person form": response_person_form_score,
#           "LLM pronoun": response_pronoun_score,
#           "LLM tense": response_tense_score,
#           "LLM exclamation": response_exclamations_score,
#           "LLM stimulating dialogue": response_stimulating_score,
#           "LLM acknowledging": response_acknowledging_score,
#           "LLM collective reasoning": response_collective_reasoning_score,
#           "LLM imperative statement": response_imperative_socre,
#           "LLM interim questions": response_interim_q_score,
#           "LLM caring statement": response_caring_statement_score,
#           "baseline response length": baseline_length,
#           "baseline sentiment": baseline_sent_score,
#           "baseline person form": baseline_person_form_score,
#           "baseline pronoun": baseline_pronoun_score,
#           "baseline tense": baseline_tense_score,
#           "baseline exclamation": baseline_exclamations_score,
#           "baseline stimulating dialogue": baseline_stimulating_score,
#           "baseline acknowledging": baseline_acknowledging_score,
#           "baseline collective reasoning": baseline_collective_reasoning_score,
#           "baseline imperative statement": baseline_imperative_socre,
#           "baseline interim questions": baseline_interim_q_score,
#           "baseline caring statement": baseline_caring_statement_score}
    print(response)
    return {"response": response}

init=300
batch_length=5
sub_data = data.select(range(init, init+batch_length))
responses = sub_data.map(lambda x: PromptandMetrics(x))