import langchain
import huggingface_hub
import datasets

from huggingface_hub import login
from datasets import load_dataset

import os
from dotenv import load_dotenv

# Use ChatHuggingFace to set up an open-source model chatbot
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models import ChatHuggingFace

zephyr_model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
zephyr_chat = ChatHuggingFace(
    llm=zephyr_model
)

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from the environment variable
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

login(token=HF_TOKEN)  

dataset = load_dataset("RCODI/chatbot-conv")

print(dataset.type())

from langchain import PromptTemplate, FewShotPromptTemplate
import ast

def chat_prompt(messages, demo_template, prefix, suffix):

    # Define a prompt template for the demonstrations.
    demo_prompt = PromptTemplate(
        input_variables=["user", "assistant"],
        template=demo_template,
    )

    examples = []
    user_entry = None
    assistant_messages = []

    for turn in messages:
        # Skip system messages
        if turn['role'] == 'system':
            continue

        if turn['role'] == 'user':
            # If encountering a user message, process accumulated assistant messages first
            if assistant_messages:
                # Combine assistant messages and add to examples
                examples.append({'user': user_entry, 'assistant': ' '.join(assistant_messages)})
                assistant_messages = []  # Reset assistant messages list
            # Update the user entry with the current message
            user_entry = turn['content']
        else:
            # Accumulate assistant messages
            assistant_messages.append(turn['content'])

    # After the loop, check if there are unprocessed assistant messages
    if assistant_messages:
        if user_entry:
            examples.append({'user': user_entry, 'assistant': ' '.join(assistant_messages)})
        else:
            # Handle case where there are only assistant messages at the end without a corresponding user message
            examples.append({'user': 'Continue', 'assistant': ' '.join(assistant_messages)})

    # Determine the latest message for prompt continuation
    latest_message = 'Continue' if not user_entry else user_entry

    # Prepare the few shot template
    few_shot_prompt = FewShotPromptTemplate(

        # This is the demonstration data we want to insert into the prompt.
        examples=examples,
        example_prompt=demo_prompt,
        example_separator="",

        # This is the boilerplate portion of the prompt corresponding to
        # the prompt task instructions.
        prefix=prefix[0],

        # The suffix of the prompt is where we will put the output indicator
        # and define where the "on-the-fly" user input would go.
        suffix=suffix[0],
        input_variables=["input"],
    )

    return few_shot_prompt.format(input=latest_message)

## Helper function to prepare prompt into zephyr form
def prepare_prompt_zephyr(context):

    # Prepare the few shot demonstration template
    demo_template = """<|user|>
    {user}</s>
    <|assistant|>
    {assistant}</s>
    """

    tcontext = ast.literal_eval(context)
    # This is the boilerplate portion of the prompt corresponding to
    # the prompt task instructions.
    system = ''
    for turn in tcontext:
        if turn['role'] == 'system':
            system = turn['content']
    prefix = "<|user|>\n" + system + "</s>\n",

    # The suffix of the prompt is where we will put the output indicator
    # and define where the "on-the-fly" user input would go.
    suffix="<|user|>\n{input}</s>\n<|assistant|>\n",

    return chat_prompt(
        tcontext,
        demo_template,
        prefix,
        suffix
    )

def get_zephyr_response(context):
    result = zephyr_chat.invoke(context)
    return result

def PromptandMetrics(row, model='llama'):
    context = row['Context']
    baseline = row['Baseline Response']
    if baseline is None:
        print('Empty baseline')
        baseline = 'Thank you.'
    if model == 'zephyr':
        response = get_zephyr_response(context)
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
    return response

init=300
batch_length=5
sub_data = dataset.select()
responses = []
responses = sub_data.map(lambda x: PromptandMetrics(x, model='zephyr'))