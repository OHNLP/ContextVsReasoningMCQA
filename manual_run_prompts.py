from typing import Union

from openai import AzureOpenAI
from regex import regex
from vllm import LLM, SamplingParams
import global_values as gv

from global_values import HuggingfaceModelWrapper

model_name = 'azure/o1'
question = "A 79-year-old woman comes to the physician for the evaluation of a 2-month history of a non-productive cough and fatigue. During this period, she also has had a 4.5-kg (10-lb) weight loss and has become increasingly short of breath with mild exertion. She has congestive heart failure and hypertension. Three months ago, she was in India for 3 weeks to attend a family wedding. She worked as a seamstress in a textile factory for 50 years. She has smoked one pack of cigarettes daily for 47 years. Her current medications include enalapril, digoxin, isosorbide, spironolactone, and metoprolol. She appears thin. Her temperature is 37.0Â°C (98.6Â°F), pulse is 90/min, respirations are 25/min, and blood pressure is 110/70 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 94%. There is dullness to percussion and decreased breath sounds over the right lung base. The remainder of the examination shows no abnormalities. Laboratory studies show a glucose level of 90 mg/dL, serum lactate dehydrogenase of 227 U/L, and serum protein of 6.3 g/dL. An x-ray of the chest shows nodular pleural lesions on the right side and a moderate-sized pleural effusion. Thoracentesis shows 250 ml of turbid fluid. Analysis of the pleural fluid aspirate shows:\nErythrocyte count 1/mm3\nLeukocyte count 4,000/mm3\nGlucose 59 mg/dl\nLactate dehydrogenase 248 U/L\nProtein 3.8 g/dL\nWhich of the following is the most likely underlying cause of this patient's effusion?"
options = {'A': 'Bronchial adenocarcinoma', 'B': 'Mesothelioma', 'C': 'Nephrotic syndrome', 'D': 'Tuberculosis', 'E': 'Congestive heart failure'}

def generate_mcqa_plain():
    prompt: str = question + "\n Options: \n"
    for k, v in options.items():
        prompt = prompt + k + ': ' + v + '\n'
    messages = [
        {
            "role": "user",
            "content": "Select the appropriate option from the provided question. "
                       "Respond in the format A, B, C, D, or E. \n\n"
                       "The question:\n" + prompt
        }
    ]
    return messages

if __name__ == '__main__':
    model: Union[None, LLM, AzureOpenAI, HuggingfaceModelWrapper] = None
    if model_name.startswith('azure'):
        model = gv.get_pipeline_openai()
    elif model_name.startswith('opea'):
        model = gv.get_pipeline_deepseek(model_name)
    else:
        model = gv.get_pipeline_vllm(model_name)
    messages = generate_mcqa_plain()
    if model_name.startswith('azure'):
        try:
            val = model.chat.completions.create(
                model=model_name[6:],
                messages=messages,
                max_completion_tokens=20000
            )
            ret = val.choices[0].message.content.strip()
            print(ret)
        except Exception as e:
            print(e)

    else:
        to_model = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        val = model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6))[0].outputs[0].text
        val = regex.split('</think>', val)[1]
        val = regex.sub('</?answer>', '', val)
        print(val)