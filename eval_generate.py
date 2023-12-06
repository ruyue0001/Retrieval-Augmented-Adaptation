import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from typing import List, Union
from tqdm import tqdm
from datasets import load_dataset, Dataset
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
# from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

alpaca = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"
}


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = alpaca
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def main(
    load_8bit: bool = True,
    base_model: str = "yahma/llama-7b-hf",
    lora_weights: str = "",
    eval_path : str = "",
    eval_result_path : str = "",
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
    eval_batch_size: int = 3,
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    print(prompt_template)
    prompter = Prompter(prompt_template)

    # Setting tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:", bos, eos, pad, "   => It should be 1,2,none")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference

    # Setting model
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
        print ("Load Lora from:", lora_weights)
    model.print_trainable_parameters()
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # # Process inference data
    # if eval_path.endswith(".json") or eval_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=eval_path)
    # else:
    #     data = load_dataset(eval_path)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors="pt",
        )
        # if (
        #         result["input_ids"][-1] != tokenizer.eos_token_id
        #         and len(result["input_ids"]) < cutoff_len
        #         and add_eos_token
        # ):
        #     result["input_ids"].append(tokenizer.eos_token_id)
        #     result["attention_mask"].append(1)
        # result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(ins, input):
        full_prompt = prompter.generate_prompt(
            ins,
            input,
        )
        tokenized_full_prompt = tokenize(full_prompt)

        return tokenized_full_prompt

    # val_data = data.map(generate_and_tokenize_prompt)

    def evaluate(
        instructions,
        inputs,
        temperature=0,
        top_p=0.75,
        top_k=20,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        # prompt = prompter.generate_prompt(instruction, input)
        # inputs = tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(device)
        prompts = [prompter.generate_prompt(ins, input) for ins, input in zip (instructions, inputs)]
        encodings = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_outputs = model.generate(
                **encodings,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
            )
        # print (generation_output)

        return tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
        # gens = [tokenizer.decode(gen).split("### Response:")[1].strip().split('\n\n')[0] for gen in generation_output]
        # print (gens)
        # return gens

        # s = generation_output.sequences[0]
        # output = tokenizer.decode(s)
        # output = output.split('### Response:')[1].strip()
        # # print (output)
        # results.append(output)
        # return output
        # yield prompter.get_response(output)

    # gr.Interface(
    #     fn=evaluate,
    #     inputs=[
    #         gr.components.Textbox(
    #             lines=2,
    #             label="Instruction",
    #             placeholder="Tell me about alpacas.",
    #         ),
    #         gr.components.Textbox(lines=2, label="Input", placeholder="none"),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.1, label="Temperature"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.75, label="Top p"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=100, step=1, value=40, label="Top k"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=4, step=1, value=4, label="Beams"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
    #         ),
    #         gr.components.Checkbox(label="Stream output"),
    #     ],
    #     outputs=[
    #         gr.inputs.Textbox(
    #             lines=5,
    #             label="Output",
    #         )
    #     ],
    #     title="🦙🌲 Alpaca-LoRA",
    #     description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    # ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # # Old testing code follows.


    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    #     print("Instruction:", instruction)
    #     print("Response:", evaluate(instruction))
    #     print()
    #
    # print (results)

    # Test
    # instructions = ["Please identify all entities from the input text. If there is no entity, please output None.\n",
    #                 "Please identify all entities from the input text. If there is no entity, please output None.\n",
    #                 "Please identify all entities from the input text. If there is no entity, please output None.\n",
    #                 "Please identify all entities from the input text. If there is no entity, please output None.\n"
    # ]
    # inputs = ["Physical mapping 220 kb centromeric of the human MHC and DNA sequence analysis of the 43 - kb segment including the RING1 , HKE6 , and HKE4 genes .",
    #           "The synthesis of endo - adduct [ 4aS , 5S , 8R , 8aR , SS ] - 9d resulting from cycloaddition on the substituted C ( 2 ) - C ( 3 ) double bond was achieved in a chemo - and diastereoselective way from quinone 1d in the presence of ZnBr ( 2 ) .",
    #           "DNA elements recognizing NF - Y and Sp1 regulate the human multidrug - resistance gene promoter .",
    #           "None of the following actions may be undertaken by the Servicer without the prior written consent of the Company : 4 ( i ) modify , amend or waive in any respect whatsoever ( A ) the interest rate , monthly payment , or other monetary or economic provisions ( including with respect to the date or time upon which any obligations are due ) of the Loan , including to defer interest payments ; ( B ) any provision in the Loan Documents that restricts Borrower from incurring additional indebtedness ; or ( C ) any other provisions in the Loan Documents other than non - monetary , non - economic or administrative amendments or modifications which the Servicer believes in good faith and in accordance with Accepted Loan Servicing Practices will not in any material and adverse way affect any Holder ' s rights under this Agreement , the Loan Documents or the value of the Property ; ( ii ) waive or reduce the amount of any reserves required to be maintained by Borrower , except as explicitly permitted by the Loan Agreement ; ( iii ) modify the principal amount of the Loan ; ( iv ) extend or shorten the maturity date of the Loan or any note , other than in accordance with the express provisions of the Loan Agreement ; ( v ) waive , compromise or settle any material claim against Borrower or any or other Person liable for payment of the Loan in whole or in part or for the observance and performance by Borrower of any of the terms , covenants , provisions and conditions of the Loan Documents , or release Borrower or any other Person liable for payment of the Loan in whole or in part from any obligation or liability under the Loan Documents ; ( vi ) approve or consent to a Borrower Transfer ; ( vii ) encumber , release , or modify , in whole or in part , any collateral or security interest held under the Loan Documents other than in accordance with the terms hereof or any of the express provisions of the Loan Agreement ; ( viii ) enforce or refrain from enforcing all of the rights , remedies and privileges afforded or available to the respective Holders under the terms of the Loan Documents , including , without limitation , accelerating the Loan ( unless such acceleration is automatic under the Loan Documents ), foreclosing on any mortgage or pledge or accepting a deed in lieu of foreclosure ; ( ix ) following a foreclosure of the Mortgage or any pledge or acceptance of a deed in lieu of foreclosure , approve a recommended course of action for the Property , approve the property manager and selling agent , and approve the sale price of the Property ; ( x ) the approval or adoption of any plan of bankruptcy , reorganization , restructuring or similar event in an Insolvency Proceeding with respect to the Borrower or any guarantor ; ( xi ) any incurrence of additional debt by the Borrower or any mezzanine financing by any direct or indirect beneficial owner of the Borrower ( to the extent that the Lender has consent rights pursuant to the Loan Documents with respect thereto ); 5 ( xii ) any waiver of the enforcement of any insurance requirements under the Loan Documents with respect to terrorism , earthquake , flooding , windstorm or political risk ; ( xiii ) any material amendment to the special purpose entity provisions in the Loan Agreement ; ( xiv ) the subordination of any mortgage or pledge to any other mortgage or pledge or other material monetary claim against the Property ; or ( xv ) waiver of any material default or Event of Default ."
    # ]

    # outputs = evaluate(instructions, inputs)
    # print (outputs)
    # for output in outputs:
    #     response = output.split('### Response:')[1].strip()
    #     print (response)


    print("Evaluate from: ", eval_path)
    print ("Evaluate result path:", eval_result_path)
    print("Eval batch size:", eval_batch_size)
    with open(eval_result_path, "w") as r:
        with open(eval_path) as f:
            eval_data = json.load(f)
            instructions = []
            inputs= []
            for i in tqdm(range(0, len(eval_data))):
                instructions.append(eval_data[i]['instruction'])
                tmp = eval_data[i]['input']
                if len(tmp.split()) > 300:
                    inputs.append(' '.join(tmp.split()[:300]))
                else:
                    inputs.append(tmp)

                if (i+1) % eval_batch_size == 0:
                    outputs = evaluate(instructions, inputs)
                    for k in range(0, len(outputs)):
                        output = outputs[k]
                        response = output.split('### Response:')[1].strip()
                        response = response.replace('\n', '')
                        print ("Input:", inputs[k])
                        print ("Response:", response)
                        r.write(response + '\n')
                    instructions = []
                    inputs = []
                if i == len(eval_data) - 1 and instructions != []:
                    outputs = evaluate(instructions, inputs)
                    for k in range(0, len(outputs)):
                        output = outputs[k]
                        response = output.split('### Response:')[1].strip()
                        response = response.replace('\n', '')
                        print("Input:", inputs[k])
                        print("Response:", response)
                        r.write(response + '\n')

    f.close()
    r.close()



if __name__ == "__main__":
    fire.Fire(main)
