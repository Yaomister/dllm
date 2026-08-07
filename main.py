import re
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from math_verify import parse, verify
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from schedulers import CosScheduler, LinearScheduler, ConstantScheduler, InverseScheduler, scheduler 


methods = [
    CosScheduler, LinearScheduler, ConstantScheduler, InverseScheduler, None
]

def add_gumbel_noise(logits, temp):

    # use gumbel's maximum instead of sampling from a probability distribution

    # the idead is we add a random amount of noise (kick) to each value and take the argmax, the maxium value after the noise is equivallent to drawing from a softmax distribution.
    if temp == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype= torch.float32)
    gumbel_noise = (-torch.log(-torch.log(noise))) * temp
    return logits + gumbel_noise


def get_num_transfer_tokens(mask_index, steps):

    # count the number of mask tokens left
    mask_num = mask_index.sum(dim=1, keepdim = True)


    # the number of mask tokens to fill per step
    base = mask_num // steps

    # if there is an uneven amount
    remainder = mask_num % steps

    # a tensor where the value at each position represents the number of tokens you need to fill at each step
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    # fill the remainders
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def contains_token_squence(tokens, sequence):
    # checks whether a shorter sequence of tokens exists in a longer sequence of tokens
    if sequence.numel() == 0 or tokens.numel() < sequence.numel():
        return False
    for start in range(tokens.numel() - sequence.numel() + 1):
        if torch.equal(tokens[start:start + sequence.numel()], sequence):
            return True
    return False


def get_next_sequence_token_id(tokens, position, sequence, context_start):
    # when we're halkway through writing a target phrase, this gets the next token in it
    max_prefix_length = min(sequence.numel() - 1, position - context_start)
    for prefix_length in range(max_prefix_length, 0, -1):
        if torch.equal(tokens[position - prefix_length:position], sequence[:prefix_length]):
            return sequence[prefix_length].item()
    return sequence[0].item()

def apply_end_think_logit_boost(logits, tokens, candidate_mask_index, context_start, total_gen_length, end_think_token_ids = None, end_think_logit_boost=0., end_think_boost_power = 2.0):

    # if the feature isnt turned on, do nothing
    if end_think_logit_boost <= 0 or not end_think_token_ids or total_gen_length <= 0:
        return logits
    if any(token_id <0 or token_id >= logits.shape[-1] for token_id in end_think_token_ids):
        return logits

    sequence = torch.tensor(end_think_token_ids, dtype=torch.long, device=logits.device)
    logits = logits.clone()

    # loop over each sequence in the batch
    for batch_index in range(tokens.shape[0]):
        # for this sequence, check if the stop-phrase has already been written
        if contains_token_squence(tokens[batch_index, context_start:], sequence):
            continue

        candidate_position = torch.nonzero(
            candidate_mask_index[batch_index], as_tuple=False
        ).flatten()

        # find all the still-blank positions
        if candidate_position.numel() == 0:
            continue
        
        position = candidate_position[0].item()

        generated_length = position - context_start + 1

        progress = min(max(generated_length / float(total_gen_length), 0.), 1.)

        boost = end_think_logit_boost * (progress ** end_think_boost_power)

        token_id = get_next_sequence_token_id(
            tokens[batch_index], position, sequence, context_start 
        )

        logits[batch_index, position, token_id] += boost

    return logits


@torch.no_grad()
def generate(model, prompt, scheduler, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
             confidence_eos_eot_inf=False, end_think_token_ids=None, end_think_logit_boost=0.,
             end_think_boost_power=2., end_think_context_start=None,
             end_think_total_gen_length=None):
    # temperature = T_token, remasting = T_pos, but they only offer two extremes for T_token, either greedy, or random

    base_model = getattr(model, "module", model)

    model_name = getattr(getattr(base_model, 'config', None), "_name_or_path", "")

    # batch size can only be 1
    if 'illada' in model_name.lower():
        assert prompt.shape[0] == 1, 'iLLaDA currently does not support padded batch generation.'

    # fill the entire thing with mask tokens
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype= torch.long).to(model.device)

    # paste the prompt in front of all the masked tokens
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        # make the model tend to all positions
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    # where the prompt is
    prompt_index = (x != mask_id)

    # make sure the generation length is divisible by the block length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # if we dont know when to start generating, just make it when the promp ends
    if end_think_context_start is None:
        end_think_context_start = prompt.shape[1]

    # if not given, just make it the generation length
    if end_think_total_gen_length is None:
        end_think_total_gen_length = gen_length

    # for each chunk we're generating
    for num_block in range(num_blocks):
        # this picks out which sections of the block is still blank
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        # the number of blanks in the block and splits them across steps
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # loop over the denoising step within the current_block
        for i in range(steps):

            # find the positions in the whole squeence that are still blank
            mask_index = (x == mask_id)

            # the boundaries of the current block
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            # restricts the blank-set to only this block. So basically, it is set to True only the still-blank positions inside the current block
            candidate_mask_index = mask_index.clone()
            candidate_mask_index[:, :block_start] = False
            candidate_mask_index[:, block_end:] = False

            # the forward pass
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)

                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim = 0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits


            # nudge the logits towards not thinking anymore
            logits = apply_end_think_logit_boost(
                logits,
                x,
                candidate_mask_index,
                end_think_context_start,
                end_think_total_gen_length,
                end_think_token_ids=end_think_token_ids,
                end_think_logit_boost=end_think_logit_boost,
                end_think_boost_power=end_think_boost_power
            )

            # forbut the model from selecting end of sequence tokens
            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            # add the gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temp=temperature)

            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
            
            if remasking == "margin":
                p = F.softmax(logits, dim=-1)
                top_two = torch.topk(p, 2, dim=-1).values
                # assign real confidence scores
                x0_p = top_two[..., 0] - top_two[..., 1]
            elif remasking == 'scheduler':
                # assign random confidence scores
                p = F.softmax(logits, dim=-1)
                x0_p = p.max(dim=-1).values 
            else:
                raise NotImplementedError(remasking)

            # prevents future blocks from being selected
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf


            # picks x0 if mask_index is true, x if mas_index is false (predictions for blank, real tokens everywhere else)
            x0 = torch.where(mask_index, x0, x)
            # same trick for confidences
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            position_temperature = scheduler.get_temperature(num_block * steps + i) if remasking == 'scheduler' else None

            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i].item()
                # grab the k highest confidence positions, this is the greedy selection
                conf_j = confidence[j]
          
                valid_indexes = torch.isfinite(conf_j).nonzero().flatten()
                k = min(k, valid_indexes.numel())
                if remasking == "margin":
                    _, s = torch.topk(conf_j[valid_indexes], k=k, largest=False)
                    selected_index = valid_indexes[s]
                elif remasking == "scheduler":
                    if (position_temperature <= 1e-6):
                        _, selected_index = torch.topk(conf_j, k)
                    else:
                        w = conf_j[valid_indexes] ** (1 / position_temperature)
                        w = torch.nan_to_num(w)
                        if (w.sum() == 0):
                            _, selected_index = torch.topk(conf_j, k = k)
                        else:
                            w = w / w.sum()
                            picked = torch.multinomial(w, k, replacement=False)
                            selected_index  = valid_indexes[picked]
                    
                transfer_index[j, selected_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

            
def calculate_pass_k(n, k, c):
    # n = how many samples you generate
    # c = how many of the samples came out correct
    # k = how many attemps we're asking about

    # not enough failues to fill a subset, so guarenteed hit
    if n - c < k:
        return 1

    return 1 - np.prod(1 - k / np.arange(n - c + 1, n + 1))


def grade(prediction, answer, dataset_name):
    if dataset_name == "math":
        return verify(parse(prediction), parse(answer))
    else:
        correct_answer = answer.split("####")[-1].strip().replace(",", "")
        hit = re.findall(r"-?\d+", prediction.replace(",", ""))
        return hit and hit[-1] == correct_answer


def main(seed, dataset_name):

    print("CUDA available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    print("torch built for CUDA:", torch.version.cuda)

    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    model = AutoModel.from_pretrained(pretrained_model_name_or_path="GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test") if dataset_name == "math" else load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(100))

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    total_steps = 256

    results = {
                "config": {"model": "LLaDA-8B-Instruct", "N": 8,
                   "T_token": 0.8, "block_length": 32, "steps": 256},
                "method": {
    
                }
            }

    N = 8
    BATCH = 8    

    print(f"Starting experiments")
    for method in methods:
        samples = [[] for _ in range(N)]
        for start in range(0, len(dataset), BATCH):
            chunk = dataset.select(range(start, min(len(dataset), start + BATCH)))
            field = "problem" if dataset_name == "math" else "question"
            messages = [{"role": "user", "content": row[field]} for row in chunk]            
            prompts = [
                tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False) for m in messages
            ]
            assert tokenizer.pad_token_id != 126336
            encoded_outputs = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )

            print('experiment is running...')
        
            input_ids = encoded_outputs['input_ids'].to(device)
            attention_mask = encoded_outputs['attention_mask'].to(device)
            for n in range(N):
                scheduler_obj = method(1, 0, total_steps) if method is not None else None
                out = generate(model, input_ids, scheduler_obj, attention_mask, steps=total_steps, gen_length=256, block_length=32, temperature=0.8, cfg_scale=0., remasking='margin' if method is None else "scheduler")
                output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
                samples[n].extend(output)
            torch.cuda.empty_cache()
            print(f"  done problems {start}-{start+len(chunk)-1}", flush=True)

        for p in range(len(dataset)):
            print(f"\nproblem {p}:")
            for n in range(N):
                print(f"  sample {n}: {samples[p][n] if False else samples[n][p]}")
            print('-' * 50)

        K = [1, 2, 4, 8]


        C = []
        for i in range(len(dataset)):
            ground_truth_answer = dataset[i]['answer']
            inference_answer = [grade(samples[n][i], ground_truth_answer, dataset_name=dataset_name) for n in range(N)]
            c = sum(inference_answer)
            C.append(c)

        for _ in K:
            results[method.__name__ if method else "margin"] =  {
                "c_counts" : C,
                "pass_k" :{k : float(np.average([calculate_pass_k(N, k, c) for c in C])) for k in K}
            }

    with open(f"results_{seed}.json", 'w') as f:
        json.dump(results, f, indent=2)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    main(args.seed, args.dataset)