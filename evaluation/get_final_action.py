import argparse
import json
import os
import random
import sys
import time

import numpy as np
import openai
import pandas as pd
# import torch  # Removed - local model dependencies not needed
from dotenv import load_dotenv
from procoder.functional import (
    add_refnames,
    format_multiple_prompts,
)
from tqdm import tqdm
# from transformers import AutoTokenizer  # Removed - local model dependencies not needed

sys.path.append('../data_construction')
from toolemu.prompts.agent import *
from toolemu.tools import get_toolkits_by_names

sys.path.append('../helper')
from utils import openai_chat_completion_with_retry, print_api_usage, claude_chat_completion_with_retry, gemini_chat_completion_with_retry
# from utils import VLLM  # Removed - local model dependencies not needed


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path of the data in json format.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the probing results in csv format.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the cases to evaluate.')
    parser.add_argument('--num', type=int, default=1,
                        help='Number of cases to evaluate. If -1, evaluate all remaining cases.')
    parser.add_argument('--specific-case-name', type=str, default=None,
                        help='If not None, only evaluate the case with the given name.')
    parser.add_argument('--prompt-type', type=str,
                        choices=['naive', 'privacy_enhanced', 'privacy_prompt_by_zach'],
                        help='The type of the prompt to use for the agent. Options: naive (original paper baseline), privacy_enhanced (original paper), privacy_prompt_by_zach (enhanced by Zach with explicit privacy requirements)')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use for the agent. Supports GPT models (e.g., gpt-4-1106-preview), Claude models (e.g., claude-sonnet-4-5), and Gemini models (e.g., gemini-2.5-flash).')
    # GPU arguments removed - no longer needed for API-based models
    # parser.add_argument('--gpu-num', type=int, default=1, help='Number of GPUs to use for VLLM.')
    # parser.add_argument('--hf-cache-dir', type=str, help='The cache directory for the Hugging Face model.')

    return parser.parse_args()


def prepare_agent_prompt(
        prompt_type,
        user_name,
        user_email,
        user_instruction,
        toolkits,
        executable_trajectory,
        final_action
):
    toolkits = get_toolkits_by_names(toolkits)
    all_tools = []
    for toolkit in toolkits:
        all_tools += toolkit.tools

    toolkit_strings = "\n".join(
        [toolkit.create_description("medium") for toolkit in toolkits]
    )
    tool_names = ", ".join([tool.name for tool in all_tools])

    inputs = dict(toolkit_descriptions=toolkit_strings, tool_names=tool_names)

    add_refnames(AGENT_DUMMY_VARS, inputs, include_brackets=False)

    if prompt_type == 'naive':
        system_info = AGENT_NAIVE_SYSTEM_INFO
        prompt_instruction = AGENT_NAIVE_PROMPT
    elif prompt_type == 'privacy_enhanced':
        system_info = AGENT_PRIVACY_ENHANCED_SYSTEM_INFO
        prompt_instruction = AGENT_PRIVACY_ENHANCED_PROMPT
    elif prompt_type == 'privacy_prompt_by_zach':
        system_info = AGENT_PRIVACY_PROMPT_BY_ZACH_SYSTEM_INFO
        prompt_instruction = AGENT_PRIVACY_ENHANCED_PROMPT  # Use same instruction format, just different system prompt
    else:
        raise ValueError(f'[Error] Unknown prompt type: {prompt_type}.')

    system_info, prompt_instruction = format_multiple_prompts(
        [system_info, prompt_instruction], inputs, include_brackets=[False, True]
    )

    prompt_instruction = prompt_instruction.format(
        user_name=user_name,
        user_email=user_email,
        input=user_instruction,
        agent_scratchpad=executable_trajectory
    )

    final_prompt = system_info + '\n\n' + prompt_instruction

    final_prompt = final_prompt[:final_prompt.find('## Start the Execution')]
    final_prompt += ('## Start the Execution\n'
                     f'User Input: {user_instruction}\nPast Trajectory:\n{executable_trajectory}\n\n'
                     f'Please output your **NEXT** [Action]/[Action Input] following the provided [Scratchpad], '
                     f'directly start your response with your [Thought] for the current iteration. '
                     f'The [Action] should be {final_action}.')

    return final_prompt


def post_process(s):
    if s.startswith('<|start_header_id|>assistant<|end_header_id|>'):
        # Remove the assistant header for Llama-3.
        s = s[len('<|start_header_id|>assistant<|end_header_id|>'):].strip()
        s = s[:s.find('<|eot_id|>')]
    # Only keep the Action and Action Input.
    if 'Observation:' in s:
        s = s[:s.find('Observation:')]
    if 'Final Answer:' in s:
        s = s[:s.find('Final Answer:')]
    if '}' in s:
        s = s[:s.find('}') + 1]
    return s


@print_api_usage
def main():
    seed = 0
    # torch.manual_seed(seed)  # Removed - local model dependencies not needed
    np.random.seed(seed)
    random.seed(seed)

    args = prepare_args()
    load_dotenv()

    if 'gpt' in args.model:
        openai.api_key = os.environ['OPENAI_API_KEY']
        openai.api_type = os.environ['OPENAI_API_TYPE']
        if openai.api_type == 'azure':
            openai.api_base = os.environ['OPENAI_API_BASE']
            openai.api_version = os.environ['OPENAI_API_VERSION']

    # Local model support removed - use API-based models (GPT or Claude) instead
    # if 'mistral' in args.model or 'llama' in args.model or 'zephyr' in args.model or 'vicuna' in args.model:
    #     vllm_engine = VLLM(
    #         model=args.model,
    #         tensor_parallel_size=args.gpu_num,
    #         trust_remote_code=True,
    #         max_num_batched_tokens=16000,
    #         max_new_tokens=1000,
    #         temperature=0,
    #         download_dir=args.hf_cache_dir
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache_dir)

    with open(args.input_path, 'r') as f:
        data = json.load(f)

    if args.specific_case_name:
        for i, case in enumerate(data):
            if case['name'] == args.specific_case_name:
                args.start_index = i
                end_index = i + 1
                break
        else:
            raise ValueError(f'Error: The specific case name {args.specific_case_name} is not found.')
    else:
        if args.num == -1:
            end_index = len(data)
        else:
            end_index = min(args.start_index + args.num, len(data))

    result = {
        'name': [],
        'final_action': [],
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Save progress every 10 items
    save_interval = 10

    # Track refusals
    refusal_count = 0

    for i in tqdm(range(args.start_index, end_index)):
        result['name'].append(data[i]['name'])
        agent_prompt = prepare_agent_prompt(
            prompt_type=args.prompt_type,
            user_name=data[i]['trajectory']['user_name'],
            user_email=data[i]['trajectory']['user_email'],
            user_instruction=data[i]['trajectory']['user_instruction'],
            toolkits=data[i]['trajectory']['toolkits'],
            executable_trajectory=data[i]['trajectory']['executable_trajectory'],
            final_action=data[i]['trajectory']['final_action']
        )

        # Debug: Print prompt info
        prompt_length = len(agent_prompt)
        print(f"[INFO] Case {i} (name: {data[i]['name']}) - Prompt length: {prompt_length} chars")

        # Retry logic for API calls (max 3 attempts)
        final_action = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if 'gemini' in args.model:
                    response = gemini_chat_completion_with_retry(
                        engine=args.model, messages=[{'role': 'user', 'content': agent_prompt}])
                    final_action = response.text.strip()
                elif 'gpt' in args.model:
                    response = openai_chat_completion_with_retry(
                        engine=args.model, messages=[{'role': 'user', 'content': agent_prompt}],
                        max_tokens=400, temperature=0.0)
                    final_action = response.choices[0].message['content'].strip()
                elif 'claude' in args.model:
                    response = claude_chat_completion_with_retry(
                        engine=args.model, messages=[{'role': 'user', 'content': agent_prompt}],
                        max_tokens=400, temperature=0.0)

                    # Check for refusal first (safety filter)
                    stop_reason = response.stop_reason if hasattr(response, 'stop_reason') else 'unknown'

                    if stop_reason == 'refusal':
                        print(f"\n[SKIP] Case {i} (name: {data[i]['name']}) - Claude refused to respond (safety filter)")
                        final_action = "[REFUSAL]"  # Mark as refusal
                        refusal_count += 1
                        break  # Skip retries, continue to next case

                    # Debug: Print raw response details for non-refusal cases
                    if not (response.content and len(response.content) > 0):
                        print(f"\n[DEBUG] Case {i} - Raw Claude response:")
                        print(f"  Response type: {type(response)}")
                        print(f"  Response ID: {response.id if hasattr(response, 'id') else 'N/A'}")
                        print(f"  Stop reason: {stop_reason}")
                        print(f"  Content length: {len(response.content) if hasattr(response, 'content') else 'N/A'}")
                        if hasattr(response, 'content') and response.content:
                            print(f"  Content blocks: {len(response.content)}")
                            for idx, block in enumerate(response.content):
                                print(f"    Block {idx}: type={type(block)}, text_len={len(block.text) if hasattr(block, 'text') else 'N/A'}")

                    if response.content and len(response.content) > 0:
                        final_action = response.content[0].text.strip()
                    else:
                        raise ValueError(f"Empty response from Claude API. Stop reason: {stop_reason}")
                else:
                    raise NotImplementedError(f"Model {args.model} not supported")

                # If we got a valid response, break out of retry loop
                if final_action:
                    break

            except Exception as e:
                print(f"\n[ERROR] Attempt {attempt + 1}/{max_retries} failed for case {i} (name: {data[i]['name']})")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {e}")
                import traceback
                print(f"  Traceback:\n{traceback.format_exc()}")

                if attempt == max_retries - 1:
                    # On final attempt, raise exception to fail the script
                    raise RuntimeError(f"Failed to get valid response for case {i} (name: {data[i]['name']}) after {max_retries} attempts. Last error: {e}")
                else:
                    # Wait a bit before retrying
                    print(f"  Waiting 2 seconds before retry...\n")
                    time.sleep(2)
                    continue

        # Post-process the model output.
        result['final_action'].append(post_process(final_action))

        # Save progress periodically
        if (i + 1) % save_interval == 0:
            try:
                pd.DataFrame(result).to_csv(args.output_path, index=False)
                print(f"\nProgress saved: {i + 1 - args.start_index} items completed")
            except Exception as e:
                print(f"\nWarning: Failed to save progress: {e}")

    # Final save
    try:
        pd.DataFrame(result).to_csv(args.output_path, index=False)
    except Exception as e:
        print(f'Error: {e}')
        with open(args.output_path.replace('.csv', 'json'), 'w') as f:
            json.dump(result, f)

    # Print summary
    total_processed = end_index - args.start_index
    successful = total_processed - refusal_count
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total processed: {total_processed}")
    print(f"  Successful: {successful} ({100*successful/total_processed:.1f}%)")
    print(f"  Refusals (safety filter): {refusal_count} ({100*refusal_count/total_processed:.1f}%)")
    print(f"  Results saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
