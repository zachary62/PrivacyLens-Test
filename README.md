# PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action

<p align="center">
| <a href="https://arxiv.org/abs/2409.00138"><b>Paper</b></a> | <a href="https://huggingface.co/datasets/SALT-NLP/PrivacyLens"><b>HF Dataset</b></a> | <a href="https://salt-nlp.github.io/PrivacyLens/"><b>Website</b></a> |
</p>


## Overview

<p align="center">
  <img src="assets/overview.png" style="width: 90%; height: auto;">
</p>

PrivacyLens is a data construction and multi-level evaluation framework for **evaluating privacy norm awareness of language models in action**.

## Setup
1. Clone the git repository.
    ```shell
    git clone https://github.com/SALT-NLP/PrivacyLens.git
    cd PrivacyLens
    ```
2. Install the required packages.
   ```shell
   python3 -m venv privacylens
   source privacylens/bin/activate  # On Windows use: privacylens\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up API keys. Create a `.env` file in the root directory and add the following:
   ```
   ANTHROPIC_API_KEY={anthropic_api_key}
   GEMINI_API_KEY={gemini_api_key}
   OPENAI_API_KEY={openai_api_key}
   ```
   Note: You only need to set the API keys for the models you plan to use.

## Quick Start
Before delving into the details of replicating our study or running experiment scripts, we provide a [quick-start notebook](helper/quick_start.ipynb) to walk you through what PrivacyLens can do.

### Quick Start Commands

Here are some example commands to quickly get started with evaluating Claude Sonnet 4.5. Run these commands from the `evaluation/` directory:

```bash
# 1a. Get final action with naive prompt (original paper - no privacy guidance)
python get_final_action.py \
  --input-path '../data/main_data.json' \
  --output-path '../results/actions_naive.csv' \
  --model 'claude-sonnet-4-5' \
  --prompt-type 'naive' \
  --start-index 0 \
  --num 493

# 1b. Get final action with privacy-enhanced prompt (original paper - balanced)
python get_final_action.py \
  --input-path '../data/main_data.json' \
  --output-path '../results/actions_privacy_enhanced.csv' \
  --model 'claude-sonnet-4-5' \
  --prompt-type 'privacy_enhanced' \
  --start-index 0 \
  --num 493

# 1c. Get final action with Zach's enhanced privacy prompt (maximum protection)
python get_final_action.py \
  --input-path '../data/main_data.json' \
  --output-path '../results/actions_privacy_by_zach.csv' \
  --model 'claude-sonnet-4-5' \
  --prompt-type 'privacy_prompt_by_zach' \
  --start-index 0 \
  --num 493

# 2. Evaluate leakage
python evaluate_final_action.py \
  --data-path '../data/main_data.json' \
  --action-path '../results/actions_privacy_by_zach.csv' \
  --step 'judge_leakage' \
  --output-path '../results/leakage_privacy_by_zach.json' \
  --model 'claude-sonnet-4-5' \
  --start-index 0 \
  --num 493

**Note**: Make sure your `.env` file is configured with `ANTHROPIC_API_KEY` before running these commands.
