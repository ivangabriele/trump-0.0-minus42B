# Trump-0.0-minus42B — The Trump LLM

**A really dumb and opinionated LLM — exclusively trained on Donald J. Trump's social media posts[^note].**

Arguably the dumbest Large Language Model ever created.

> [!WARNING]  
> This a work in progress,

[^note]: _Twitter, X and Truth Social posts from 2009 to 2025._

---

## Table of Contents

- [Why?](#why)
- [Local Run \& Development](#local-run--development)
  - [Getting started](#getting-started)
  - [1. Download Trump's social media posts](#1-download-trumps-social-media-posts)
  - [2. Teach the Generator LLM Preference Dataset](#2-teach-the-generator-llm-preference-dataset)
  - [3. Prepare the Generator LLM](#3-prepare-the-generator-llm)
  - [3. Generate the training data](#3-generate-the-training-data)
  - [4. Train Trump model](#4-train-trump-model)
    - [From a pre-trained model](#from-a-pre-trained-model)
    - [From scratch](#from-scratch)
  - [5. Run model (CLI chat)](#5-run-model-cli-chat)

---

## Why?

I needed a self-educational project to learn how to train an LLM from scratch and how to fine-tune pre-trained ones.

**Topics:**

- Datasets
- Fine-tuning
  - Transformers Trainer
  - RLHF (Reinforcement Learning from Human Feedback)
    1. Human Feedback
    2. RM (Reward Model)
    3. PPO (Proximal Policy Optimization)
- Training

## Local Run & Development

> [!WARNING]  
> There are more than 80,000 posts. This means that each step of the process will take a while to complete if you want to
> reproduce or customize them instead of using the existing repository data.

### Getting started

> [!IMPORTANT]  
> **Prerequisites:**
> - **Git LFS** (to download the `data/` directory)
> - **huggingface-cli** (installed globally)
> - **pyenv**
> - **uv**

```sh
git clone https://github.com/ivangabriele/Trump-0.0-minus42B.git
cd Trump-0.0-minus42B
uv sync
```

### 1. Download Trump's social media posts

This script use Roll Call's (FactSquared) API to download [Trump's social media
posts](https://rollcall.com/factbase-twitter/?platform=all&sort=date&sort_order=asc&page=1) — including deleted ones —
and store them as JSON files — by lots of 50 posts — in the `data/posts/` directory. 

Then it populates a local SQLite database under `data/posts.db` with the posts raw text and basic metadata, merging
Twitter/X posts that are split into multiple tweets (e.g. threads), and filtering out the posts that are useless for
training (e.g. images, videos, etc.).

```sh
make download
```

### 2. Teach the Generator LLM Preference Dataset

This script generates a human preference (feedback) dataset (`data/preference.json`) using your RLHF to select or
provide the best normalized output for each post. The model used for this step is prepped with a preliminary fwe-shot
prompt living in `generator_prompt.json`.

```sh
make teach <SAMPLE_SIZE>
```

### 3. Prepare the Generator LLM

_Not ready yet!_

This script build the Generator LLM by fine-tuning a pre-trained model using RM and PPO.

```sh
make prepare
```

where `<SAMPLE_SIZE>` is the posts random sample size to use for the human feedback dataset. It's a mandatory positional
argument.

### 3. Generate the training data

_Not ready yet!_

This script normalizes the posts using the custom Generator LLM and update them in the local SQLite database
`data/posts.db`.


```sh
make generate
```

### 4. Train Trump model

You have 2 choices here:
- Either fine-tune a pre-trained model, by default `facebook/opt-125m`.
- Or train the model from scratch, which will give you the worst results (but the most fun!).

#### From a pre-trained model

Using the default `facebook/opt-125m` model:

```sh
make tune
```

You can also specify a different model:

_Not ready yet!_

```sh
make tune MODEL=facebook/opt-350m
```

#### From scratch

_Not ready yet!_

```sh
make train
```

### 5. Run model (CLI chat)

Starts a CLI chat with the model.

```sh
make run
```
