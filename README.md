# Trump-0.0-minus42B — The Trump LLM

**A really dumb and opinionated LLM — exclusively trained on Donald J. Trump's social media posts[^note].**

Arguably the dumbest Large Language Model ever created.

> [!WARNING]  
> This a work in progress,

[^note]: _Twitter, X and Truth Social posts from 2009 to 2025._

---

## Table of Contents

- [Why?](#why)
- [Usage](#usage)
  - [Getting started](#getting-started)
  - [1. Download Trump's social media posts](#1-download-trumps-social-media-posts)
  - [2. Prepare the Normalizer LLM](#2-prepare-the-normalizer-llm)
  - [3. Normalize and generate the training data](#3-normalize-and-generate-the-training-data)
  - [3. Train Trump model](#3-train-trump-model)
    - [Fine-tune a pre-trained model](#fine-tune-a-pre-trained-model)
    - [Train from scratch](#train-from-scratch)
  - [Run model (CLI chat)](#run-model-cli-chat)

---

## Why?

I needed a self-educational project to learn how to train an LLM from scratch and how to fine-tune pre-trained ones.

## Usage

### Getting started

> [!IMPORTANT]  
> **Prerequisites:**
> - **Git LFS** (to download the `data/` directory)
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

> [!WARNING]  
> There are more than 80,000 posts.

```sh
make data
```

### 2. Prepare the Normalizer LLM

_Not ready yet!_

### 3. Normalize and generate the training data

_Not ready yet!_

This script normalizes the posts content using the RLHF fine-tuned local "normalizer" LLM with few-shot prompting.

It then stores the normalized posts in the local SQLite database `posts.db`.

> [!WARNING]  
> There are more than 80,000 posts. 

```sh
make normalize
```

### 3. Train Trump model

You have 2 choices here:
- Either fine-tune a pre-trained model, by default `facebook/opt-125m`.
- Or train the model from scratch, which will give you the worst results (but the most fun!).

#### Fine-tune a pre-trained model

Using the default `facebook/opt-125m` model:

```sh
make tune
```

You can also specify a different model:

_Not ready yet!_

```sh
make tune MODEL=facebook/opt-350m
```

#### Train from scratch

_Not ready yet!_

```sh
make train
```

### Run model (CLI chat)

Starts a CLI chat with the model.

```sh
make run
```
