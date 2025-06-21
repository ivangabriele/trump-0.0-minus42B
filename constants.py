DOWNLOAD_API_URL = "https://api.factsquared.com/json/factba.se-trump-social.php"

GENERATOR_MODEL = "facebook/opt-125m"
# GENERATOR_MODEL = "google/gemma-3-1b-it"
# GENERATOR_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# GENERATOR_MODEL = "QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF"
GENERATOR_MODEL_DIR_PATH = "./models/generator"
GENERATOR_PROMPT_CONFIG_PATH = "generator_prompt.json"
# REWARD_MODEL = "distilbert/distilbert-base-uncased"
REWARD_MODEL = "facebook/opt-125m"
REWARD_MODEL_DIR_PATH = "./models/rm"

POSTS_DATA_DIR_PATH = "data/posts"
# Human (Feedback) Preference Dataset
PREFERENCE_DATASET_PATH = "data/preference.json"
SQLITE_DB_FILE_PATH = "data/posts.db"
