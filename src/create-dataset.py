from types import SimpleNamespace
import os
import re

from datasets import Dataset, Audio, load_dataset
from dotenv import load_dotenv
import pandas as pd

load_dotenv(".env")

path_cfg = SimpleNamespace(
    to_folder_data=os.getenv("PATH_TO_FOLDER_DATA"),
)

huggingface_cfg = SimpleNamespace(
    repo_id=os.getenv("HUGGINFACE_REPO_ID"),
    auth_token=os.getenv("HUGGINFACE_AUTH_TOKEN"),
)

CFG = SimpleNamespace(
    path=path_cfg,
    huggingface=huggingface_cfg,
)


if __name__ == "__main__":
    # Loading Data
    users_df = pd.read_csv(f"{CFG.path.to_folder_data}/users.csv")

    # Preprocessing Data
    users_df["text"] = users_df["text"].apply(lambda text: re.sub(r"[^0-9а-яА-Яіїєґ]", " ", text))
    users_df["audio"] = users_df.apply(lambda row: f"{row['user_id']}/{row['index']}.wav", axis=1)

    # Converting to HuggingFace Dataset format
    common_voice_train = Dataset.from_pandas(users_df)
    print(common_voice_train.shape, list(users_df))

    # Convert wav files to arrays
    common_voice_train = common_voice_train.cast_column("audio", Audio())
    print(common_voice_train[0])

    # Pushing to HuggingFace HUB
    common_voice_train.push_to_hub(
        path=CFG.huggingface.repo_id, use_auth_token=CFG.huggingface.auth_token
    )

    # Loading from HuggingFace HUB
    common_voice_train = load_dataset(
        path=CFG.huggingface.repo_id,
        use_auth_token=CFG.huggingface.auth_token,
        split="train",
        streaming=False,
    )
    print(len(common_voice_train))
