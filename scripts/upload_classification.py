import pandas as pd

import wandb

groups = {
    "embeddings/one_direction/sentences_simcse": "Sentences SimCSE",
    "embeddings/one_direction/sentences_word2vec": "Sentences Word2Vec",
    "embeddings/one_direction/words_simcse": "Words SimCSE",
    "embeddings/one_direction/words_word2vec": "Words Word2Vec",
    "embeddings/one_direction/perfect.dill": "Perfect",
}


def main(file: str):
    # File is a path to a csv file, open it with pandas
    # and upload it to wandb
    df = pd.read_csv(file)

    started = False
    last_step = 1
    # Iterate over the rows
    for _, row in df.iterrows():
        if row["z/env_steps"] < last_step:
            if started:
                wandb.finish()
            wandb.init(
                project="Classify Vectors",
                group=groups[row["file"]],
            )
            started = True
        # Log the metrics
        wandb.log(row.to_dict())
        last_step = row["z/env_steps"]


if __name__ == "__main__":
    main(
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\classifier\\progress-filled.csv"
    )
