import kagglehub
import os
import pandas as pd
from   sklearn.model_selection import train_test_split

def load_csv(kaggle_path: str, csv: str):
    dataset_path = kagglehub.dataset_download(kaggle_path)
    csv_file = dataset_path + "/" +csv
    df = pd.read_csv(csv_file)

    df = df.drop(columns=["User_ID"]);
    
    return df


#random_state is the seed used in order to do the random splitting According to Hitchhiker’s Guide, 
#recommended values are 7,17,42.
def split_csv(df: pd.DataFrame, target_column: str, output_dir: str, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 17):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df[target_column]
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = load_csv(kaggle_path="atharvasoundankar/global-music-streaming-trends-and-listener-insights", csv="Global_Music_Streaming_Listener_Preferences.csv")    
    df.drop("Listening Time (Morning/Afternoon/Night)", axis=1)

    train_df, val_df, test_df = split_csv(df, output_dir="../data/raw", target_column="Listening Time (Morning/Afternoon/Night)")