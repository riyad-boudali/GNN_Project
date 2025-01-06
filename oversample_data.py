#%% Lodaing raw dataset 
import pandas as pd

data = pd.read_csv("data/raw/HIV_train.csv")
data.index = data["index"]
data["HIV_active"].value_counts()
start_index = data.iloc[0]["index"]
data

#%% Check how many additional samples we need
neg_class = data["HIV_active"].value_counts()[0]
pos_calss = data["HIV_active"].value_counts()[1]
multiplier = int(neg_class/pos_calss) + 2
print(multiplier)

# Ruplicating the positive samples in the dataset
replicated_pos = [data[data["HIV_active"] == 1]]*multiplier
replicated_pos_df = pd.concat(replicated_pos, ignore_index=True)
replicated_pos_df

#%% Concatinate the two dataframes
data_merged = pd.concat([data, replicated_pos_df], ignore_index=True)
print(data_merged.shape)

# Shuffle the dataset
data_merged = data_merged.sample(frac=1).reset_index(drop=True)

# Re-assign index (This is our ID later)
index = range(start_index, start_index + data_merged.shape[0])
data_merged.index = index
data_merged["index"] = data_merged.index
data_merged.head()

# %% Save the sample dataset into a csv file
data_merged.to_csv("data/raw/HIV_train_oversampled.csv")