import os
import shutil

# # Define the base directory
# base_dir = "adversarial_evaluation/adversarial_examples"

# # Iterate through each folder in the base directory
# for folder in os.listdir(base_dir):
#     folder_path = os.path.join(base_dir, folder)
#     if os.path.isdir(folder_path):  # Check if it's a folder
#         gamma_path = os.path.join(folder_path, "gamma")
#         if os.path.exists(gamma_path) and os.path.isdir(gamma_path):
#             target_dir = os.path.join(gamma_path, "50")
            
#             # Move files that are not folders into the "50" folder
#             files_moved = 0
#             for item in os.listdir(gamma_path):
#                 item_path = os.path.join(gamma_path, item)
#                 if os.path.isfile(item_path):  # Check if it's a file
#                     shutil.move(item_path, os.path.join(target_dir, item))
#                     files_moved += 1
            
#             # Print the number of files moved
#             print(f"Moved {files_moved} files from {gamma_path} to {target_dir}.")

#%%

import pandas as pd
import os

data = "results/adv/data"
data_sp = "results/adv/data_sp"

for filename in os.listdir(data):
    if filename.endswith(".csv"):
        data_file = os.path.join(data, filename)
        data_sp_file = os.path.join(data_sp, filename)
        
        df_data = pd.read_csv(data_file)

        if os.path.exists(data_sp_file):

            df_sp = pd.read_csv(data_sp_file)
            # Ensure the first column is used for matching
            key_col = df_data.columns[0]
            df_merged = pd.merge(df_data, df_sp.iloc[:, 0:3], left_on=key_col, right_on=df_sp.columns[0], how='left')
            # Fill missing values with -1
            # df_merged.iloc[:, -2:] = df_merged.iloc[:, -2:].fillna(-1)
            # Drop the duplicate key column from df_sp
            # Drop the duplicate key column from df_sp, but keep the original hash column from df_data
            # df_merged = df_merged.drop(columns=[df_sp.columns[0] + "_y"])
        else:
            # Add two columns of -1
            df_merged = df_data.copy()
            df_merged["sp_col1"] = -1
            df_merged["sp_col2"] = -1
            
        df_merged.columns.values[-2:] = ["gamma_10_sp", "gamma_5_sp"]
        # Save merged file to a new folder
        output_dir = "results/adv/data_merged"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)
        df_merged.to_csv(output_file, index=False)

#%% 
import os 

folder1 = "/mnt/data/aponte/repos/exebenchmark/adversarial_evaluation/adversarial_examples_1/BBDnn/gamma/5"
folder2 = "malware"
filename = "032703331e26b28661519bba0b561b9919d787560bfe6c9e43109315bb8f2c30"

file1_path = os.path.join(folder1, filename + "_adv")
file2_path = os.path.join(folder2, filename)

size1 = os.path.getsize(file1_path) if os.path.exists(file1_path) else None
size2 = os.path.getsize(file2_path) if os.path.exists(file2_path) else None

print(f"Size in {folder1}: {size1} bytes")
print(f"Size in {folder2}: {size2} bytes")
if size1 is not None and size2 is not None:
    print(f"Difference in size: {size1 - size2} bytes")
else:
    print("One or both files do not exist.")
# %%
