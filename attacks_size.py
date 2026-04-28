# import os
# from utils import read_json_file


# malware_dir = "malware"
# adv_dir_root = "adversarial_evaluation/adversarial_examples/"

# macro_config = read_json_file("configurations/ADVERSARIAL/adversarial.json")
# models = macro_config["models"][:6]
# attacks = macro_config["attacks"]

# result_file = "attacks_size.csv"

# for model in models:
#     for attack, param in attacks.items():
#         if "full_dos" in attack:
#             attack = "full_dos"
#         if "content_shift" in attack:
#             attack = "content_shift"
#         if param == "OptimizerBackends.NEVERGRAD":
#             param = "nevergrad"
#         if param == "OptimizerBackends.GRADIENT":
#             param = "gradient"
#         if attack == "gamma_5":
#             attack = "gamma"
#             param = "5"
#         if attack == "gamma_10":
#             attack = "gamma"
#             param = "10"
#         if attack == "gamma_20":
#             attack = "gamma"
#             param = "20"
#         if attack == "gamma_30":
#             attack = "gamma"
#             param = "30"
#         if attack == "gamma_50":
#             attack = "gamma"
#             param = "50"
#         adv_dir = os.path.join(adv_dir_root, model, attack, str(param))
#         print(f"\nModel: {model}, Attack: {attack}, Param: {param}")
#         diffs = []
#         for filename in os.listdir(malware_dir):
#             malware_path = os.path.join(malware_dir, filename)
#             adv_filename = f"{os.path.splitext(filename)[0]}_adv{os.path.splitext(filename)[1]}"
#             adv_path = os.path.join(adv_dir, adv_filename)

#             if os.path.isfile(malware_path) and os.path.isfile(adv_path):
#                 malware_size = os.path.getsize(malware_path)
#                 adv_size = os.path.getsize(adv_path)
#                 diff = adv_size - malware_size
#                 diffs.append(diff)
#                 # print(f"{filename}: {malware_size} bytes -> {adv_filename}: {adv_size} bytes | Difference: {diff} bytes")
#             else:
#                 print()
        
#         if diffs:
#             if attack == "content_shift":
#                 attack = "CS"
#                 param = ""
#             if attack == "full_dos":
#                 attack = "FD"
#                 param = ""
#             mean_diff = sum(diffs) / len(diffs)
#             mean_diff = f"{mean_diff:.2f}"
#             with open(result_file, "a") as f:
#                 f.write(f"{model}_{attack}_{param},{mean_diff}\n")


#%%
import os
from utils import read_json_file
import pandas as pd
import lief

malware_dir = "malware"
adv_dir_root = "adversarial_evaluation/adversarial_examples_1/"
adv_scores_root = "adversarial_evaluation/adversarial_scores_1/"
adv_transfer_root = "adversarial_evaluation/transfer_scores_1/"

macro_config = read_json_file("configurations/ADVERSARIAL/adversarial.json")
models_s = macro_config["models"][:6]
models_t = macro_config["models"][6:14]
attacks = dict(reversed(list(macro_config["attacks_inference"].items())))


thresholds = pd.read_csv("thresholds/vanilla_thresholds.csv")

all_results = pd.DataFrame({'hash': os.listdir(malware_dir)})

# model vs same model

# all_results = pd.DataFrame({'hash': sorted(os.listdir(malware_dir))})

# for model in models_s:

#     result_file = f"results/adv/data_sp/{model}_vs_{model}.csv"

#     for attack, param in attacks.items():
#         column = []
#         if "full_dos" in attack:
#             attack = "full_dos"
#         if "content_shift" in attack:
#             attack = "content_shift"
#         if param == "OptimizerBackends.NEVERGRAD":
#             param = "nevergrad"
#         if param == "OptimizerBackends.GRADIENT":
#             param = "gradient"
#         if attack == "gamma_5":
#             attack = "gamma"
#             param = "5"
#         if attack == "gamma_10":
#             attack = "gamma"
#             param = "10"
#         if attack == "gamma_20":
#             attack = "gamma"
#             param = "20"
#         if attack == "gamma_30":
#             attack = "gamma"
#             param = "30"
#         if attack == "gamma_50":
#             attack = "gamma"
#             param = "50"

#         adv_dir = os.path.join(adv_dir_root, model, attack, str(param))
#         adv_scores = os.path.join(adv_scores_root, model, attack + "_" + str(param) + ".csv")
#         scores = pd.read_csv(adv_scores, header=None)

#         print(f"\nModel: {model}, Attack: {attack}, Param: {param}")

#         for filename in sorted(os.listdir(malware_dir)):
#             malware_path = os.path.join(malware_dir, filename)

#             adv_filename = f"{os.path.splitext(filename)[0]}_adv"
#             adv_path = os.path.join(adv_dir, adv_filename)
            
#             match = scores[scores.iloc[:, 0] == adv_filename]
#             score_value = match.iloc[0, 1]

#             if score_value < thresholds[thresholds.iloc[:, 0] == model].iloc[0, 1]:
#                 if attack == "gamma": 
#                     adv_manipulation = os.path.getsize(adv_path) - os.path.getsize(malware_path)
#                 if attack == "full_dos":
#                     binary = lief.parse(malware_path)
#                     adv_manipulation = binary.dos_header.addressof_new_exeheader
#                 if attack == "content_shift":
#                     adv_manipulation = 4096
#                 column.append(adv_manipulation)
#             else:
#                 column.append(-1)
#         column_name = f"{attack}_{param}"
#         if attack == "content_shift":
#             column_name = "CS"
#         if attack == "full_dos":
#             column_name = "FD"
#         all_results[column_name] = pd.Series(column)

#     all_results.to_csv(result_file, index=False) 


# vanilla vs certificates

for model_s in models_s:    
    for model_t in models_t:
        if model_s == model_t:
            continue

        result_file = f"results/adv/data_sp/{model_s}_vs_{model_t}.csv"

        print(f"{model_s}_vs_{model_t}")

        for attack, param in attacks.items():
            column = []
            if "full_dos" in attack:
                attack = "full_dos"
            if "content_shift" in attack:
                attack = "content_shift"
            if param == "OptimizerBackends.NEVERGRAD":
                param = "nevergrad"
            if param == "OptimizerBackends.GRADIENT":
                param = "gradient"
            if attack == "gamma_5":
                attack = "gamma"
                param = "5"
            if attack == "gamma_10":
                attack = "gamma"
                param = "10"
            if attack == "gamma_20":
                attack = "gamma"
                param = "20"
            if attack == "gamma_30":
                attack = "gamma"
                param = "30"
            if attack == "gamma_50":
                attack = "gamma"
                param = "50"

            adv_dir = os.path.join(adv_dir_root, model_s, attack, str(param))
            adv_scores = os.path.join(adv_transfer_root, model_t, attack, str(param), model_s + ".csv")
            scores_t = pd.read_csv(adv_scores, header=None)

            for i, adv_filename in enumerate(os.listdir(adv_dir)):

                malware_filename = adv_filename.replace("_adv", "")
                malware_path = os.path.join(malware_dir, malware_filename)

                # adv_filename = f"{os.path.splitext(filename)[0]}_adv"
                adv_path = os.path.join(adv_dir, adv_filename)
                
                score = scores_t.iloc[i,1]

                if score < 0.5:
                    if attack == "gamma": 
                        adv_manipulation = os.path.getsize(adv_path) - os.path.getsize(malware_path)
                    elif attack == "full_dos":
                        binary = lief.parse(malware_path)
                        adv_manipulation = binary.dos_header.addressof_new_exeheader
                    elif attack == "content_shift":
                        adv_manipulation = 4096
                else:
                    adv_manipulation = -1

                # Find the row where 'hash' matches malware_filename and set the value
                column_name = f"{attack}_{param}"
                if attack == "content_shift":
                    column_name = "CS"
                if attack == "full_dos":
                    column_name = "FD"
                if column_name not in all_results.columns:
                    all_results[column_name] = adv_manipulation
                all_results.loc[all_results['hash'] == malware_filename, column_name] = adv_manipulation

        all_results.to_csv(result_file, index=False) 

        
# vanilla vs vanilla 

# for model_s in models_s:    
#     for model_t in models_t:
#         if model_s == model_t:
#             continue

#         result_file = f"results/adv/data_sp/{model_s}_vs_{model_t}.csv"

#         print(f"{model_s}_vs_{model_t}")

#         for attack, param in attacks.items():
#             column = []
#             if "full_dos" in attack:
#                 attack = "full_dos"
#             if "content_shift" in attack:
#                 attack = "content_shift"
#             if param == "OptimizerBackends.NEVERGRAD":
#                 param = "nevergrad"
#             if param == "OptimizerBackends.GRADIENT":
#                 param = "gradient"
#             if attack == "gamma_5":
#                 attack = "gamma"
#                 param = "5"
#             if attack == "gamma_10":
#                 attack = "gamma"
#                 param = "10"
#             if attack == "gamma_20":
#                 attack = "gamma"
#                 param = "20"
#             if attack == "gamma_30":
#                 attack = "gamma"
#                 param = "30"
#             if attack == "gamma_50":
#                 attack = "gamma"
#                 param = "50"

#             adv_dir = os.path.join(adv_dir_root, model_s, attack, str(param))
#             adv_scores = os.path.join(adv_transfer_root, model_t, attack, str(param), model_s + ".csv")
#             scores_t = pd.read_csv(adv_scores, header=None)

#             for i, adv_filename in enumerate(os.listdir(adv_dir)):

#                 malware_filename = adv_filename.replace("_adv", "")
#                 malware_path = os.path.join(malware_dir, malware_filename)

#                 # adv_filename = f"{os.path.splitext(filename)[0]}_adv"
#                 adv_path = os.path.join(adv_dir, adv_filename)
                
#                 score = scores_t.iloc[i,1]

#                 if score < thresholds[thresholds.iloc[:, 0] == model_t].iloc[0, 1]:
#                     if attack == "gamma": 
#                         adv_manipulation = os.path.getsize(adv_path) - os.path.getsize(malware_path)
#                     elif attack == "full_dos":
#                         binary = lief.parse(malware_path)
#                         adv_manipulation = binary.dos_header.addressof_new_exeheader
#                     elif attack == "content_shift":
#                         adv_manipulation = 4096
#                 else:
#                     adv_manipulation = -1

#                 # Find the row where 'hash' matches malware_filename and set the value
#                 column_name = f"{attack}_{param}"
#                 if attack == "content_shift":
#                     column_name = "CS"
#                 if attack == "full_dos":
#                     column_name = "FD"
#                 if column_name not in all_results.columns:
#                     all_results[column_name] = adv_manipulation
#                 all_results.loc[all_results['hash'] == malware_filename, column_name] = adv_manipulation

#         all_results.to_csv(result_file, index=False) 


# %%
# import os 
# print(os.path.getsize("malware/0007f8bbd76b65572308471d8b48a32e8f0dcf86ca0dabda6039007d39df4642"))
# print(os.path.getsize("/mnt/data/aponte/repos/exebenchmark/adversarial_evaluation/adversarial_examples/MalConv/gamma/5/0007f8bbd76b65572308471d8b48a32e8f0dcf86ca0dabda6039007d39df4642_adv"))
# print(os.path.getsize("/mnt/data/aponte/repos/exebenchmark/adversarial_evaluation/adversarial_examples/MalConv/gamma/10/0007f8bbd76b65572308471d8b48a32e8f0dcf86ca0dabda6039007d39df4642_adv"))
# print(os.path.getsize("/mnt/data/aponte/repos/exebenchmark/adversarial_evaluation/adversarial_examples/MalConv/gamma/20/0007f8bbd76b65572308471d8b48a32e8f0dcf86ca0dabda6039007d39df4642_adv"))
# print(os.path.getsize("/mnt/data/aponte/repos/exebenchmark/adversarial_evaluation/adversarial_examples/MalConv/gamma/30/0007f8bbd76b65572308471d8b48a32e8f0dcf86ca0dabda6039007d39df4642_adv"))
# print(os.path.getsize("/mnt/data/aponte/repos/exebenchmark/adversarial_evaluation/adversarial_examples/MalConv/gamma/50/0007f8bbd76b65572308471d8b48a32e8f0dcf86ca0dabda6039007d39df4642_adv"))

# %%
# import lief
# import os 

# malware_dir = "malware"

# for filename in sorted(os.listdir(malware_dir)):
#     malware_path = os.path.join(malware_dir, filename)
#     if os.path.isfile(malware_path):
#         binary = lief.parse(malware_path)
#         dos_header_size = binary.dos_header.addressof_new_exeheader  # = e_lfanew
#         with open("DOS_sizes.txt", "a") as f:
#             f.write(f"{filename},{dos_header_size}\n")
#         print(f"DOS header size: {dos_header_size} bytes")
# %%
