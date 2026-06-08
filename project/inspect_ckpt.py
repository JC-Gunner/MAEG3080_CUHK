import torch
ck = torch.load(r"C:\Users\86186\Desktop\machineLearning\best.pt", map_location="cpu")
print("model_name:", ck.get("model_name"))
print("base_width:", ck.get("base_width"))
print("blocks_per_stage:", ck.get("blocks_per_stage"))
print("use_se:", ck.get("use_se"))
