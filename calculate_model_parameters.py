
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.resnet18 import ResNet18

malconv = MalConv(
        embedding_size=8,
        max_len=2000000,
        threshold=0.5,
        padding_idx=256,
        kernel_size=512,
        stride=512
    )
avastconv = AvastStyleConv(
        embedding_size=8,
        max_len=51200,
        threshold=0.5,
        padding_idx=256
    )
ngramconv = NGramConv(
        embedding_size=8,
        max_len=512000,
        threshold=0.5,
        padding_idx=256
    )
bbdnn = BBDnn(
        embedding_size=10,
        max_len=102400,
        threshold=0.5,
        padding_idx=256
    )
resnet = ResNet18()

models = [malconv, avastconv, ngramconv, bbdnn, resnet]

with open("model_parameters.txt", "w") as f:
    for model in models:
        # Total and trainable parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model: ", model)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        f.write("Model: {}\n".format(model))
        f.write("Total parameters: {}\n".format(total_params))
        f.write("Trainable parameters: {}\n\n\n".format(trainable_params))
