import os, json
import numpy as np, torch, torch.nn as nn
from torchvision import datasets
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from model import SmallCNN

ds = datasets.MNIST("data", train=False, download=True)
x = (ds.data.numpy().astype("float32") / 255.0)[:200, None]
y = ds.targets.numpy()[:200]
m = SmallCNN(); m.load_state_dict(torch.load("models/standard.pt", weights_only=True)); m.eval()
clf = PyTorchClassifier(m, nn.CrossEntropyLoss(), input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1))
adv = FastGradientMethod(clf, eps=0.2).generate(x)
pc, pa = clf.predict(x).argmax(1), clf.predict(adv).argmax(1)
i = int(np.where((pc == y) & (pa != y))[0][0])
os.makedirs("samples", exist_ok=True)
json.dump({"image": x[i, 0].tolist()}, open("samples/clean.json", "w"))
json.dump({"image": adv[i, 0].tolist()}, open("samples/adv.json", "w"))
print(f"true={y[i]} standard(clean)={pc[i]} standard(adv)={pa[i]}")