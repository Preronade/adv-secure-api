import os, json
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer
from model import SmallCNN

torch.manual_seed(0); np.random.seed(0)
os.makedirs("models", exist_ok=True)

def load(train):
    ds = datasets.MNIST("data", train=train, download=True)
    return (ds.data.numpy().astype("float32") / 255.0)[:, None], ds.targets.numpy()

x_train, y_train = load(True); x_test, y_test = load(False)
x_train, y_train = x_train[:20000], y_train[:20000]      # subset: keeps it fast on an i3
x_eval, y_eval = x_test[:1000], y_test[:1000]

def wrap(m):
    return PyTorchClassifier(model=m, loss=nn.CrossEntropyLoss(),
                             optimizer=optim.Adam(m.parameters(), lr=1e-3),
                             input_shape=(1, 28, 28), nb_classes=10, clip_values=(0.0, 1.0))

def acc(clf, x, y):
    return float((clf.predict(x).argmax(1) == y).mean())

def evaluate(clf):
    fgsm = FastGradientMethod(clf, eps=0.2)
    pgd = ProjectedGradientDescent(clf, eps=0.2, eps_step=0.05, max_iter=20, verbose=False)
    return {"clean": acc(clf, x_eval, y_eval),
            "fgsm": acc(clf, fgsm.generate(x_eval), y_eval),
            "pgd": acc(clf, pgd.generate(x_eval), y_eval)}

results = {}

m1 = SmallCNN(); c1 = wrap(m1)
c1.fit(x_train, y_train, batch_size=128, nb_epochs=3)
results["standard"] = evaluate(c1); print("standard:", results["standard"])
torch.save(m1.state_dict(), "models/standard.pt")

m2 = SmallCNN(); c2 = wrap(m2)
pgd_train = ProjectedGradientDescent(c2, eps=0.2, eps_step=0.05, max_iter=5, verbose=False)
AdversarialTrainer(c2, attacks=pgd_train, ratio=0.5).fit(x_train, y_train, batch_size=128, nb_epochs=5)
results["robust"] = evaluate(c2); print("robust:", results["robust"])
torch.save(m2.state_dict(), "models/robust.pt")

json.dump(results, open("models/metrics.json", "w"), indent=2)