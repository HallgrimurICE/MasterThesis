import dill

with open("observations.npz", "rb") as f:
    data = dill.load(f)

print(type(data))
print(data)
