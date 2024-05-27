import pickle

a = {
  'a': 1,
  'b': 2
}

with open('file.txt', 'wb') as handle:
  pickle.dump(a, handle)

with open('file.txt', 'rb') as handle:
  b = pickle.loads(handle.read())

print(a == b) # True






















