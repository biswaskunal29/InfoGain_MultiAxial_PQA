import pickle



word_info_file = "word_info.pkl"

#read the dict
with open(word_info_file, 'rb') as handle:
  word_info = pickle.loads(handle.read())

print(word_info['joaqamole']) # True























