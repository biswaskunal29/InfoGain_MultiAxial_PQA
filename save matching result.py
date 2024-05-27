#from Final_full_preprocessor import full_preprocess_text_to_token_list as prep
import io
from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np

#keys = ['Profile Picture', 'image', 'Profile ']

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['Image Shared Liked', 'Text in the Image', 'Profile Picture', 'Profile Banner', 'Profile Description']

question = 'What is the personality based on the Image text in the image?'

q1 = "What is the personality based on the Image shared?"
q2 = "What is the personality based on the Text in the image?"
q3 = "What is the personality based on the Profile Picture?"
q4 = "What is the personality based on the Profile Banner?"
q5 = "What is the personality based on the Profile Description?"
q6 = "What is the personality based on the entire Profile?"

ques_list = [q1,q2,q3,q4,q5,q6]

q11 = "Image shared liked"
q22 = "Text in the image"
q33 = "Profile Picture"
q44 = "Profile Banner"
q55 = "Profile Description"
q66 = "Profile"

#Sentences are encoded by calling model.encode()
sent_emb = model.encode(sentences)
ques_emb = model.encode(q6)
ques_list_emb = model.encode(ques_list)

#Check similarities
for i in range(len(sentences)):
    #    print(i)
    print(round((1 - spatial.distance.cosine(sent_emb[i], ques_emb)),5)   , sentences[i])

print()

#save similarities
ques_similarities = np.zeros(5, dtype = np.float64)
#print(ques_similarities)

for i in range(len(ques_list)):
#    print(type(question))
    ques_similarities_of_i = np.zeros(0, dtype = np.float64)
    for j in range(len(sentences)):
    #    print(i)
#        print(round((1 - spatial.distance.cosine(sent_emb[j], ques_list_emb[i])),5)   , sentences[j])
        ques_similarities_of_i = np.hstack((ques_similarities_of_i,(1 - spatial.distance.cosine(sent_emb[j], ques_list_emb[i]))  ))
    
    
    ques_similarities = np.vstack((ques_similarities, ques_similarities_of_i))
    
#    break

ques_similarities = np.delete(ques_similarities, 0, axis=0)

#save the similarities
#np.save("question similarities.npy", ques_similarities)


similar = np.load("question similarities.npy")













# =============================================================================
# new = prep(keys[0])
# 
# print(new)
# =============================================================================

# =============================================================================
# Glove trial
# with io.open("glove.6B.50d.txt", mode="r", encoding="utf-8") as f:
#     for line in f:
#         print(line.split())
#         break
# =============================================================================


#Our sentences we like to encode
# =============================================================================
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']
# =============================================================================


# =============================================================================
# #Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")
# =============================================================================


# =============================================================================
# print(cos(embeddings[0].reshape(-1, 1), embeddings[1].reshape(-1, 1)))
# print(cos(embeddings[0].reshape(-1, 1), embeddings[2].reshape(-1, 1)))
# =============================================================================

# =============================================================================
# print(spatial.distance.cosine(sent_emb[0], ques_emb))
# print(spatial.distance.cosine(sent_emb[3], ques_emb))
# =============================================================================
