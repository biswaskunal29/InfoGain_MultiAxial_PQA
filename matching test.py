from Final_full_preprocessor import full_preprocess_text_to_token_list as prep
import io
from sentence_transformers import SentenceTransformer
from scipy import spatial


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

q11 = "Image shared liked"
q22 = "Text in the image"
q33 = "Profile Picture"
q44 = "Profile Banner"
q55 = "Profile Description"
q66 = "Profile"

#Sentences are encoded by calling model.encode()
sent_emb = model.encode(sentences)
ques_emb = model.encode(q2)


#Check similarities
for i in range(len(sentences)):
    #    print(i)
    print(round((1 - spatial.distance.cosine(sent_emb[i], ques_emb)),5)   , sentences[i])
























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
