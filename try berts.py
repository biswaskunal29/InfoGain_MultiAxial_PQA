from sentence_transformers import SentenceTransformer
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

text = 'joaqamole'
non_text1 = 'king'
non_text2 = 'queen'
list_text = [non_text1,non_text2]
#str_text = ' '.join(list_text)

#non_text1 = 'later'
#non_text2 = 'after'
text1 = 'and'
text2 = ['and', 'second', 'later', 'a', 'powerful', 'roar', 'away', 'into', 'the', 'night', 'and', 'now', 'to', 'fly', 'to', 'my', 'secret', 'where', 'not', 'even', 'batman', 'can', 'find', 'us', 'ign', 'catwoman', 'steal', 'the', 'diamond', 'shipment', 'call', 'the', 'police', 'a', 'swift', 'change', 'playboy', 'wayne', 'and', 'his', 'dick', 'greyson', 'become']
text3 = ['one', 'hour', 'later', 'on', 'the', 'gamb', 'a', 'good', 'thing', 'too', 'le', 'ship', 'two', 'sentry', 'idle', 'for', 'the', 'sea', 'gull', 'is', 'away', 'the', 'time', 'in', 'reality', 'a', 'unique', 'camouflage', 'under', 'pete', 'i', 'think', 'the', 'water', 'helmet', 'wear', 'try', 'some', 'target', 'shot', 'would', 'by', 'batman', 'practise', 'on', 'that', 'panic', 'the', 'chumps', 'sea', 'gull', 'at', 'table', 'put', 'your', 'gun', 'away', 'then', 'the', 'churning', 'sternwheel', 'carry', 'the', 'acrobatman', 'unseen', 'to', 'a', 'top', 'deck', 'are', 'tricky', 'but', 'a', 'sho', 'cut', 'to', 'the', 'wheel', 'room', 'one', 'hour', 'later', 'on', 'the', 'gamb', 'a', 'good', 'thing', 'too', 'le', 'ship', 'two', 'sentry', 'idle', 'for', 'the', 'sea', 'gull', 'is', 'away', 'the', 'time', 'in', 'reality', 'a', 'unique', 'camouflage', 'under', 'pete', 'i', 'think', 'the', 'water', 'helmet', 'wear', 'try', 'some', 'target', 'shot', 'would', 'by', 'batman', 'practise', 'on', 'that', 'panic', 'the', 'chumps', 'sea', 'gull', 'at', 'table', 'put', 'your', 'gun', 'away', 'then', 'the', 'churning', 'sternwheel', 'carry', 'the', 'acrobatman', 'unseen', 'to', 'a', 'top', 'deck', 'are', 'tricky', 'but', 'a', 'sho', 'cut', 'to', 'the', 'wheel', 'room']


# =============================================================================
# text_emb = model.encode(non_text1)
# text_emb2 = model.encode(non_text2)
# list_text_emb = model.encode(list_text)
# #str_text_emb = model.encode(str_text)
# 
# print(text_emb.shape)
# print(text_emb2.shape)
# 
# #sim1 = cosine_similarity([text_emb], [text_emb2])
# sim1 = 1 - spatial.distance.cosine(text_emb, text_emb2)
# 
# print(sim1)
# 
# list_sim = 1 - spatial.distance.cosine(list_text_emb[0], list_text_emb[1])
# 
# print(list_sim)
# =============================================================================

#print(str_text)

# =============================================================================
# Testing time to encode in 2 ways
# =============================================================================

print(len(text1), len(text2), len(text3))

for i in range(10):
    text1_emb = model.encode(text1)

print('done')

text3_emb = model.encode(text3)

print('done')

