#import math
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

categories = ['talk.religion.misc',
              'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)

X, Y = newsgroups_train.data, newsgroups_train.target
cv = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english')
X_vec = cv.fit_transform(X)

res = dict(zip(cv.get_feature_names_out(),
               mutual_info_classif(X_vec, Y, discrete_features=True)
               ))
print(res)

# =============================================================================
# def ig(class_, feature):
#   classes = set(class_)
# 
#   Hc = 0
#   for c in classes:
#     pc = list(class_).count(c)/len(class_)
#     Hc += - pc * math.log(pc, 2)
#   print('Overall Entropy:', Hc)
#   feature_values = set(feature)
# 
#   Hc_feature = 0
#   for feat in feature_values:
# 
#     pf = list(feature).count(feat)/len(feature)
#     indices = [i for i in range(len(feature)) if feature[i] == feat]
#     clasess_of_feat = [class_[i] for i in indices]
#     for c in classes:
#         pcf = clasess_of_feat.count(c)/len(clasess_of_feat)
#         if pcf != 0:
#             temp_H = - pf * pcf * math.log(pcf, 2)
#             Hc_feature += temp_H
#   ig = Hc - Hc_feature
#   return ig 
# 
# a = ig(Y,X)
# =============================================================================




























