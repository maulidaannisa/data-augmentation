import pickle
from scipy import spatial

cadec_disease = pickle.load( open( "cadec_disease.pickle", "rb" ) )

mention = 'ill feeling'
mention_key = ''
sim = 0
for key, val in cadec_disease.items():
    if key.lower() == mention.lower():
        continue
        
    # Note that spatial.distance.cosine computes the distance, and not the similarity. So, you must subtract the value from 1 to get the similarity.
    res = 1 - spatial.distance.cosine(val, cadec_disease[mention])
    if res > sim:
        sim = res
        mention_key = key

print(sim, mention_key)

