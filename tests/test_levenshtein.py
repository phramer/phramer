from phramer.deploy.models.levenshtein import LevenshteinModel
import time

model = LevenshteinModel()
print("Prediction started:")
start_time = time.clock()
model.predict()
end_time = time.clock()
print("Prediction took {} seconds".format(end_time - start_time))
