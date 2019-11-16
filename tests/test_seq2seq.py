from phramer.deploy.models.seq2seq import Seq2SeqModel
import time

model = Seq2SeqModel()
print("Prediction started:")
start_time = time.clock()
model.predict()
end_time = time.clock()
print("Prediction took {} seconds".format(end_time - start_time))
