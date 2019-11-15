from phramer.data.dataset import CNNDailyMail
from phramer.data import split

cnndm = CNNDailyMail()
cnndm.preprocess(
        '/home/phramer/data/cnn_dailymail/cnn/stories',
        '/home/phramer/data/cnn_dailymail/dailymail/stories',
        '/home/whiteRa2bit/phramer/res',
)
