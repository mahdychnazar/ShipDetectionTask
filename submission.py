import pandas as pd
import numpy as np
import model as md
import utils as ut
import tensorflow as tf


submission = pd.read_csv(".//data//airbus-ship-detection//sample_submission_v2.csv")
test_image_path = './/data//airbus-ship-detection//test_v2//'
weights_path = './/model//best_small_unet.h5'

def get_prediction(row: pd.Series) -> pd.Series:
    image = ut.load_image(test_image_path + row['ImageId'])
    pred_mask = md.small_unet.predict(image)
    pred_mask = tf.reshape(pred_mask, [768, 768])
    pred_mask = np.array(pred_mask)
    pred_mask = pred_mask.astype(np.uint8)
    row['EncodedPixels'] = ut.rle_encode(pred_mask)
    if row['EncodedPixels'] == '':
        row['EncodedPixels'] = np.nan
    return row


md.small_unet.load_weights(weights_path)
submission = submission.apply(lambda x: get_prediction(x), axis=1).set_index("ImageId")

submission.to_csv("./submission.csv")

