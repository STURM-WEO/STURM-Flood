import tensorflow as tf
from tensorflow.keras import backend as K

# losses ref. https://arxiv.org/pdf/2312.05391.pdf
# categorical crossentropy
cce = tf.keras.losses.CategoricalCrossentropy()
# Tversky loss ref. https://arxiv.org/abs/1706.05721
# α=β=0.5 --> Sørensen–Dice coefficient - F1-score based loss
# α=β=1 --> Jaccard (Tanimoto) coefficient - IoU-score based loss
class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.3, beta=0.7, smooth = 1e-3, name='tversky'):
        super(TverskyLoss, self).__init__(name=name)
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
    def call(self, y_true, y_pred):
        axis_to_reduce = range(1, K.ndim(y_pred))  # All axis but first (batch)
        true_positives = tf.reduce_sum(y_true * y_pred, axis=axis_to_reduce)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=axis_to_reduce)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=axis_to_reduce)
        tversky = (true_positives + self.smooth) / (true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth) # epsilon
        return 1 - tversky
# # Sørensen–Dice coefficient - F1-score based loss
# def dice_loss(y_true, y_pred):
#     smooth = 0.5
#     axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
#     numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=axis_to_reduce)
#     denominator = tf.reduce_sum(y_true + y_pred, axis=axis_to_reduce) 
#     score = (numerator  + smooth) / (denominator + smooth)
#     return 1 - score
# weighted dice
class WeightedDiceLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, smooth=0.5, name="weighted_dice_loss"): #smaller smooth for slower convergence
        super().__init__(name=name)
        if not isinstance(class_weights, tf.Tensor):
            class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.class_weights = class_weights
        self.smooth = smooth

    def call(self, y_true, y_pred):
        axis_to_reduce = list(range(1, K.ndim(y_pred)))
        numerator = y_true * y_pred * self.class_weights
        numerator = 2. *  tf.reduce_sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * self.class_weights
        denominator =  tf.reduce_sum(denominator, axis=axis_to_reduce) 
        score = (numerator + self.smooth) / (denominator + self.smooth)
        return 1 - score
# # Jaccard intex - IoU based loss
# def jaccard_loss(y_true, y_pred):
#     smooth = 0.5
#     axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
#     intersection = tf.reduce_sum(y_true * y_pred, axis=axis_to_reduce) 
#     union = tf.reduce_sum(y_true + y_pred, axis=axis_to_reduce) - intersection 
#     score = (intersection + smooth) / (union  + smooth)
#     return 1 - score
# weighted jaccard
class WeightedJaccardLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, smooth=0.5, name="weighted_jaccard_loss"):
        super().__init__(name=name)
        if not isinstance(class_weights, tf.Tensor):
            class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.class_weights = class_weights
        self.smooth = smooth
    def call(self, y_true, y_pred):
        axis_to_reduce = list(range(1, K.ndim(y_pred)))
        intersection = y_true * y_pred * self.class_weights
        intersection = tf.reduce_sum(intersection, axis=axis_to_reduce)
        union= (y_true + y_pred) * self.class_weights
        union =  tf.reduce_sum(union, axis=axis_to_reduce) 
        score = (intersection + self.smooth) / (union + self.smooth)
        return 1 - score
# # categorical focal loss:  to handle class imbalance without using `class_weights`
# # ref: https://doi.org/10.48550/arXiv.1708.02002 it helps to apply a focal factor to down-weight easy examples and focus more on hard examples.
class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name='general_categorical_focal_crossentropy'):
        super(CategoricalFocalCrossentropy, self).__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = 1e-7
        if self.alpha is None:
            raise ValueError("Alpha should be set before training.")
        y_pred_softmax = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.reduce_sum(y_true * y_pred_softmax, axis=-1)
        p_t_broadcasted = tf.expand_dims(p_t, axis=-1)  # Expand dimensions for broadcast
        focal_loss = -(self.alpha * tf.pow(1 - p_t_broadcasted, self.gamma) * tf.math.log(p_t_broadcasted + epsilon))    
        # Reduce the focal loss along the class dimension
        focal_loss = tf.reduce_sum(focal_loss, axis=-1)
        return focal_loss
    
cfce = CategoricalFocalCrossentropy(alpha=0.25)
# region-based
tversky_loss = TverskyLoss()
dice_loss = TverskyLoss(alpha=0.5, beta=0.5)
jaccard_loss = TverskyLoss(alpha=1, beta=1)

# Compound losses
def cce_dice_loss(y_true, y_pred):
    return cce(y_true, y_pred) + dice_loss(y_true, y_pred)
def cce_jaccard_loss(y_true, y_pred):
    return cce(y_true, y_pred) + jaccard_loss(y_true, y_pred)
def cfce_dice_loss(y_true, y_pred):
    return cfce(y_true, y_pred) + dice_loss(y_true, y_pred)
def cfce_jaccard_loss(y_true, y_pred):
    return cfce(y_true, y_pred) + jaccard_loss(y_true, y_pred)