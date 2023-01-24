import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from math import pi
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from augmentations import avg_aug
from imutils.paths import list_images
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import det_curve, DetCurveDisplay, roc_curve, RocCurveDisplay

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.metrics import FalsePositives, FalseNegatives, AUC, TruePositives, TrueNegatives
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint

from aikit.graphics.biometric_performance import performance_evaluation
from aikit.graphics.confusion_matrix import (
    plot_confusion_matrix,
    plot_system_confusion_matrix
)
from aikit.graphics.det_plot import DETPlot
from aikit.metadata import __version__ as aikit_ver
from aikit.metrics.det_curve import det_curve_pais, eer_pais
from aikit.metrics.iso_30107_3 import (
    acer,
    apcer_max,
    apcer_pais,
    bpcer,
    bpcer_ap,
    riapar
)
from aikit.metrics.scores import (
    max_error_pais_scores,
    pad_scores,
    split_attack_scores,
    split_scores
)

np.random.seed(42)

parser = argparse.ArgumentParser(description='thesis train file')

# add arguments
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id to use', choices=[0, 1])
parser.add_argument('-f', '--alpha', type=float, default=1.0, help='Only applicable for mnet archs')
parser.add_argument('-e', '--epochs', type=int, default=200, help='Epochs to run')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('-i', '--input_shape', type=int, nargs=2, default=(224, 224), help='Input shape must be two nums')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Learning rate of optimizer')
parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', help='Optimizer', choices=['rmsprop', 'sgd', 'adam'])
parser.add_argument('-a', '--architecture', type=str, default='mnetv2', help='Architecture to be trained', choices=['mnetv2', 'mnetv3S', 'mnetv3L'])
parser.add_argument('-m', '--multiclass', default=False, help='Enables multiclass training', action='store_true')
parser.add_argument('-r', '--arcloss', default=False, help='Enables ArcFace loss training', action='store_true')
parser.add_argument('-p', '--periocular', default=False, help='Enables periocular training', action='store_true')
args = parser.parse_args()

gpu_id = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpu_id}"
alpha = args.alpha
epochs = args.epochs
batch_size = args.batch_size
shape = tuple(args.input_shape)
learning_rate = args.learning_rate
buffer_size = 10101101
arcloss = args.arcloss
multiclass = args.multiclass
periocular = args.periocular
arch = args.architecture # mnetv2 | mnetv3S | mnetv3L | densenet | effnet
optim_str = args.optimizer # rmsprop / sgd / adam


_classes = ('attack', 'bona fide') if not multiclass else ('bona fide', 'printed', 'screen')
_bf_index = _classes.index('bona fide')

if periocular:
    datasets = {
        'flickr': '../data/02_intermediate/flickr-periocular',
        'splunk': '../data/02_intermediate/splunk-periocular'
    }

else:
    datasets = {
        'flickr': '../data/02_intermediate/flickr',
        'splunk': '../data/02_intermediate/splunk'
    }

output_func = 'softmax' if len(_classes) > 2 else 'sigmoid'
loss = 'categorical_crossentropy' if len(_classes) > 2 else 'binary_crossentropy'


def plot_confusion_matrix2(cm, class_names, figsize=(16, 16), fontsize=36, draw_zeros=True, cmap="YlGnBu_r"):
    sns.set_context("notebook", font_scale=4)
    style = {
        'axes.facecolor': "#EAEAF2",
        'axes.edgecolor': "white",
        'axes.labelcolor': ".15",
        'figure.facecolor': "white",
        'figure.edgecolor': "white",
        'grid.color': "white",
        'grid.linestyle': "-",
        'legend.facecolor': ".8",
        'legend.edgecolor': ".8",
        'text.color': ".15",
        'xtick.color': ".15",
        'ytick.color': ".15",
        'font.family': "sans-serif",
        'font.sans-serif': "Helvetica",
        'savefig.facecolor': "#EAEAF2",
        'savefig.edgecolor': "#white",
        'savefig.transparent': True,
        'eer.color': ".66"
    }
    sns.set_style(style)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    h, w = cm.shape[:2]
    for i in range(w):
        for j in range(h):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = f"{c}/{s[0]}\n{p:.1f}%"
            elif c == 0 and not draw_zeros:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{c}\n{p:.1f}%"
    cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm.index.name = "Ground Truth"
    cm.columns.name = "Predicted"
    _, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(cm, cmap=cmap, annot=annot, fmt="",
                          annot_kws={'size': fontsize}, ax=ax)
    fig = heatmap.get_figure()
    fig.patch.set_alpha(0)

    return fig

def _parse_function_with_DA(impath, label):
    # instantiate augmentation here to overcome "not-changing-random-seed" bug
    # this could lead to an speed decrease or an overheap, dont know
    aug = avg_aug() # think about deleting aug after use
    
    # image reading
    image = tf.io.read_file(impath)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, shape)
    image = tf.cast(image, tf.uint8)
    image = tf.numpy_function(func=aug.augment_image, inp=[image], Tout=tf.uint8)
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image, label

def _parse_function_without_DA(impath, label):
    # image reading
    image_string = tf.io.read_file(impath)
    image_decoded = tf.io.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.image.resize(image, shape)
    image = image / 255.
    return image, label

# return tf.data
def load_dataset(
    db_dict=None, db_key='flickr', multiclass=False, 
    data_augmentation=False, shape=(224, 224), color='rgb', 
    load_csv=False, return_class_weights=False, batch_size=16
):
    
    db_dir = db_dict.get(db_key)
    _classes = ('attack', 'bonafide') if not multiclass else ('bonafide', 'print', 'screen')
    imlist = [*list_images(db_dir)]
    
    if multiclass:
        # generate multiclass_y vector
        X, y = [], []
        regex_str = "|".join(_classes)
        
        for imdir in imlist:
            regex_res = re.search(regex_str, imdir)
            regex_class = regex_res.group()
            _id_class = _classes.index(regex_class)
            X.append(imdir)
            y.append(_id_class)
        
        X = np.array(X)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights = dict(enumerate(class_weights))
        y = tf.keras.utils.to_categorical(y)

    else:
        bf_list, att_list = [], []
        for imdir in imlist:
            if _classes[_bf_index] in imdir.split(os.sep): # if 'bonafide' in imdir
                bf_list.append(imdir)
            else: # identify attack class
                att_list.append(imdir)
                
        bf_y = np.ones(len(bf_list))
        att_y = np.zeros(len(att_list)) # use labelencoder here
        # split|load data into train|test|val
        X = np.concatenate([att_list, bf_list])
        y = np.concatenate([att_y, bf_y])
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights = dict(enumerate(class_weights))
        y = tf.keras.utils.to_categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
    
    # train dataset
    train_dataset = tf.cast(X_train, dtype=tf.string)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset, tf.cast(y_train, tf.float32)))
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.map(_parse_function_with_DA, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.batch(batch_size)
    
    # val dataset
    val_dataset = tf.cast(X_val, dtype=tf.string)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_dataset, tf.cast(y_val, tf.float32)))
    val_dataset = val_dataset.shuffle(buffer_size)
    val_dataset = val_dataset.map(_parse_function_without_DA, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(1)
    
    # test dataset
    test_dataset = tf.cast(X_test, dtype=tf.string)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset, tf.cast(y_test, tf.float32)))
    test_dataset = test_dataset.shuffle(buffer_size)
    test_dataset = test_dataset.map(_parse_function_without_DA, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.batch(1)
    
    if return_class_weights:
        return train_dataset, val_dataset, test_dataset, class_weights
    else:
        return train_dataset, val_dataset, test_dataset, None


# Define ArchLoss
# ref: https://www.kaggle.com/code/chankhavu/keras-layers-arcface-cosface-adacos/notebook7
# ref: https://github.com/yinguobing/arcface/blob/main/train.py

class L2Normalization(tf.keras.layers.Layer):
    """This layer normalizes the inputs with l2 normalization."""

    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        inputs = tf.nn.l2_normalize(inputs, axis=1)

        return inputs

    def get_config(self):
        config = super().get_config()
        return config

class ArcLayer(tf.keras.layers.Layer):
    """Custom layer for ArcFace.
    This layer is equivalent a dense layer except the weights are normalized.
    """

    def __init__(self, units, kernel_regularizer=None, **kwargs):
        super(ArcLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-1], self.units],
                                      dtype=tf.float32,
                                      initializer=tf.keras.initializers.HeNormal(),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='kernel')
        self.built = True

    @tf.function
    def call(self, inputs):
        weights = tf.nn.l2_normalize(self.kernel, axis=0)
        return tf.matmul(inputs, weights)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units,
                       "kernel_regularizer": self.kernel_regularizer})
        return config

class ArcLoss(tf.keras.losses.Loss):
    """Additive angular margin loss.
    Original implementation: https://github.com/luckycallor/InsightFace-tensorflow
    """

    def __init__(self, margin=0.5, scale=64, name="arcloss"):
        """Build an additive angular margin loss object for Keras model."""
        super().__init__(name=name)
        self.margin = margin
        self.scale = scale
        self.threshold = tf.math.cos(pi - margin)
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)

        # Safe margin: https://github.com/deepinsight/insightface/issues/108
        self.safe_margin = self.sin_m * margin

    @tf.function
    def call(self, y_true, y_pred):

        # Calculate the cosine value of theta + margin.
        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))

        cos_t_margin = tf.where(cos_t > self.threshold,
                                cos_t * self.cos_m - sin_t * self.sin_m,
                                cos_t - self.safe_margin)

        # The labels here had already been onehot encoded.
        mask = y_true
        cos_t_onehot = cos_t * mask
        cos_t_margin_onehot = cos_t_margin * mask

        # Calculate the final scaled logits.
        logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * self.scale

        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, logits)

        return losses

    def get_config(self):
        config = super(ArcLoss, self).get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config

# train block
with tf.device(f'/gpu:{gpu_id}'):
    sess = tf.compat.v1.Session()

    # Make dataset
    train_dataset, val_dataset, test_dataset, class_weights = load_dataset(
        db_dict=datasets, db_key='flickr', 
        batch_size=batch_size, multiclass=multiclass, 
        return_class_weights=True
    )

    # # Define arch model
    input_shape = (*shape, 3)

    archs = {
        'mnetv2': MobileNetV2(input_shape=input_shape, alpha=alpha, include_top=False, weights=None, pooling='avg'),
        'densenet':DenseNet121(input_shape=input_shape, include_top=False, pooling='max', weights=None),
        'mnetv3S': MobileNetV3Small(input_shape=input_shape, alpha=alpha, include_top=False, weights=None, pooling='avg', dropout_rate=0.25),
        'mnetv3L': MobileNetV3Large(input_shape=input_shape, alpha=alpha, include_top=False, weights=None, pooling='avg', dropout_rate=0.5),
        'effnet': EfficientNetB0(input_shape=input_shape, include_top=False, pooling='max', weights=None),
    }

    model = archs.get(arch)

    if arcloss:
        x = model.output
        x = L2Normalization()(x)
        x = ArcLayer(
            units=len(_classes), 
            activity_regularizer=tf.keras.regularizers.L1(1e-4),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        )(x)
        loss = ArcLoss()
        model = Model(inputs=model.input, outputs=x)
        
    else:
        x = model.output
        x = Dense(
            units=len(_classes), activation=output_func, 
            use_bias=True, name="Logits", 
            activity_regularizer=tf.keras.regularizers.L1(1e-4),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        )(x)
        model = Model(inputs=model.input, outputs=x)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=15, min_lr=1e-15, verbose=True),
        EarlyStopping(monitor='val_loss', patience=30, verbose=1),
        ModelCheckpoint(
            filepath='models/E{epoch:03d}-{val_loss:.4f}.hdf5',
            monitor='val_loss',
            mode='min',
        )
    ]

    # compile model
    opts = {
        'rmsprop': RMSprop(learning_rate=learning_rate, momentum=0.9, decay=1e-4),
        'sgd': SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True, decay=1e-4),
        'adam': Adam(learning_rate=learning_rate, amsgrad=True),
    }

    optimizer = opts.get(optim_str)

    model.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=[
            'acc', 
        ], 
    )

    with warnings.catch_warnings():
        warnings.warn("deprecated", DeprecationWarning)
        warnings.simplefilter("ignore")
        history = model.fit(
            train_dataset, callbacks=callbacks,
            epochs=epochs, max_queue_size=32, 
            use_multiprocessing=True, workers=4,
            validation_data=val_dataset, shuffle=True,
            class_weight=class_weights
        )

# define a plot function to plot history scores
def plot_history(history, title, save_path, figsize=(12, 6), font_scale=1.5, linewidth=3.5):
    with sns.plotting_context(
        "notebook", font_scale=font_scale, rc={"lines.linewidth": linewidth}
    ):
        epochs = len(history.history["val_loss"])
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_ylabel("Loss/Acc.")
        ax.set_xlabel("Epochs")
        plt.ylim([-0.1, 1.1])
        sns.lineplot(history.history["loss"], label="Train Loss", ax=ax, linestyle='--')
        sns.lineplot(history.history["val_loss"], label="Val. Loss", ax=ax, linestyle='--')
        sns.lineplot(history.history["acc"], label="Train Acc.", ax=ax, linestyle='--')
        sns.lineplot(history.history["val_acc"], label="Val. Acc.", ax=ax, linestyle='--')
        fig.savefig(save_path)


# call plot function
plot_history(history, "Model Training Loss & Acc.", "train_plot.png")


# # Eval Test set

y_true, y_pred, y_score = [], [], []

for im, lb in tqdm(test_dataset):
    lb = np.argmax(lb, axis=-1)[0]
    scores = model.predict(im)[0]
    pred = np.argmax(scores)
    
    y_true.append(lb)
    y_pred.append(pred)
    y_score.append(scores[_bf_index])

y_true, y_pred, y_score = np.array(y_true), np.array(y_pred), np.array(y_score)
clf = classification_report(y_true=y_true, y_pred=y_pred, digits=4, target_names=_classes)
cm = confusion_matrix(y_true, y_pred)
cm_fig = plot_confusion_matrix2(cm, _classes)
# plt.plot()

if arcloss: # normalize scores
    y_score = np.nan_to_num(y_score)
    y_score = (y_score - np.min(y_score))/np.ptp(y_score)


# get ISO 30107-3 metrics
_no_bf = list(set(np.unique(y_true)) - set([_bf_index, ]))
det = DETPlot(context="notebook")

for i in _no_bf:
    bonafide_scores = y_score[ y_true == _bf_index ]
    attack_scores = y_score[ y_true == i ]
    det.set_system(attack_scores, bonafide_scores, label=f"{_classes[i]}")
    
    
det_plot = det.plot()

# TODO: get worst ths for worst eer
for system in det.systems:
    ths = det.systems[system]['eer_thres']
    eer = det.systems[system]['eer']
    print(f'{system} EER: {eer}')
    print(f'{system} EER Threshold: {ths}\n')

threshold = ths
attack_scores, bonafide_scores, attack_true, bonafide_true = split_scores(y_true, y_score, bonafide_label=_bf_index)
pais_attack_scores = split_attack_scores(attack_true, attack_scores)

det_pais = det_curve_pais(attack_true, attack_scores, bonafide_scores)
eer_pais_ = eer_pais(det_pais, percentage=True)

max_eer_pais = max(eer_pais_, key=eer_pais_.get)
max_attack_scores, max_attack_pais = max_error_pais_scores(attack_true, attack_scores, threshold=threshold)

acer_ = acer(attack_true, attack_scores, bonafide_scores, threshold=threshold)
apcer_ = apcer_pais(attack_true, attack_scores, threshold=threshold, percentage=True)
bpcer_ = bpcer(bonafide_scores, threshold=threshold)
bpcer10, bpcer10thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 10, percentage=True)
bpcer20, bpcer20thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 20, percentage=True)
bpcer50, bpcer50thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 50, percentage=True)
bpcer100, bpcer100thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 100, percentage=True)
bpcer200, bpcer200thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 200, percentage=True)
bpcer500, bpcer500thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 500, percentage=True)
bpcer1000, bpcer1000thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 1000, percentage=True)
bpcer10000, bpcer10000thres = bpcer_ap(det_pais[max_eer_pais][0], det_pais[max_eer_pais][1], det_pais[max_eer_pais][2], 10000, percentage=True)
riapar_ = riapar(max_attack_scores, bonafide_scores, attack_threshold=threshold, bonafide_threshold=threshold)

# save all this metrics into a readable 
print(
    f"        Bona Fide label: {_bf_index}: {_classes[_bf_index]}\n"
    f"            Threshold t: {threshold}\n"
    "--------------------------------------------\n"
    f"           Max EER PAIS: {max_eer_pais}: {_classes[max_eer_pais]}\n"
    f"                 EER[{max_eer_pais}]: {eer_pais_[max_eer_pais][0]}%\n"
    f"       EER threshold[{max_eer_pais}]: {eer_pais_[max_eer_pais][1]}\n"
    "--------------------------------------------\n"
    f"      Max APCER PAIS(t): {max_attack_pais}: {_classes[max_attack_pais]}\n"
    f"                ACER(t): {acer_*100}%\n"
    f"               APCER(t): {apcer_}%\n"
    f"               BPCER(t): {bpcer_*100}%\n"
    f"              RIAPAR(t): {riapar_*100}%\n"
    f"                 EER[{max_attack_pais}]: {eer_pais_[max_attack_pais][0]}%\n"
    f"       EER threshold[{max_attack_pais}]: {eer_pais_[max_attack_pais][1]}\n"
    "--------------------------------------------\n"
    f"   BPCER10(APCER=10.0%): {bpcer10}%, {bpcer10thres}\n"
    f"   BPCER20(APCER=5.00%): {bpcer20}%, {bpcer20thres}\n"
    f"   BPCER50(APCER=2.00%): {bpcer50}%, {bpcer50thres}\n"
    f"  BPCER100(APCER=1.00%): {bpcer100}%, {bpcer100thres}\n"
    f"   BPCER200(APCER=0.50%): {bpcer200}%, {bpcer200thres}\n"
    f"   BPCER500(APCER=0.20%): {bpcer500}%, {bpcer500thres}\n"
    f" BPCER1000(APCER=0.10%): {bpcer1000}%, {bpcer1000thres}\n"
    f"BPCER10000(APCER=0.01%): {bpcer10000}%, {bpcer10000thres}\n"
)