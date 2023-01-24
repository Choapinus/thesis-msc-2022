import os
import json
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from utils import (
    NpEncoder,
    L2Normalization,
    ArcLayer,
    ArcLoss,
    load_dataset,
    plot_history,
    plot_confusion_matrix2,
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

from aikit.graphics.biometric_performance import performance_evaluation
from aikit.graphics.confusion_matrix import (
    plot_confusion_matrix,
    plot_system_confusion_matrix,
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
    riapar,
)
from aikit.metrics.scores import (
    max_error_pais_scores,
    pad_scores,
    split_attack_scores,
    split_scores,
)

np.random.seed(42)

# add arguments
parser = argparse.ArgumentParser(description="MSC-PAD train file")
parser.add_argument(
    "-g", "--gpu", type=int, default=0, help="GPU id to use", choices=[0, 1]
)
parser.add_argument(
    "-f", "--alpha", type=float, default=1.0, help="Only applicable for mnet archs"
)
parser.add_argument("-e", "--epochs", type=int, default=200, help="Epochs to run")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
parser.add_argument(
    "-i",
    "--input_shape",
    type=int,
    nargs=2,
    default=(224, 224),
    help="Input shape must be two nums",
)
parser.add_argument(
    "-l", "--learning_rate", type=float, default=1e-4, help="Learning rate of optimizer"
)
parser.add_argument(
    "-o",
    "--optimizer",
    type=str,
    default="rmsprop",
    help="Optimizer",
    choices=["rmsprop", "sgd", "adam"],
)
parser.add_argument(
    "-a",
    "--architecture",
    type=str,
    default="mnetv2",
    help="Architecture to be trained",
    choices=["mnetv2", "mnetv3S", "mnetv3L"],
)
parser.add_argument(
    "-m",
    "--multiclass",
    default=False,
    help="Enables multiclass training",
    action="store_true",
)
parser.add_argument(
    "-r",
    "--arcloss",
    default=False,
    help="Enables ArcFace loss training",
    action="store_true",
)
parser.add_argument(
    "-p",
    "--periocular",
    default=False,
    help="Enables periocular training",
    action="store_true",
)
args = parser.parse_args()

# TODO: make a set of experiments
gpu_id = args.gpu
alpha = args.alpha
epochs = args.epochs
batch_size = args.batch_size
shape = tuple(args.input_shape)
learning_rate = args.learning_rate
arcloss = args.arcloss
multiclass = args.multiclass
periocular = args.periocular
arch = args.architecture  # mnetv2 | mnetv3S | mnetv3L | densenet | effnet
optim_str = args.optimizer  # rmsprop / sgd / adam

_classes = ("attack", "bona fide") if not multiclass else ("bona fide", "printed", "screen")
_bf_index = _classes.index("bona fide")

if periocular:
    datasets = {
        "flickr": "../data/02_intermediate/flickr-periocular",
        "splunk": "../data/02_intermediate/splunk-periocular",
    }

else:
    datasets = {
        "flickr": "../data/02_intermediate/flickr",
        "splunk": "../data/02_intermediate/splunk",
    }

output_func = "softmax" if len(_classes) > 2 else "sigmoid"
loss = "categorical_crossentropy" if len(_classes) > 2 else "binary_crossentropy"

# train block
with tf.device(f"/gpu:{gpu_id}"):
    sess = tf.compat.v1.Session()

    # Make dataset
    train_dataset, val_dataset, test_dataset, class_weights = load_dataset(
        db_dict=datasets,
        db_key="flickr",
        shape=shape,
        batch_size=batch_size,
        multiclass=multiclass,
        return_class_weights=True,
    )

    # # Define arch model
    input_shape = (*shape, 3)

    archs = {
        "mnetv2": MobileNetV2(
            input_shape=input_shape,
            alpha=alpha,
            include_top=False,
            weights=None,
            pooling="avg",
        ),
        "densenet": DenseNet121(
            input_shape=input_shape, include_top=False, pooling="max", weights=None
        ),
        "mnetv3S": MobileNetV3Small(
            input_shape=input_shape,
            alpha=alpha,
            include_top=False,
            weights=None,
            pooling="avg",
            dropout_rate=0.25,
        ),
        "mnetv3L": MobileNetV3Large(
            input_shape=input_shape,
            alpha=alpha,
            include_top=False,
            weights=None,
            pooling="avg",
            dropout_rate=0.5,
        ),
        "effnet": EfficientNetB0(
            input_shape=input_shape, include_top=False, pooling="max", weights=None
        ),
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
            units=len(_classes),
            activation=output_func,
            use_bias=True,
            name="Logits",
            activity_regularizer=tf.keras.regularizers.L1(1e-4),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        )(x)
        model = Model(inputs=model.input, outputs=x)

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.25, patience=15, min_lr=1e-15, verbose=True
        ),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1),
        ModelCheckpoint(
            filepath="models/E{epoch:03d}-{val_loss:.4f}.hdf5",  # TODO: change filepath name
            monitor="val_loss",
            mode="min",
        ),
    ]

    # compile model
    opts = {
        "rmsprop": RMSprop(learning_rate=learning_rate, momentum=0.9, decay=1e-4),
        "sgd": SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True, decay=1e-4
        ),
        "adam": Adam(learning_rate=learning_rate, amsgrad=True, epsilon=1e-3),
    }

    optimizer = opts.get(optim_str)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "acc",
        ],
    )

    with warnings.catch_warnings():
        warnings.warn("deprecated", DeprecationWarning)
        warnings.simplefilter("ignore")
        history = model.fit(
            train_dataset,
            callbacks=callbacks,
            epochs=epochs,
            max_queue_size=32,
            use_multiprocessing=True,
            workers=4,
            validation_data=val_dataset,
            shuffle=True,
            class_weight=class_weights,
        )

    # call plot function
    plot_history(
        history, "Model Training Loss & Acc.", "train_plot.png", arcloss=arcloss
    )  # TODO: change history plot name

    # # Eval Test set block
    y_true, y_pred, y_score = [], [], []

    for im, lb in tqdm(test_dataset, desc="Predicting over test dataset"):
        lb = np.argmax(lb, axis=-1)[0]
        scores = model.predict(im)[0]
        pred = np.argmax(scores)  # TODO: check this

        y_true.append(lb)
        y_pred.append(pred)
        y_score.append(scores[_bf_index])
        # TODO: store scores?

if arcloss:  # normalize scores
    y_score = np.nan_to_num(y_score)
    y_score = (y_score - np.min(y_score)) / np.ptp(y_score)

# make confusion matrix
y_true, y_pred, y_score = np.array(y_true), np.array(y_pred), np.array(y_score)
clf = classification_report(
    y_true=y_true, y_pred=y_pred, digits=4, target_names=_classes
)
cm = confusion_matrix(y_true, y_pred)
cm_fig = plot_confusion_matrix2(
    cm, _classes
)  # TODO: change this with cm of aikit and check if multiclass can be ploted as att vs bon

# get ISO 30107-3 metrics and DET plot
_no_bf = list(
    set(np.unique(y_true))
    - set(
        [
            _bf_index,
        ]
    )
)
det = DETPlot(context="notebook")

for i in _no_bf:
    bonafide_scores = y_score[y_true == _bf_index]
    attack_scores = y_score[y_true == i]
    det.set_system(attack_scores, bonafide_scores, label=f"{_classes[i]}")

det_plot = det.plot()

ths_list, eer_list = [], []
for system in det.systems:
    ths = det.systems[system]["eer_thres"]
    eer = det.systems[system]["eer"]
    # print(f'{system} EER: {eer}')
    # print(f'{system} EER Threshold: {ths}\n')
    ths_list.append(ths)
    eer_list.append(eer)

threshold = ths_list[np.argmax(eer_list)]
attack_scores, bonafide_scores, attack_true, bonafide_true = split_scores(
    y_true, y_score, bonafide_label=_bf_index
)
pais_attack_scores = split_attack_scores(attack_true, attack_scores)

det_pais = det_curve_pais(attack_true, attack_scores, bonafide_scores)
eer_pais_ = eer_pais(det_pais, percentage=True)

max_eer_pais = max(eer_pais_, key=eer_pais_.get)
max_attack_scores, max_attack_pais = max_error_pais_scores(
    attack_true, attack_scores, threshold=threshold
)

acer_ = acer(attack_true, attack_scores, bonafide_scores, threshold=threshold)
apcer_ = apcer_pais(attack_true, attack_scores, threshold=threshold, percentage=True)
bpcer_ = bpcer(bonafide_scores, threshold=threshold)
bpcer10, bpcer10thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    10,
    percentage=True,
)
bpcer20, bpcer20thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    20,
    percentage=True,
)
bpcer50, bpcer50thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    50,
    percentage=True,
)
bpcer100, bpcer100thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    100,
    percentage=True,
)
bpcer200, bpcer200thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    200,
    percentage=True,
)
bpcer500, bpcer500thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    500,
    percentage=True,
)
bpcer1000, bpcer1000thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    1000,
    percentage=True,
)
bpcer10000, bpcer10000thres = bpcer_ap(
    det_pais[max_eer_pais][0],
    det_pais[max_eer_pais][1],
    det_pais[max_eer_pais][2],
    10000,
    percentage=True,
)
riapar_ = riapar(
    max_attack_scores,
    bonafide_scores,
    attack_threshold=threshold,
    bonafide_threshold=threshold,
)

# save all this metrics into a readable
metrics_and_info = {
    "MODEL": arch,
    "ARCLOSS": arcloss,
    "CLASSES": _classes,
    "LR": learning_rate,
    "OPTIMIZER": optim_str,
    "THRESHOLD": threshold,
    "MULTICLASS": multiclass,
    "PERIOCULAR": periocular,
    "BONAFIDE": {"index": _bf_index, "classname": _classes[_bf_index]},
    "EER": {"classid": max_eer_pais, "value": eer_pais_[max_eer_pais][0]},
    "MAX EER PAIS": {"classid": max_eer_pais, "classname": _classes[max_eer_pais]},
    "EER_THRESHOLD": {"classid": max_eer_pais, "value": eer_pais_[max_eer_pais][1]},
    "MAX APCER PAIS": {
        "classid": max_attack_pais,
        "classname": _classes[max_attack_pais],
    },
    "ACER": acer_ * 100,
    "APCER": {_classes[k]: apcer_[k] for k in apcer_.keys()},
    "BPCER": bpcer_ * 100,
    "RIAPAR": riapar_ * 100,
    "BPCER10": {"value": bpcer10, "threshold": bpcer10thres},
    "BPCER20": {"value": bpcer20, "threshold": bpcer20thres},
    "BPCER50": {"value": bpcer50, "threshold": bpcer50thres},
    "BPCER100": {"value": bpcer100, "threshold": bpcer100thres},
    "BPCER200": {"value": bpcer200, "threshold": bpcer200thres},
    "BPCER500": {"value": bpcer500, "threshold": bpcer500thres},
    "BPCER1000": {"value": bpcer1000, "threshold": bpcer1000thres},
    "BPCER10000": {"value": bpcer10000, "threshold": bpcer10000thres},
}

# TODO: change names of metrics file stored
with open("metrics.json", "w", encoding="utf8") as json_file:
    json.dump(metrics_and_info, json_file, cls=NpEncoder)

# store plots?
# cm_fig
# det_plot
