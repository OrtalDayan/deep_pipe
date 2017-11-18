import os
import json

import numpy as np
from tqdm import tqdm

from dpipe.batch_predict import BatchPredict
from dpipe.config import register
from dpipe.dl.model import FrozenModel, Model
from dpipe.medim.metrics import dice_score as dice
from dpipe.medim.metrics import multichannel_dice_score

register_cmd = register(module_type='command')


@register_cmd
def train_model(train, model, save_model_path, restore_model_path=None, transfer_model_path=None):
    if transfer_model_path:
        model.transfer_load(transfer_model_path)
    if restore_model_path:
        model.load(restore_model_path)

    train()
    model.save(save_model_path)


@register_cmd
def transform(input_path, output_path, transform_fn):
    os.makedirs(output_path)

    for f in tqdm(os.listdir(input_path)):
        np.save(os.path.join(output_path, f), transform_fn(np.load(os.path.join(input_path, f))))


@register_cmd
def predict(ids, output_path, load_x, frozen_model: FrozenModel, batch_predict: BatchPredict):
    os.makedirs(output_path)

    for identifier in tqdm(ids):
        x = load_x(identifier)
        y = batch_predict.predict(x, predict_fn=frozen_model.do_inf_step)

        np.save(os.path.join(output_path, str(identifier)), y)
        # saving some memory
        del x, y


@register_cmd
def compute_dices(load_msegm, predictions_path, dices_path):
    dices = {}
    for f in tqdm(os.listdir(predictions_path)):
        patient_id = f.replace('.npy', '')
        y_true = load_msegm(patient_id)
        y = np.load(os.path.join(predictions_path, f))

        dices[patient_id] = multichannel_dice_score(y, y_true)

    with open(dices_path, 'w') as f:
        json.dump(dices, f, indent=0)


@register_cmd
def find_dice_threshold(load_msegm, ids, predictions_path, thresholds_path):
    thresholds = np.linspace(0, 1, 20)
    dices = []

    for patient_id in ids:
        y_true = load_msegm(patient_id)
        y_pred = np.load(os.path.join(predictions_path, f'{patient_id}.npy'))

        # get dice with individual threshold for each channel
        channels = []
        for y_pred_chan, y_true_chan in zip(y_pred, y_true):
            channels.append([dice(y_pred_chan > thr, y_true_chan) for thr in thresholds])
        dices.append(channels)
        # saving some memory
        del y_pred, y_true

    optimal_thresholds = thresholds[np.mean(dices, axis=0).argmax(axis=1)]
    with open(thresholds_path, 'w') as file:
        json.dump(optimal_thresholds.tolist(), file)


@register_cmd
def gleb_metrics_msegm(ids, dataset, dices_path, losses_path, model: Model, batch_predict: BatchPredict, print_stats):
    losses = {}
    dices = {}

    for id in tqdm(ids):
        x, y = dataset.load_mscan(id), dataset.load_msegm(id)
        y_pred, loss = batch_predict.validate(x, y, validate_fn=model.do_val_step)
        # [gleb] this is a hardcoded threshold. Results might be significantly if threshold is optimally computed. Maybe
        # fix it on the next iteration.
        y_pred = y_pred > 0.5
        losses[id] = loss
        dices[id] = multichannel_dice_score(y_pred, y)

    with open(dices_path, 'w') as f:
        json.dump(dices, f, indent=0)

    with open(losses_path, 'w') as f:
        json.dump(losses, f, indent=0)

    if print_stats:
        dices_values = list(dices.values())
        dices_mean, dices_std = np.mean(dices_values, axis=0), np.std(dices_values, axis=0)
        losses_values = list(losses.values())
        losses_mean, losses_std = np.mean(losses_values, axis=0), np.std(losses_values, axis=0)
        print("{}: mean = {}, std = {}".format(dices_path, dices_mean, dices_std))
        print("{}: mean = {}, std = {}".format(losses_path, losses_mean, losses_std))

