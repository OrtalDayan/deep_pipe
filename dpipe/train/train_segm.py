import math
from functools import partial

import numpy as np
from tensorboard_easy.logger import Logger

from dpipe.batch_iter_factory import BatchIterFactory
from dpipe.batch_predict import BatchPredict
from dpipe.config import register
from dpipe.dl.model import Model
from dpipe.medim.metrics import multichannel_dice_score
from .logging import make_log_vector
from .utils import make_find_next_lr, make_check_loss_decrease

import time

class SampleStatsLogger:
    def __init__(self, sample_name, ids, logger, dataset, validate_fn):
        self.ids = ids
        self.mscans = [dataset.load_mscan(p) for p in ids]
        self.segms = [dataset.load_segm(p) for p in ids]
        self.msegms = [dataset.load_msegm(p) for p in ids]
        self.avg_log_write = logger.make_log_scalar('avg_{}_loss_gleb'.format(sample_name))
        self.dices_log_write = make_log_vector(logger, '{}_dices_gleb'.format(sample_name))
        self.dataset = dataset
        self.validate_fn = validate_fn

    def make_record(self):
        losses = []
        dices = []
        for mscan, segm, msegm in zip(self.mscans, self.segms, self.msegms):
            y_pred, loss = self.validate_fn(mscan, segm)
            msegm_pred = self.dataset.segm2msegm(np.argmax(y_pred, axis=0))
            losses.append(loss)
            dices.append(multichannel_dice_score(msegm_pred, msegm))
        print("glebgleb: in a sample stats logger. Making a record.")
        self.avg_log_write(np.mean(losses))
        self.dices_log_write(np.mean(dices, axis=0))

@register()
def train_segm(model: Model, train_batch_iter_factory: BatchIterFactory, batch_predict: BatchPredict, log_path, val_ids,
               dataset, *, n_epochs, lr_init, lr_dec_mul=0.5, patience: int, rtol=0, atol=0, train_ids=None):
    logger = Logger(log_path)

    mscans_val = [dataset.load_mscan(p) for p in val_ids]
    segms_val = [dataset.load_segm(p) for p in val_ids]
    msegms_val = [dataset.load_msegm(p) for p in val_ids]

    find_next_lr = make_find_next_lr(lr_init, lambda lr: lr * lr_dec_mul,
                                     partial(make_check_loss_decrease, patience=patience, rtol=rtol, atol=atol))

    train_log_write = logger.make_log_scalar('train_loss')
    train_avg_log_write = logger.make_log_scalar('avg_train_loss')
    val_avg_log_write = logger.make_log_scalar('avg_val_loss')
    val_dices_log_write = make_log_vector(logger, 'val_dices')

    validate_fn = partial(batch_predict.validate, validate_fn=model.do_val_step)
    val_logger = SampleStatsLogger("val", val_ids, logger, dataset, validate_fn)
    train_logger = None if train_ids is None else SampleStatsLogger("train", train_ids, logger, dataset, validate_fn)

    lr = find_next_lr(math.inf)
    with train_batch_iter_factory, logger:
        for i in range(n_epochs):
            print("glebgleb epoch = {}".format(i))
            st_time = time.time()
            with next(train_batch_iter_factory) as train_batch_iter:
                train_losses = []
                for inputs in train_batch_iter:
                    train_losses.append(model.do_train_step(*inputs, lr=lr))
                    train_log_write(train_losses[-1])
                train_avg_log_write(np.mean(train_losses))
            print("glebgleb spent {} s training on a batch".format(time.time() - st_time))

            print("glebgleb moving to making val records in an old fashion")
            st_time = time.time()
            msegms_pred = []
            val_losses = []
            for x, y in zip(mscans_val, segms_val):
                y_pred, loss = batch_predict.validate(x, y, validate_fn=model.do_val_step)
                msegms_pred.append(dataset.segm2msegm(np.argmax(y_pred, axis=0)))
                val_losses.append(loss)

            val_loss = np.mean(val_losses)
            val_avg_log_write(val_loss)
            val_dices = [multichannel_dice_score(pred, true) for pred, true in zip(msegms_pred, msegms_val)]
            val_dices_log_write(np.mean(val_dices, axis=0))
            print("glebgleb spent {} s making on making val records in an old fashion".format(time.time() - st_time))

            print("glebgleb moving to making a val record in a new fashion")
            st_time = time.time()
            val_logger.make_record()
            print("glebgleb spent {} s on making a val record in a new fashion".format(time.time() - st_time))

            if train_logger:
                print("glebgleb moving to making a train record in a new fashion")
                st_time = time.time()
                train_logger.make_record()
                print("glebgleb spent {} s on making a train record in a new fashion".format(time.time() - st_time))

            lr = find_next_lr(val_loss)
