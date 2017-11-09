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


class LoggingHelper(object):
    def __init__(self, ids, sample_name, dataset, logger, batch_predict, model):
        self.mscans = [dataset.load_mscan(p) for p in ids]
        self.segms = [dataset.load_segm(p) for p in ids]
        self.msegms = [dataset.load_msegm(p) for p in ids]
        self.avg_log_write = logger.make_log_scalar('avg_{}_loss'.format(sample_name))
        self.dices_log_write = make_log_vector(logger, '{}_dices'.format(sample_name))
        self.batch_predict = batch_predict
        self.model = model
        self.dataset = dataset

    def make_record(self):
        msegms_pred = []
        losses = []

        for x, y in zip(self.mscans, self.segms):
            y_pred, loss = self.batch_predict.validate(x, y, validate_fn=self.model.do_val_step)
            msegms_pred.append(self.dataset.segm2msegm(np.argmax(y_pred, axis=0)))
            losses.append(loss)

        avg_loss = np.mean(losses)
        self.avg_log_write(avg_loss)
        dices = [multichannel_dice_score(pred, true) for pred, true in zip(msegms_pred, self.msegms)]
        avg_dice = np.mean(dices, axis=0)
        self.dices_log_write(avg_dice)
        return avg_loss, avg_dice

@register()
def train_segm(model: Model, train_batch_iter_factory: BatchIterFactory, batch_predict: BatchPredict, log_path,
               train_ids, val_ids, dataset, *, n_epochs, lr_init, lr_dec_mul=0.5, patience: int, rtol=0, atol=0):
    logger = Logger(log_path)

    find_next_lr = make_find_next_lr(lr_init, lambda lr: lr * lr_dec_mul,
                                     partial(make_check_loss_decrease, patience=patience, rtol=rtol, atol=atol))

    train_log_write = logger.make_log_scalar('train_loss_old')
    train_avg_log_write = logger.make_log_scalar('avg_train_loss_old')

    val_logging = LoggingHelper(val_ids, "val", dataset, logger, batch_predict, model)
    train_logging = LoggingHelper(train_ids, "train", dataset, logger, batch_predict, model)

    lr = find_next_lr(math.inf)
    with train_batch_iter_factory, logger:
        for i in range(n_epochs):
            with next(train_batch_iter_factory) as train_batch_iter:
                train_losses = []
                for inputs in train_batch_iter:
                    train_losses.append(model.do_train_step(*inputs, lr=lr))
                    train_log_write(train_losses[-1])
                train_avg_log_write(np.mean(train_losses))

            train_logging.make_record()
            val_loss, _ = val_logging.make_record()

            lr = find_next_lr(val_loss)
