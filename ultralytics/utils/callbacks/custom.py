from ultralytics import Callbacks

class EpochCallback(Callbacks):
    _current_epoch = 0
    _last_epoch = -1

    @staticmethod
    def on_pretrain_routine_end(trainer):
        EpochCallback._last_epoch = trainer.epochs

    @staticmethod
    def on_train_epoch_start(trainer):
        EpochCallback._current_epoch += 1