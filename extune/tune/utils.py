import tensorflow


class TuneKerasCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, reporter, config, logs={}):
        self.reporter = reporter
        self.config = config
        self.iteration = 0
        self.logs_copy = {}
        super(TuneKerasCallback, self).__init__()

    def on_train_end(self, logs={}):
        print(logs)
        print(self.logs_copy)
        self.reporter(
            timesteps_total=self.iteration, done=1,
            val_metric=(1.0 - float(self.logs_copy["val_" + self.config.METRIC[self.config.MONITORED_METRIC]])))

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        self.logs_copy = logs
        self.iteration += 1
        self.reporter(
            timesteps_total=self.iteration,
            val_metric=(1.0 - float(self.logs_copy["val_" + self.config.METRIC[self.config.MONITORED_METRIC]])))
