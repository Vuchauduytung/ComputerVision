import tensorflow.keras as keras


class CustomCallback(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(CustomCallback, self).__init__()
        self.patience = patience
        
    def on_epoch_end(self, epoch, logs=None):
        self.stopped_epoch = epoch
        # if (epoch - 1) % 100 == 0:
        #     stop = input('Continue train [y/n]:')
        #     if stop.lower() == 'n':
        #         self.model.stop_training = True
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
                    