import tensorflow as tf
import keras   
import keras.backend as K   
import warnings
import numpy as np
import matplotlib.pyplot as plt

class SnapshotEnsemble(keras.callbacks.Callback):
  def __init__(self, epochs_per_cycle, lrate_max, delay=0.8, Tmult=2, verbose=0):
    self.epochs_per_cycle = epochs_per_cycle
    self.lr_max = lrate_max
    self.delay = delay
    self.Tmult = Tmult
    self.lrates = list()
    self.epochs = 0

  def cosine_annealing(self, epoch, epochs_per_cycle, lrate_max):
    res=[]
    cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
    for lrate in lrate_max:
        res.append(lrate/2 * (np.cos(cos_inner) + 1))
    return res

  def on_epoch_begin(self, epoch, logs={}):
    lr = self.cosine_annealing(epoch-self.epochs, self.epochs_per_cycle, self.lr_max)
    print(f'epoch {epoch+1}, lr {lr}')
    if len(lr) == 1:
        K.set_value(self.model.optimizer.lr, lr[0])
    else:
        for i in range(len(lr)):
            K.set_value(self.model.optimizer.lr[i], lr[i])   
    self.lrates.append(lr)

  def on_epoch_end(self, epoch, logs={}):
    if epoch != 0 and (epoch + 1 - self.epochs) % self.epochs_per_cycle == 0:
        self.epochs += self.epochs_per_cycle
        self.epochs_per_cycle = int(self.epochs_per_cycle * self.Tmult)
        self.lr_max = [lr * self.delay for lr in self.lr_max]
        print(f'epoch {epoch + 1} reset epochs_per_cycle {self.epochs_per_cycle} lr_max {self.lr_max}')

class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    '''Schedule learning rates with restarts
     A simple restart technique for stochastic gradient descent.
    The learning rate decays after each batch and peridically resets to its
    initial value. Optionally, the learning rate is additionally reduced by a
    fixed factor at a predifined set of epochs.
     # Arguments
        epochsize: Number of samples per epoch during training.
        batchsize: Number of samples per batch during training.
        start_epoch: First epoch where decay is applied.
        epochs_to_restart: Initial number of epochs before restarts.
        mult_factor: Increase of epochs_to_restart after each restart.
        lr_fac: Decrease of learning rate at epochs given in
                lr_reduction_epochs.
        lr_reduction_epochs: Fixed list of epochs at which to reduce
                             learning rate.
     # References
        - [SGDR: Stochastic Gradient Descent with Restarts](http://arxiv.org/abs/1608.03983)
    '''
    def __init__(self,
                 batches_per_epoch,
                 epochs_to_restart=2,
                 mult_factor=2,
                 lr_fac=0.1,
                 lr_reduction_epochs=(60, 120, 160)):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.epoch = -1
        self.batch_since_restart = 0
        self.next_restart = epochs_to_restart
        self.epochs_to_restart = epochs_to_restart
        self.mult_factor = mult_factor
        self.batches_per_epoch = batches_per_epoch
        self.lr_fac = lr_fac
        self.lr_reduction_epochs = lr_reduction_epochs
        self.lr_log = []

    def on_train_begin(self, logs={}):
        print(K.get_value(self.model.optimizer.lr))
        self.lr = K.get_value(self.model.optimizer.lr)[0]

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        fraction_to_restart = self.batch_since_restart / \
            (self.batches_per_epoch * self.epochs_to_restart)
        lr = 0.5 * self.lr * (1 + np.cos(fraction_to_restart * np.pi))
        K.set_value(self.model.optimizer.lr[0], lr)

        self.batch_since_restart += 1
        self.lr_log.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.epochs_to_restart *= self.mult_factor
            self.next_restart += self.epochs_to_restart

        if (self.epoch + 1) in self.lr_reduction_epochs:
            self.lr *= self.lr_fac

class CyclicLR(keras.callbacks.Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """
 
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
 
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
 
        self._reset()
 
    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
 
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)
 
    def on_train_begin(self, logs={}):
        logs = logs or {}
 
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr[0], self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr[0], self.clr())
 
    def on_batch_end(self, epoch, logs=None):
 
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
 
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr)[0])
        self.history.setdefault('iterations', []).append(self.trn_iterations)
 
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
 
        K.set_value(self.model.optimizer.lr[0], self.clr())


class LR_Updater(keras.callbacks.Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''
    
    def __init__(self, iterations):
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''
        self.epoch_iterations = iterations
        self.trn_iterations = 0.
        self.history = {}
        
    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr[0], self.setRate())
#        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
#        self.history.setdefault('iterations', []).append(self.trn_iterations)
#        for k, v in logs.items():
#            self.history.setdefault(k, []).append(v)
    
    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.history['iterations'], self.history['lr'])
    
    def plot(self, n_skip=10):
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')

        
class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement cyclical learning rates
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    '''
    
    def __init__(self, iterations, cycle_mult = 1):
        '''
        iterations = dataset size / batch size
        iterations = number of iterations in one annealing cycle
        cycle_mult = used to increase the cycle length cycle_mult times after every cycle
        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$
        '''
        self.min_lr = 0
        self.cycle_mult = cycle_mult
        self.cycle_iterations = 0.
        super().__init__(iterations)
    
    def setRate(self):
        self.cycle_iterations += 1
        if self.cycle_iterations == self.epoch_iterations:
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
        cos_out = np.cos(np.pi*(self.cycle_iterations)/self.epoch_iterations) + 1
        return self.max_lr / 2 * cos_out
    
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={}) #changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)[0]


class ReduceLROnPlateau_dlr(keras.callbacks.Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Example

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=[0,0],
                 **kwargs):
        super(ReduceLROnPlateau_dlr, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = K.get_value(self.model.optimizer.lr)
#                    old_lr1 = float(K.get_value(self.model.optimizer.lr)[1])
                    for i in range(len(old_lr)):
                        if old_lr[i] > self.min_lr[i]:
                            new_lr = float(old_lr[i]) * self.factor
                            new_lr = max(new_lr, self.min_lr[i])
                            K.set_value(self.model.optimizer.lr[i], new_lr)
                            if self.verbose > 0:
                                print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                      'learning rate to %s.' % (epoch + 1, new_lr), K.get_value(self.model.optimizer.lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class SGD_dlr(keras.optimizers.Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, split_1, lr=[0.01, 1e-4], momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGD_dlr, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.split_1 = [weight.name for split in split_1 for weight in split.weights]
        self.initial_decay = decay
        self.nesterov = nesterov

    @keras.optimizers.interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        lr_grp = lr[0]
        for p, g, m in zip(params, grads, moments):
            # Updating lr when the split layer is encountered
            if p.name in self.split_1:
#                print('lr1', p.name)
                lr_grp = lr[1]
            
            v = self.momentum * m - lr_grp * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr_grp * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': K.get_value(self.lr),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD_dlr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adam_dlr(keras.optimizers.Optimizer):

    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        split_1: split layer 1
        split_2: split layer 2
        lr: float >= 0. List of Learning rates. [Early layers, Middle layers, Final Layers]
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, split_1, lr=[1e-7, 1e-4], beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(Adam_dlr, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            # Extracting name of the split layers
            self.split_1 = [weight.name for split in split_1 for weight in split.weights]
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    @keras.optimizers.interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats
        
        # Setting lr of the initial layers
        lr_grp = lr_t[0]
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            
            # Updating lr when the split layer is encountered
            if p.name in self.split_1:
#                print('lr1', p.name)
                lr_grp = lr_t[1]
                
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_grp * m_t / (K.sqrt(vhat_t) + self.epsilon) # 使用更新后的学习率
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_grp * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
#         print('Optimizer LR: ', K.get_value(self.lr))
#         print()
        config = {
                  'lr': (K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(Adam_dlr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InteractAtt(keras.layers.Layer):
    def __init__(self,  attention_hidden=256, **kwargs):
        self.attention_hidden = attention_hidden
        super(InteractAtt, self).__init__(**kwargs)

    def build(self, input_shape):

        content_shape, pic_shape = input_shape[0],input_shape[1]
        self.x_content_it_W = self.add_weight(name="WCIT_{:s}".format(self.name),
                                 shape=(content_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_content_it_B = self.add_weight(name="BCIT_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)   
        self.x_content_et_W = self.add_weight(name="WCET_{:s}".format(self.name),
                                 shape=(content_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_content_et_B = self.add_weight(name="BCET_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)   
        self.x_content_ct_W = self.add_weight(name="WCCT_{:s}".format(self.name),
                                 shape=(content_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_content_ct_B = self.add_weight(name="BCCT_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)   
        self.x_content_ot_W = self.add_weight(name="WCOT_{:s}".format(self.name),
                                 shape=(content_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_content_ot_B = self.add_weight(name="BCOT_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)   
        
        
        self.x_pic_it_W = self.add_weight(name="WPIT_{:s}".format(self.name),
                                 shape=(pic_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_pic_it_B = self.add_weight(name="BPIT_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)
        self.x_pic_et_W = self.add_weight(name="WPET_{:s}".format(self.name),
                                 shape=(pic_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_pic_et_B = self.add_weight(name="BPET_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)
        self.x_pic_ct_W = self.add_weight(name="WPCT_{:s}".format(self.name),
                                 shape=(pic_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_pic_ct_B = self.add_weight(name="BPCT_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)
        self.x_pic_ot_W = self.add_weight(name="WPOT_{:s}".format(self.name),
                                 shape=(pic_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.x_pic_ot_B = self.add_weight(name="BPOT_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)
#        self.alpha = self.add_weight(name="ALPHA_{:s}".format(self.name),
#                                 shape=(4,),
#                                 initializer="glorot_normal",
#                                 trainable=True)
        super(InteractAtt, self).build(input_shape)

    def call(self, x, mask=None):

        x_content, x_pic = x[0], x[1]
    
        x_content_it = K.sigmoid(K.dot(x_content, self.x_content_it_W) + self.x_content_it_B)
        x_pic_it = K.sigmoid(K.dot(x_pic, self.x_pic_it_W) + self.x_pic_it_B)
        x_content_et = K.dot(x_content, self.x_content_et_W) + self.x_content_et_B
        x_pic_et = K.dot(x_pic, self.x_pic_et_W) + self.x_pic_et_B
        x_content_et_ = x_content_it * x_content_et
        x_pic_et_ = x_pic_it * x_pic_et
        x_content_att = K.softmax(x_content_et_)
        x_pic_att = K.softmax(x_pic_et_)
#        similary1 = K.sum(x_content_et * x_pic_et, axis=-1, keepdims=True) / (K.sqrt(K.sum(K.pow(x_content, 2), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.pow(x_pic, 2), axis=-1, keepdims=True)))
#        similary2 = K.sum(x_content_et_ * x_pic_et_, axis=-1, keepdims=True) / (K.sqrt(K.sum(K.pow(x_content_et_, 2), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.pow(x_pic_et_, 2), axis=-1, keepdims=True)))
        similary3 = K.sum(x_content_et * x_pic_et_, axis=-1, keepdims=True) / (K.sqrt(K.sum(K.pow(x_content_et, 2), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.pow(x_pic_et_, 2), axis=-1, keepdims=True)))
        similary4 = K.sum(x_pic_et * x_content_et_, axis=-1, keepdims=True) / (K.sqrt(K.sum(K.pow(x_pic_et, 2), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.pow(x_pic_et_, 2), axis=-1, keepdims=True)))
        
#        similary5 = K.softmax(x_content_et * x_pic_et_, axis=-1)
#        similary6 = K.softmax(x_pic_et * x_content_et_, axis=-1)
        
        similary_t = similary3 #* similary5 #- (similary1 - similary2)#similary2 #
        similary_p = similary4 #* similary6#- (similary1 - similary2)#similary2 #
        
        x_content_ct_ = K.tanh(K.dot(x_content, self.x_content_ct_W) + self.x_content_ct_B)
        x_pic_ct_ = K.tanh(K.dot(x_pic, self.x_pic_ct_W) + self.x_pic_ct_B)
        x_content_ct = x_content_ct_ * x_content_att + x_pic_ct_ * similary_t 
        x_pic_ct = x_pic_ct_ * x_pic_att + x_content_ct_ * similary_p 
        x_content_ot = K.sigmoid(K.dot(x_content, self.x_content_ot_W) + self.x_content_ot_B)
        x_pic_ot = K.sigmoid(K.dot(x_pic, self.x_pic_ot_W) + self.x_pic_ot_B)
        x_content_out = x_content_ot * K.tanh(x_content_ct)
        x_pic_out = x_pic_ot * K.tanh(x_pic_ct)
        
        
#        if mask is not None:
#            at *= K.cast(mask, K.floatx())

        return [x_content_out, x_pic_out]

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.attention_hidden), (input_shape[1][0], self.attention_hidden)]
  

class InteractAttention(keras.layers.Layer):
    def __init__(self,  attention_hidden=256, **kwargs):
        self.attention_hidden = attention_hidden
        super(InteractAttention, self).__init__(**kwargs)

    def build(self, input_shape):

#        print("attention intput :::_____________" , input_shape)
        content_shape, pic_shape = input_shape[0],input_shape[1]
        self.linnear_content_W = self.add_weight(name="LW1_{:s}".format(self.name),
                                 shape=(content_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.linnear_content_B = self.add_weight(name="LB1_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)        
        self.linnear_pic_W = self.add_weight(name="LW2_{:s}".format(self.name),
                                 shape=(pic_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.linnear_pic_B = self.add_weight(name="LB2_{:s}".format(self.name),
                                 shape=(self.attention_hidden,),
                                 initializer="zeros",
                                 trainable=True)
        self.attention_content_W = self.add_weight(name="AW1_{:s}".format(self.name),
                                 shape=(pic_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.attention_content_W_3 = self.add_weight(name="AW12_{:s}".format(self.name),
                                 shape=(content_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.attention_pic_W = self.add_weight(name="AW2_{:s}".format(self.name),
                                 shape=(content_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)

        self.attention_pic_W_3 = self.add_weight(name="AW23_{:s}".format(self.name),
                                 shape=(pic_shape[-1], self.attention_hidden),
                                 initializer="glorot_normal",
                                 trainable=True)

        super(InteractAttention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
#        
        x_content, x_pic = x[0], x[1]
    
        x_content_bak = K.dot(x_content, self.linnear_content_W) + self.linnear_content_B
        x_pic_bak = K.dot(x_pic, self.linnear_pic_W) + self.linnear_pic_B

        x_pic_weight = x_pic_bak * K.sigmoid(K.dot(x_pic, self.attention_content_W))
        x_content_weight = x_content_bak * K.sigmoid(K.dot(x_content, self.attention_content_W_3))
        x_content_weight_2 = K.tanh(K.mean(K.batch_dot(K.expand_dims(x_pic_weight, axis=-1),  K.expand_dims(x_content_weight, axis=-2)), axis=-1))
        
        x_content_weight_1 = x_content_bak * K.sigmoid(K.dot(x_content, self.attention_pic_W)) 
        x_pic_weight_1 = x_pic_bak * K.sigmoid(K.dot(x_pic, self.attention_pic_W_3))
        x_pic_weight_2 = K.tanh(K.mean(K.batch_dot(K.expand_dims(x_content_weight_1, axis=-1),  K.expand_dims(x_pic_weight_1, axis=-2)), axis=-1))
#           x_pic_weight = K.batch_dot(K.dot(x_content, self.attention_pic_W) * x_pic, axes=1)
   
        similary = K.sum(x_content_bak * x_pic_bak, axis=-1, keepdims=True) / (K.sqrt(K.sum(K.pow(x_content_bak, 2), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.pow(x_pic_bak, 2), axis=-1, keepdims=True)))
        print("similary intput :::_____________" , similary.get_shape().as_list())
#        x_content_out = x_content_bak * K.relu(x_content_weight)
#        x_content_out = x_content_bak * K.exp(K.relu(x_content_weight)) / (K.exp(K.relu(x_pic_weight)) + K.exp(K.relu(x_content_weight)))
        x_content_out = K.sigmoid(x_content_bak)  * x_content_weight_2
        
        x_pic_out = K.sigmoid(x_pic_bak)  * x_pic_weight_2 * similary
         
        
#        if mask is not None:
#            at *= K.cast(mask, K.floatx())

        return [x_content_out, x_pic_out]

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.attention_hidden), (input_shape[1][0], self.attention_hidden)]
   
    
class SelfAtt(keras.layers.Layer):
    def __init__(self, hiddensize=128,
                 return_probabilities=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):
        self.attention_size = hiddensize
        self.return_probabilities = return_probabilities
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        super(SelfAtt, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.dims_int = int(input_shape[1]) * int(input_shape[2])
        self.dims_float = float(self.dims_int)
        self.W1 = self.add_weight(name="W1_{:s}".format(self.name),
                                 shape=(int(input_shape[2]), self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.W2 = self.add_weight(name="W2_{:s}".format(self.name),
                                 shape=(self.attention_size, int(input_shape[1])),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(SelfAtt, self).build(input_shape)
        
    def call(self, x, mask=None):
        shape = x.get_shape().as_list()
        if len(shape) == 4:
            x = K.squeeze(x, axis=-1)
            
        OA = K.dot(x, self.W1)
        OA = K.tanh(OA)

        OA = K.dot(OA, self.W2)
        
        if mask is not None:
            mask = K.permute_dimensions(K.repeat(mask, OA.shape[-1]), [0,2,1]) #shape (none, repeat, x)
            OA *= K.cast(mask, K.floatx())
            
        OA = K.softmax(OA, axis=-2)    
        ot = K.batch_dot(K.permute_dimensions(x, [0,2,1]), OA)
        ot = K.permute_dimensions(ot, [0,2,1])
       
        if self.return_probabilities:
            return OA
        else:
            return ot

    def get_config(self):
        config = {
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':keras.regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(SelfAtt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_mask(self, input, input_mask=None):
        return input_mask

    def compute_output_shape(self, input_shape):
        return input_shape

    
class Scaled_Dot_Product_Att(keras.layers.Layer):
    def __init__(self, hiddensize=128, return_probabilities=False, **kwargs):
        self.attention_size = hiddensize
        self.return_probabilities = return_probabilities
        super(Scaled_Dot_Product_Att, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.dims_int = int(input_shape[1]) * int(input_shape[2])
        self.dims_float = float(self.attention_size)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(3, int(input_shape[2]), self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Scaled_Dot_Product_Att, self).build(input_shape)
        
    def call(self, x, mask=None):
        shape = x.get_shape().as_list()
        if len(shape) == 4:
            x = K.squeeze(x, axis=-1)
        
        
        WQ = K.dot(x, self.W[0])
        WK = K.dot(x, self.W[1])
        WV = K.dot(x, self.W[2])

        at = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1])) / tf.sqrt(self.dims_float)        
        at = K.softmax(at, axis=-1)
        ot = K.batch_dot(at, WV)


        return ot

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.attention_size)



    
    
       
