import tensorflow as tf
import copy

slim = tf.contrib.slim
arg_scope = tf.contrib.framework.arg_scope

DATA_FORMAT_NHWC = 'NHWC'
INVALID = 'null'
# The cap for tf.clip_by_value, it's hinted from the activation distribution
# that the majority of activation values are in the range [-6, 6].
CLIP_BY_VALUE_CAP = 6

'''
- calc_reduction_layers
- get_channel_index
- get_channel_dim
- global_avg_pool
- factorized_reduction
- drop_path

- NasNetABaseCell
- NasNetANormalCell
- NasNetAReductionCell
'''

def cifar_config():
    return tf.contrib.training.HParams(
            stem_multiplier=3.0,
            drop_path_keep_prob=0.6,
            num_cells=18,
            use_aux_head=1,
            num_conv_filters=32,
            dense_dropout_keep_prob=1.0,
            filter_scaling_rate=2.0,
            data_format='NHWC',
            skip_reduction_layer_input=0,
            total_training_stpes=937500,
            use_bounded_activation=False,)
    
    
def calc_reduction_layers(num_cells, num_reduction_layers):
  """Figure out what layers should have reductions."""
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers

# decorator --> function을 매개변수로
@tf.contrib.framework.add_arg_scope
def get_channel_index(data_format=INVALID):
  assert data_format != INVALID
  axis = 3 if data_format == 'NHWC' else 1
  return axis

@tf.contrib.framework.add_arg_scope
def get_channel_dim(shape, data_format=INVALID):
  assert data_format != INVALID
  assert len(shape) == 4
  if data_format == 'NHWC':
    return int(shape[3])
  elif data_format == 'NCHW':
    return int(shape[1])
  else:
    raise ValueError('Not a valid data_format', data_format)
    
@tf.contrib.framework.add_arg_scope
def global_avg_pool(x, data_format=INVALID):
  """Average pool away the height and width spatial dimensions of x."""
  assert data_format != INVALID
  assert data_format in ['NHWC', 'NCHW']
  assert x.shape.ndims == 4
  if data_format == 'NHWC':
    return tf.reduce_mean(x, [1, 2])
  else:
    return tf.reduce_mean(x, [2, 3])

@tf.contrib.framework.add_arg_scope
def factorized_reduction(net, output_filters, stride, data_format='null'):
    # channel reduction
    assert data_format != 'null'
    if stride == 1:
        net = slim.conv2d(net, output_filters, 1, scope='path_conv')
        net = slim.batch_norm(net, scope='path_bn')
        return net
    if data_format == 'NHWC':
        stride_spec = [1, stride, stride, 1]
    else:
        stride_spec = [1, 1, stride, stride]
    
    # skip path
    path1 = tf.nn.avg_pool(value=net, ksize=[1, 1, 1, 1], stride=stride_spec,
                           padding='valid', data_format=data_format)
    path1 = slim.conv2d(path1, int(output_filters/2), 1, scope='path1_conv')
    
    # skip path 2
    if data_format == 'NHWC':
        pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
        path2 = tf.pad(net, pad_arr)[:, 1:, 1:, :]
        concat_axis = 3
    else:
        pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
        path2 = tf.pad(net, pad_arr)[:, :, 1:, 1:]
        concat_axis = 1
    path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], stride_spec, 'VALID', data_format=data_format)
    
    final_filter_size = int(output_filters / 2) + int(output_filters % 2)
    path2 = slim.conv2d(path2, final_filter_size, 1, scope='path2_conv')
    final_path = tf.concat(values=[path1, path2], axis=concat_axis)
    final_path = slim.batch_norm(final_path, scope='final_path_bn')
    return final_path

@tf.contrib.framework.add_arg_scope
def drop_path(net, keep_prob, is_training=True):
  """Drops out a whole example hiddenstate with the specified probability."""
  if is_training:
    batch_size = tf.shape(net)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.cast(tf.floor(random_tensor), net.dtype)
    keep_prob_inv = tf.cast(1.0 / keep_prob, net.dtype)
    net = net * keep_prob_inv * binary_tensor

  return net    

def _operation_to_filter_shape(operation):
  splitted_operation = operation.split('x')
  filter_shape = int(splitted_operation[0][-1])
  assert filter_shape == int(
      splitted_operation[1][0]), 'Rectangular filters not supported.'
  return filter_shape


def _operation_to_num_layers(operation):
  splitted_operation = operation.split('_')
  if 'x' in splitted_operation[-1]:
    return 1
  return int(splitted_operation[-1])


def _operation_to_info(operation):
  """Takes in operation name and returns meta information.
  An example would be 'separable_3x3_4' -> (3, 4).
  Args:
    operation: String that corresponds to convolution operation.
  Returns:
    Tuple of (filter shape, num layers).
  """
  num_layers = _operation_to_num_layers(operation)
  filter_shape = _operation_to_filter_shape(operation)
  return num_layers, filter_shape


def _stacked_separable_conv(net, stride, operation, filter_size,
                            use_bounded_activation):
  """Takes in an operations and parses it to the correct sep operation."""
  num_layers, kernel_size = _operation_to_info(operation)
  activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
  for layer_num in range(num_layers - 1):
    net = activation_fn(net)
    net = slim.separable_conv2d(
        net,
        filter_size,
        kernel_size,
        depth_multiplier=1,
        scope='separable_{0}x{0}_{1}'.format(kernel_size, layer_num + 1),
        stride=stride)
    net = slim.batch_norm(
        net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, layer_num + 1))
    stride = 1
  net = activation_fn(net)
  net = slim.separable_conv2d(
      net,
      filter_size,
      kernel_size,
      depth_multiplier=1,
      scope='separable_{0}x{0}_{1}'.format(kernel_size, num_layers),
      stride=stride)
  net = slim.batch_norm(
      net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, num_layers))
  return net


def _operation_to_pooling_type(operation):
  """Takes in the operation string and returns the pooling type."""
  splitted_operation = operation.split('_')
  return splitted_operation[0]


def _operation_to_pooling_shape(operation):
  """Takes in the operation string and returns the pooling kernel shape."""
  splitted_operation = operation.split('_')
  shape = splitted_operation[-1]
  assert 'x' in shape
  filter_height, filter_width = shape.split('x')
  assert filter_height == filter_width
  return int(filter_height)


def _operation_to_pooling_info(operation):
  """Parses the pooling operation string to return its type and shape."""
  pooling_type = _operation_to_pooling_type(operation)
  pooling_shape = _operation_to_pooling_shape(operation)
  return pooling_type, pooling_shape


def _pooling(net, stride, operation, use_bounded_activation):
  """Parses operation and performs the correct pooling operation on net."""
  padding = 'SAME'
  pooling_type, pooling_shape = _operation_to_pooling_info(operation)
  if use_bounded_activation:
    net = tf.nn.relu6(net)
  if pooling_type == 'avg':
    net = slim.avg_pool2d(net, pooling_shape, stride=stride, padding=padding)
  elif pooling_type == 'max':
    net = slim.max_pool2d(net, pooling_shape, stride=stride, padding=padding)
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)
  return net

class NasNetABaseCell(object):
    '''
    function: NASNet cell class
    
    inputs:
        num_conv_filters: # of filters
        used_hiddenstates: binary array
        hiddenstate_indices: determines what hiddenstates
        use_bounded_activation
        
    '''
    def __init__(self, num_conv_filters, operations, used_hiddenstates, hiddenstate_indices,
                 drop_path_keep_prob, total_num_cells, total_training_steps, use_bounded_activation=False):
        self._num_conv_filters = num_conv_filters
        self._operations = operations
        self._used_hiddenstates = used_hiddenstates
        self._hiddenstate_indices = hiddenstate_indices
        self._drop_path_keep_prob = drop_path_keep_prob
        self._total_num_cells = total_num_cells
        self._total_training_steps = total_training_steps
        self._use_bounded_activation = use_bounded_activation
        
    def _reduce_prev_layer(self, prev_layer, curr_layer):
        if prev_layer == None:
            return curr_layer
        
        curr_num_filters = self._filter_size
        prev_num_filters = get_channel_dim(prev_layer.shape)
        curr_filter_shape = int(curr_layer.shape[2])
        prev_filter_shape = int(prev_layer.shape[2])
        activation_fn = tf.nn.relu6 if self._use_bounded_activation else tf.nn.relu
        if curr_filter_shape != prev_filter_shape:
            prev_layer = activation_fn(prev_layer)
            prev_layer = factorized_reduction(prev_layer, curr_num_filters, stride=2)
        elif curr_num_filters != prev_num_filters:
            prev_layer = activation_fn(prev_layer)
            prev_layer = slim.conv2d(prev_layer, curr_num_filters, 1, scope='prev_1x1')
            prev_layer = slim.batch_norm(prev_layer, scope='prev_bn')
        return prev_layer
    
    def _cell_base(self, net, prev_layer):
        num_filters = self._filter_size
        prev_layer = self._reduce_prev_layer(prev_layer, net)
        net = tf.nn.relu6(net) if self._use_bounded_activation else tf.nn.relu(net)
        net = slim.conv2d(net, num_filters, 1, scope='1x1')
        net = slim.batch_norm(net, scope='beginning_bn')
        net = [net]
        net.append(prev_layer)
        return net
    
    def __call__(self, net, scope=None, filter_scaling=1, stride=1, prev_layer=None, cell_num=-1, current_step=None):
        self._cell_num = cell_num
        self._filter_scaling = filter_scaling
        self._filter_size = int(self._num_conv_filters * filter_scaling)
        
        i = 0
        with tf.variable_scope(scope):
          net = self._cell_base(net, prev_layer)
          for iteration in range(5):
            with tf.variable_scope('comb_iter_{}'.format(iteration)):
              left_hiddenstate_idx, right_hiddenstate_idx = (
                  self._hiddenstate_indices[i],
                  self._hiddenstate_indices[i + 1])
              original_input_left = left_hiddenstate_idx < 2
              original_input_right = right_hiddenstate_idx < 2
              h1 = net[left_hiddenstate_idx]
              h2 = net[right_hiddenstate_idx]
    
              operation_left = self._operations[i]
              operation_right = self._operations[i+1]
              i += 2
              # Apply conv operations
              with tf.variable_scope('left'):
                h1 = self._apply_conv_operation(h1, operation_left,
                                                stride, original_input_left,
                                                current_step)
              with tf.variable_scope('right'):
                h2 = self._apply_conv_operation(h2, operation_right,
                                                stride, original_input_right,
                                                current_step)
    
              # Combine hidden states using 'add'.
              with tf.variable_scope('combine'):
                h = h1 + h2
                if self._use_bounded_activation:
                  h = tf.nn.relu6(h)
    
              # Add hiddenstate to the list of hiddenstates we can choose from
              net.append(h)

          with tf.variable_scope('cell_output'):
            net = self._combine_unused_states(net)
    
          return net

    def _apply_conv_operation(self, net, operation, stride, is_from_original_input, current_step):
        """Applies the predicted conv operation to net."""
        # Dont stride if this is not one of the original hiddenstates
        if stride > 1 and not is_from_original_input:
          stride = 1
        input_filters = get_channel_dim(net.shape)
        filter_size = self._filter_size
        if 'separable' in operation:
          net = _stacked_separable_conv(net, stride, operation, filter_size,
                                        self._use_bounded_activation)
          if self._use_bounded_activation:
            net = tf.clip_by_value(net, -CLIP_BY_VALUE_CAP, CLIP_BY_VALUE_CAP)
        elif operation in ['none']:
          if self._use_bounded_activation:
            net = tf.nn.relu6(net)
          # Check if a stride is needed, then use a strided 1x1 here
          if stride > 1 or (input_filters != filter_size):
            if not self._use_bounded_activation:
              net = tf.nn.relu(net)
            net = slim.conv2d(net, filter_size, 1, stride=stride, scope='1x1')
            net = slim.batch_norm(net, scope='bn_1')
            if self._use_bounded_activation:
              net = tf.clip_by_value(net, -CLIP_BY_VALUE_CAP, CLIP_BY_VALUE_CAP)
        elif 'pool' in operation:
          net = _pooling(net, stride, operation, self._use_bounded_activation)
          if input_filters != filter_size:
            net = slim.conv2d(net, filter_size, 1, stride=1, scope='1x1')
            net = slim.batch_norm(net, scope='bn_1')
          if self._use_bounded_activation:
            net = tf.clip_by_value(net, -CLIP_BY_VALUE_CAP, CLIP_BY_VALUE_CAP)
        else:
          raise ValueError('Unimplemented operation', operation)
    
        if operation != 'none':
          net = self._apply_drop_path(net, current_step=current_step)
        return net

    def _combine_unused_states(self, net):
        """Concatenate the unused hidden states of the cell."""
        used_hiddenstates = self._used_hiddenstates
    
        final_height = int(net[-1].shape[2])
        final_num_filters = get_channel_dim(net[-1].shape)
        assert len(used_hiddenstates) == len(net)
        for idx, used_h in enumerate(used_hiddenstates):
          curr_height = int(net[idx].shape[2])
          curr_num_filters = get_channel_dim(net[idx].shape)
    
          # Determine if a reduction should be applied to make the number of
          # filters match.
          should_reduce = final_num_filters != curr_num_filters
          should_reduce = (final_height != curr_height) or should_reduce
          should_reduce = should_reduce and not used_h
          if should_reduce:
            stride = 2 if final_height != curr_height else 1
            with tf.variable_scope('reduction_{}'.format(idx)):
              net[idx] = factorized_reduction(
                  net[idx], final_num_filters, stride)
    
        states_to_combine = (
            [h for h, is_used in zip(net, used_hiddenstates) if not is_used])
    
        # Return the concat of all the states
        concat_axis = get_channel_index()
        net = tf.concat(values=states_to_combine, axis=concat_axis)
        return net

    @tf.contrib.framework.add_arg_scope  # No public API. For internal use only.
    def _apply_drop_path(self, net, current_step=None, use_summaries=False, drop_connect_version='v3'):
        """Apply drop_path regularization.
        Args:
          net: the Tensor that gets drop_path regularization applied.
          current_step: a float32 Tensor with the current global_step value,
            to be divided by hparams.total_training_steps. Usually None, which
            defaults to tf.train.get_or_create_global_step() properly casted.
          use_summaries: a Python boolean. If set to False, no summaries are output.
          drop_connect_version: one of 'v1', 'v2', 'v3', controlling whether
            the dropout rate is scaled by current_step (v1), layer (v2), or
            both (v3, the default).
        Returns:
          The dropped-out value of `net`.
        """
        drop_path_keep_prob = self._drop_path_keep_prob
        if drop_path_keep_prob < 1.0:
          assert drop_connect_version in ['v1', 'v2', 'v3']
          if drop_connect_version in ['v2', 'v3']:
            # Scale keep prob by layer number
            assert self._cell_num != -1
            # The added 2 is for the reduction cells
            num_cells = self._total_num_cells
            layer_ratio = (self._cell_num + 1)/float(num_cells)
            if use_summaries:
              with tf.device('/cpu:0'):
                tf.summary.scalar('layer_ratio', layer_ratio)
            drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
          if drop_connect_version in ['v1', 'v3']:
            # Decrease the keep probability over time
            if current_step is None:
              current_step = tf.train.get_or_create_global_step()
            current_step = tf.cast(current_step, tf.float32)
            drop_path_burn_in_steps = self._total_training_steps
            current_ratio = current_step / drop_path_burn_in_steps
            current_ratio = tf.minimum(1.0, current_ratio)
            if use_summaries:
              with tf.device('/cpu:0'):
                tf.summary.scalar('current_ratio', current_ratio)
            drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
          if use_summaries:
            with tf.device('/cpu:0'):
              tf.summary.scalar('drop_path_keep_prob', drop_path_keep_prob)
          net = drop_path(net, drop_path_keep_prob)
        return net


class NasNetANormalCell(NasNetABaseCell):
  """NASNetA Normal Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps, use_bounded_activation=False):
    operations = ['separable_5x5_2',
                  'separable_3x3_2',
                  'separable_5x5_2',
                  'separable_3x3_2',
                  'avg_pool_3x3',
                  'none',
                  'avg_pool_3x3',
                  'avg_pool_3x3',
                  'separable_3x3_2',
                  'none']
    used_hiddenstates = [1, 0, 0, 0, 0, 0, 0]
    hiddenstate_indices = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    super(NasNetANormalCell, self).__init__(num_conv_filters, operations,
                                            used_hiddenstates,
                                            hiddenstate_indices,
                                            drop_path_keep_prob,
                                            total_num_cells,
                                            total_training_steps,
                                            use_bounded_activation)


class NasNetAReductionCell(NasNetABaseCell):
  """NASNetA Reduction Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps, use_bounded_activation=False):
    operations = ['separable_5x5_2',
                  'separable_7x7_2',
                  'max_pool_3x3',
                  'separable_7x7_2',
                  'avg_pool_3x3',
                  'separable_5x5_2',
                  'none',
                  'avg_pool_3x3',
                  'separable_3x3_2',
                  'max_pool_3x3']
    used_hiddenstates = [1, 1, 1, 0, 0, 0, 0]
    hiddenstate_indices = [0, 1, 0, 1, 0, 1, 3, 2, 2, 0]
    super(NasNetAReductionCell, self).__init__(num_conv_filters, operations,
                                               used_hiddenstates,
                                               hiddenstate_indices,
                                               drop_path_keep_prob,
                                               total_num_cells,
                                               total_training_steps,
                                               use_bounded_activation)

def nasnet_cifar_arg_scope(weight_decay=5e-4,
                           batch_norm_decay=0.9,
                           batch_norm_epsilon=1e-5):
  """
  function: Defines the default arg scope for the NASNet-A Cifar model.
  inputs:
    - weight_decay: The weight decay to use for regularizing the model.
    - batch_norm_decay: Decay for batch norm moving average.
    - batch_norm_epsilon: Small float added to variance to avoid dividing by zero in batch norm.
  Returns: An `arg_scope` to use for the NASNet Cifar Model.
  """
  batch_norm_params = {'decay': batch_norm_decay,
                       'epsilon': batch_norm_epsilon,
                       'scale': True,
                       'fused': True,}
  weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      mode='FAN_OUT')
  with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d],
                 weights_regularizer=weights_regularizer,
                 weights_initializer=weights_initializer):
    with arg_scope([slim.fully_connected],
                   activation_fn=None, scope='FC'):
      with arg_scope([slim.conv2d, slim.separable_conv2d],
                     activation_fn=None, biases_initializer=None):
        with arg_scope([slim.batch_norm], **batch_norm_params) as sc:
          return sc

def _build_aux_head(net, end_points, num_classes, hparams, scope):
    '''
    function: auxiliary head used for all models across all dataset
    inputs:
        - net
        - end_points
        - num_classes
        - hparams
        - scope
    '''
    activation_fn = tf.nn.relu6 if hparams.use_bounded_activation else tf.nn.relu
    with tf.variable_scope(scope):
        aux_logits = tf.identity(net)
        with tf.variable_scope('aux_ligits'):
            aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID')
            aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
            aux_logits = slim.batch_norm(aux_logits, scope='aux_bn0')
            aux_logits = activation_fn(aux_logits)
            shape = aux_logits.shape
            if hparams.data_format == 'NHWC':
                shape = shape[1:3]
            else:
                shape = shape[2:4]
            aux_logits = slim.conv2d(aux_logits, 768, shape, padding='VALID')
            aux_logits = slim.batch_norm(aux_logits, scope='aux_bn1')
            aux_logits = activation_fn(aux_logits)
            aux_logits = tf.contrib.layers.flatten(aux_logits)
            aux_logits = slim.fully_connected(aux_logits, num_classes)
            end_points['AuxLogits'] = aux_logits
            
def _update_hparams(hparams, is_training):
  """Update hparams for given is_training option."""
  if not is_training:
    hparams.set_hparam('drop_path_keep_prob', 1.0)


def _cifar_stem(inputs, hparams):
  """Stem used for models trained on Cifar."""
  num_stem_filters = int(hparams.num_conv_filters * hparams.stem_multiplier)
  net = slim.conv2d(inputs, num_stem_filters, 3, scope='l1_stem_3x3')
  net = slim.batch_norm(net, scope='l1_stem_bn')
  return net, [None, net]

def build_nasnet_cifar(images, num_classes, is_training=True, config=None, current_step=None):
    '''
    function: build cifar NASNET A
    inputs:
        - images: image
        - num_classes: 10
        - is_training
        - config
        - current_step
    '''
    hparams = cifar_config() if config is None else copy.deepcopy(config)
    _update_hparams(hparams, is_training)
    if tf.test.is_gpu_availabel() and hparams.data_foramt == 'NHWC':
        tf.logging.info('GPU is available')
    if hparams.data_forat == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
    
    total_num_cells = hparams.num_cells + 2
    normal_cell = NasNetANormalCell(num_conv_filters=hparams.num_conv_filters,
                                    drop_path_keep_prob=hparams.drop_path_keep_prob,
                                    total_num_cells=total_num_cells,
                                    total_training_steps=hparams.total_training_steps,
                                    use_bounded_activation=hparams.use_bounded_activation)
    reduction_cell = NasNetAReductionCell(num_conv_filters=hparams.num_conv_filters,
                                          drop_path_keep_prob=hparams.drop_path_keep_prob,
                                          total_num_cells=total_num_cells,
                                          total_training_steps=hparams.total_training_steps,
                                          use_bounded_activation=hparams.use_bounded_activation)
    with arg_scope([slim.dropout, drop_path, slim.batch_norm], is_training=is_training):
        with arg_scope([slim.avg_pool2d, slim.max_pool2d, slim.conv2d, slim.batch_norm,
                        slim.separable_conv2d, factorized_reduction, global_avg_pool,
                        get_channel_index, get_channel_dim], data_format=hparams.data_format):
            return _build_nasnet_base(images, normal_cell=normal_cell, reduction_cell=reduction_cell,
                                      num_classes=num_classes, hparams=hparams, is_training=is_training,
                                      stem_type='cifar', current_step=current_step)
    
def _build_nasnet_base(images,
                       normal_cell,
                       reduction_cell,
                       num_classes,
                       hparams,
                       is_training,
                       stem_type,
                       final_endpoint=None,
                       current_step=None):
  """Constructs a NASNet image model."""

  end_points = {}
  def add_and_check_endpoint(endpoint_name, net):
    end_points[endpoint_name] = net
    return final_endpoint and (endpoint_name == final_endpoint)

  # Find where to place the reduction cells or stride normal cells
  reduction_indices = calc_reduction_layers(
      hparams.num_cells, hparams.num_reduction_layers)
  stem_cell = reduction_cell
  if stem_type == 'cifar':
    stem = lambda: _cifar_stem(images, hparams)
  else:
    raise ValueError('Unknown stem_type: ', stem_type)
  net, cell_outputs = stem()
  if add_and_check_endpoint('Stem', net): return net, end_points

  # Setup for building in the auxiliary head.
  aux_head_cell_idxes = []
  if len(reduction_indices) >= 2:
    aux_head_cell_idxes.append(reduction_indices[1] - 1)

  # Run the cells
  filter_scaling = 1.0
  # true_cell_num accounts for the stem cells
  true_cell_num = 2 if stem_type == 'imagenet' else 0
  activation_fn = tf.nn.relu6 if hparams.use_bounded_activation else tf.nn.relu
  for cell_num in range(hparams.num_cells):
    stride = 1
    if hparams.skip_reduction_layer_input:
      prev_layer = cell_outputs[-2]
    if cell_num in reduction_indices:
      filter_scaling *= hparams.filter_scaling_rate
      net = reduction_cell(
          net,
          scope='reduction_cell_{}'.format(reduction_indices.index(cell_num)),
          filter_scaling=filter_scaling,
          stride=2,
          prev_layer=cell_outputs[-2],
          cell_num=true_cell_num,
          current_step=current_step)
      if add_and_check_endpoint(
          'Reduction_Cell_{}'.format(reduction_indices.index(cell_num)), net):
        return net, end_points
      true_cell_num += 1
      cell_outputs.append(net)
    if not hparams.skip_reduction_layer_input:
      prev_layer = cell_outputs[-2]
    net = normal_cell(
        net,
        scope='cell_{}'.format(cell_num),
        filter_scaling=filter_scaling,
        stride=stride,
        prev_layer=prev_layer,
        cell_num=true_cell_num,
        current_step=current_step)

    if add_and_check_endpoint('Cell_{}'.format(cell_num), net):
      return net, end_points
    true_cell_num += 1
    if (hparams.use_aux_head and cell_num in aux_head_cell_idxes and
        num_classes and is_training):
      aux_net = activation_fn(net)
      _build_aux_head(aux_net, end_points, num_classes, hparams,
                      scope='aux_{}'.format(cell_num))
    cell_outputs.append(net)

  # Final softmax layer
  with tf.variable_scope('final_layer'):
    net = activation_fn(net)
    net = global_avg_pool(net)
    if add_and_check_endpoint('global_pool', net) or not num_classes:
      return net, end_points
    net = slim.dropout(net, hparams.dense_dropout_keep_prob, scope='dropout')
    logits = slim.fully_connected(net, num_classes)

    if add_and_check_endpoint('Logits', logits):
      return net, end_points

    predictions = tf.nn.softmax(logits, name='predictions')
    if add_and_check_endpoint('Predictions', predictions):
      return net, end_points
  return logits, end_points    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






