from keras import layers, models, initializers, Model, metrics

def batch_norm():
    return layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

def relu():
    return layers.ReLU()

def conv1d(filters, kernel_size=3, strides=1):
    return layers.Conv1D(
        filters, kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_initializer=initializers.VarianceScaling())

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        num_chan = input_shape[-1]
        self.conv1 = conv1d(self.filters, self.kernel_size, self.strides)
        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.conv2 = conv1d(self.filters, self.kernel_size, 1)
        self.bn2 = batch_norm()
        self.relu2 = relu()
        if num_chan != self.filters or self.strides > 1:
            self.proj_conv = conv1d(self.filters, 1, self.strides)
            self.proj_bn = batch_norm()
            self.projection = True
        else:
            self.projection = False
        super().build(input_shape)

    def call(self, x, **kwargs):
        shortcut = x
        if self.projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_bn(shortcut)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x + shortcut)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class BottleneckBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, expansion=4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expansion = expansion

    def build(self, input_shape):
        num_chan = input_shape[-1]
        self.conv1 = conv1d(self.filters, 1, 1)
        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.conv2 = conv1d(self.filters, self.kernel_size, self.strides)
        self.bn2 = batch_norm()
        self.relu2 = relu()
        self.conv3 = conv1d(self.filters * self.expansion, 1, 1)
        self.bn3 = batch_norm()
        self.relu3 = relu()
        if num_chan != self.filters * self.expansion or self.strides > 1:
            self.proj_conv = conv1d(self.filters * self.expansion, 1, self.strides)
            self.proj_bn = batch_norm()
            self.projection = True
        else:
            self.projection = False
        super().build(input_shape)

    def call(self, x, **kwargs):
        shortcut = x
        if self.projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_bn(shortcut)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x + shortcut)
        return x


class ResNet(Model):
    def __init__(self, num_outputs=1, blocks=(2, 2, 2, 2),
                 filters=(64, 128, 256, 512), kernel_size=(3, 3, 3, 3),
                 block_fn=ResidualBlock, include_top=True, **kwargs):
        super().__init__(**kwargs)
        layers.InputLayer(input_shape=(4,))
        self.conv1 = conv1d(64, 7, 2)
        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.maxpool1 = layers.MaxPooling1D(3, 2, padding='same')
        self.blocks = []
        for stage, num_blocks in enumerate(blocks):
            for block in range(num_blocks):
                strides = 2 if block == 0 and stage > 0 else 1
                res_block = block_fn(filters[stage], kernel_size[stage], strides)
                self.blocks.append(res_block)
        self.include_top = include_top
        if include_top:
            self.global_pool = layers.GlobalAveragePooling1D()
            out_act = 'sigmoid' if num_outputs == 1 else 'softmax'
            self.classifier = layers.Dense(num_outputs, out_act)
            #self.classifier = layers.Dense(5, activation="softmax")

    def call(self, x, include_top=None, **kwargs):
        if include_top is None:
            include_top = self.include_top
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        for res_block in self.blocks:
            x = res_block(x)
        if include_top:
            x = self.global_pool(x)
            x = self.classifier(x)
        return x

def build_model(input_shape=(2500, 8)):
    model = ResNet(num_outputs = 5)
    model.build((None, 2500, 8))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')],
      )
    return model
    