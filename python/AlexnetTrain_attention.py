import tensorflow as tf
import os
from tensorflow.keras import layers, models, losses
import datetime
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]  # RGB values
    jet_heatmap = jet_colors[heatmap]

    # Resize heatmap to match image size
    jet_heatmap = tf.image.resize(jet_heatmap[tf.newaxis, ...], (img.shape[0], img.shape[1]))[0]

    # Normalize original image
    img = tf.cast(img, tf.float32) / 255.0

    # Combine heatmap and original image: original stays visible with transparent heatmap on top
    superimposed_img = img * (1 - alpha) + jet_heatmap * alpha

    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

def fixed_gaussian_blur(img, kernel_size=5, sigma=1.0):
    def _gaussian_kernel(size, sigma):
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        x = tf.exp(-0.5 * (x / sigma) ** 2)
        kernel = x[:, None] * x[None, :]
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel[:, :, tf.newaxis, tf.newaxis]

    kernel = _gaussian_kernel(kernel_size, sigma)
    kernel = tf.tile(kernel, [1, 1, img.shape[-1], 1])  # [5,5,channels,1]

    # no need to expand dims here; img already has shape [batch, height, width, channels]
    img = tf.nn.depthwise_conv2d(img, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return img

def log_gradcam_to_tensorboard(model, image, step, log_dir, last_conv_layer_name="last_conv"):
    img_batch = tf.expand_dims(image, axis=0)
    heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)
    overlay_img = overlay_heatmap(image.numpy(), heatmap)

    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        tf.summary.image("GradCAM", np.expand_dims(tf.keras.preprocessing.image.img_to_array(overlay_img) / 255.0, 0), step=step)

def preprocess_fn(x, y):
    # x = tf.cast(x, tf.float32) / 255.0  # Normalize first if needed
    x = fixed_gaussian_blur(x, kernel_size=5, sigma=0.5)
    # x = tf.clip_by_value(x * 255.0, 0, 255.0)
    x = tf.clip_by_value(x, 0, 1.0)
    return x, y

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print(e)

### --- Attention Block --- ###
def se_block(input_tensor, reduction=4, name=None):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu', name=name+'_se_dense1')(se)
    se = layers.Dense(filters, activation='sigmoid', name=name+'_se_dense2')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Lambda(lambda tensors: tensors[0] * tensors[1])([input_tensor, se])#layers.Multiply(name=name+'_se_multiply')([input_tensor, se])

### --- Core Building Blocks --- ###
def expansion_block(x, t, filters, block_id):
    prefix = f'block_{block_id}_'
    total_filters = t * filters
    x = layers.Conv2D(total_filters, 1, padding='same', use_bias=False, name=prefix+'expand')(x)
    x = layers.BatchNormalization(name=prefix+'expand_bn')(x)
    x = layers.ReLU(6., name=prefix+'expand_relu')(x)
    return x

def depthwise_block(x, stride, block_id):
    prefix = f'block_{block_id}_'
    x = layers.DepthwiseConv2D(3, strides=(stride, stride), padding='same', use_bias=False, name=prefix+'depthwise')(x)
    x = layers.BatchNormalization(name=prefix+'dw_bn')(x)
    x = layers.ReLU(6., name=prefix+'dw_relu')(x)
    return x

def projection_block(x, out_channels, block_id):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(out_channels, 1, padding='same', use_bias=False, name=prefix+'project')(x)
    x = layers.BatchNormalization(name=prefix+'project_bn')(x)
    return x

def Bottleneck(x, t, filters, out_channels, stride, block_id, use_se=True):
    shortcut = x
    x = expansion_block(x, t, filters, block_id)
    x = depthwise_block(x, stride, block_id)
    x = projection_block(x, out_channels, block_id)
    
    # Attention block
    if use_se:
        x = se_block(x, reduction=4, name=f'block_{block_id}_se')

    if stride == 1 and x.shape[-1] == shortcut.shape[-1]:
        x = layers.add([shortcut, x], name=f'block_{block_id}_add')
    return x

### --- Modified MobileNetV2 --- ###
def MobileNetV2(input_shape=(224, 224, 3), n_classes=1000):
    input = tf.keras.Input(shape=input_shape)
    # x = layers.Rescaling(1/255.0)(input)

    x = layers.Conv2D(32, 3, strides=(2,2), padding='same', use_bias=False, name='initial_conv')(input)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.ReLU(6., name='initial_relu')(x)

    # First block manually
    x = depthwise_block(x, stride=1, block_id=1)
    x = projection_block(x, out_channels=16, block_id=1)

    # Bottleneck blocks
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=24, stride=2, block_id=2)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=24, stride=1, block_id=3)

    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=32, stride=2, block_id=4)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=5)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=6)

    # --- Insert a dilated convolution after block 6 ---
    x = layers.Conv2D(x.shape[-1], 3, padding='same', dilation_rate=2, name='dilated_conv_1')(x)

    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=2, block_id=7)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=8)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=9)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=10)

    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=11)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=12)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=13)

    # --- Insert another dilated convolution after block 13 ---
    x = layers.Conv2D(x.shape[-1], 3, padding='same', dilation_rate=2, name='dilated_conv_2')(x)

    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=160, stride=2, block_id=14)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=15)
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=16)

    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=320, stride=1, block_id=17)

    # Final layers
    x = layers.Conv2D(1280, 1, padding='same', use_bias=False, name='final_conv')(x)
    x = layers.BatchNormalization(name='final_bn')(x)
    x = layers.ReLU(6., name='final_relu')(x)

    x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    output = layers.Dense(n_classes, activation='softmax', name='classifier')(x)

    model = models.Model(inputs=input, outputs=output)
    return model


class GradCAMCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample_batch, log_dir, conv_layer="last_conv"):
        super().__init__()
        self.sample_image = sample_batch[0][0]  # first image in batch
        self.log_dir = log_dir
        self.layer_name = conv_layer

    def on_epoch_end(self, epoch, logs=None):
        log_gradcam_to_tensorboard(self.model, self.sample_image, epoch, self.log_dir, self.layer_name)


## Get the dataset
# image_ds = tf.keras.utils.image_dataset_from_directory("Dataset3", labels='inferred', label_mode="categorical", batch_size=32, image_size=(224,224))

# print(image_ds)

# (train_ds, valid_ds) = image_ds
# (valid_ds, test_ds) = tf.keras.utils.split_dataset(valid_ds, 0.5, 0.5, True, 0)

train_ds = tf.keras.preprocessing.image_dataset_from_directory("Dataset1_train", labels='inferred', label_mode="categorical", batch_size=32, image_size=(224,224))
valid_ds = tf.keras.preprocessing.image_dataset_from_directory("Dataset1_valid", labels='inferred', label_mode="categorical", batch_size=32, image_size=(224,224))
test_ds = tf.keras.preprocessing.image_dataset_from_directory("Dataset1_test", labels='inferred', label_mode="categorical", batch_size=32, image_size=(224,224))

print(train_ds)
print(valid_ds)
print(test_ds)

## Create the model
inputs = tf.keras.Input(shape=(224,224,3))
scaled_layer = layers.Rescaling(scale=1/255.0)

x = scaled_layer(inputs)

# model = models.Sequential()

# model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
# model.add(layers.Lambda(tf.nn.local_response_normalization))
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D(3, strides=2))

# model.add(layers.Flatten())
# model.add(layers.Dense(4096, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(4096, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(2, activation='softmax'))

# outputs = model(x)

# custom_model = tf.keras.Model(inputs, outputs)
# input_sizes = [224, 128, 64, 32, 16]
input_sizes = [224]
# input_sizes = [16]
for input_size in input_sizes:
    tf.keras.backend.clear_session()
    custom_model = MobileNetV2((input_size, input_size, 3), 2)

    custom_model.summary()

    ## Train
    custom_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss = losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(curve="PR"), 
                tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.BinaryIoU(),  
                tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
                ]
    )
    resize_layer = tf.keras.layers.Resizing(input_size, input_size)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])
    sequential = tf.keras.Sequential([
        data_augmentation,
        resize_layer,
        scaled_layer
    ])
    aug_train_ds = train_ds.map(lambda x, y: (sequential(x, training=True), y))
    aug_train_ds = aug_train_ds.map(preprocess_fn)
    aug_valid_ds = valid_ds.map(lambda x, y: (sequential(x, training=False), y))
    logdir = os.path.join("Logs3", "custom_model2_grad_attention_"+str(input_size))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, update_freq="batch")

    # Figure out a way to conserve memory because it might crash the system with how much it's using
    epochs = 20
    # custom_model.fit(aug_train_ds, validation_data=aug_valid_ds, epochs=epochs, callbacks=[tensorboard_callback])
    sample_batch = next(iter(aug_valid_ds))
    gradcam_callback = GradCAMCallback(sample_batch, logdir, conv_layer="final_conv")

    custom_model.fit(aug_train_ds, validation_data=aug_valid_ds, epochs=epochs, callbacks=[
        tensorboard_callback,
        gradcam_callback
    ])

    custom_model.evaluate(test_ds)

    custom_model.save("Model/new_224_grad_attention")

    # ## Save the model to use in C++
    # full_model = tf.function(lambda x: custom_model(x,training=False))
    # full_model = full_model.get_concrete_function(
    #     tf.TensorSpec(custom_model.inputs[0].shape, custom_model.inputs[0].dtype))

    # # Get frozen ConcreteFunction
    # frozen_func = convert_variables_to_constants_v2(full_model)
    # frozen_func.graph.as_graph_def()

    # layers_names = [op.name for op in frozen_func.graph.get_operations()]
    # print("-" * 50)
    # print("Frozen model layers: ")
    # for layer_name in layers_names:
    #     print(layer_name)

    # print("-" * 50)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)

    # # Save frozen graph from frozen ConcreteFunction to hard drive
    # # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
    # #                 logdir="./Model/frozen_models",
    # #                 name="frozen_graph_"+str(input_size)+".pb",
    # #                 as_text=False)