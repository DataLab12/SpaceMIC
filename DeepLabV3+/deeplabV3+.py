import os
# import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import ZeroPadding2D
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from scipy.ndimage import gaussian_filter, map_coordinates

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.8/toolkit/11.8.0"
os.environ["KERAS_BACKEND"] = "tensorflow"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#Define the paths to the dataset folders
train_image_folder ='../MIC_NASA_dataset/train/images'
train_mask_folder = '../MIC_NASA_dataset/train/masks'


test_image_folder = '../MIC_NASA_dataset/test/images'
test_mask_folder = '../MIC_NASA_dataset/test/masks'

#Use os.listdir() to get the list of file names in each folder
train_image_files = os.listdir(train_image_folder)
train_mask_files = os.listdir(train_mask_folder)

test_image_files = os.listdir(test_image_folder)
test_mask_files = os.listdir(test_mask_folder)

#Create full file paths
train_image_paths = [os.path.join(train_image_folder, filename) for filename in train_image_files]
train_mask_paths = [os.path.join(train_mask_folder, filename) for filename in train_mask_files]


test_image_paths = [os.path.join(test_image_folder, filename) for filename in test_image_files]
test_mask_paths = [os.path.join(test_mask_folder, filename) for filename in test_mask_files]


train_image_paths = sorted(train_image_paths)
train_mask_paths = sorted(train_mask_paths)


test_image_paths = sorted(test_image_paths)
test_mask_paths = sorted(test_mask_paths)


import tensorflow as tf
BATCH_SIZE = 8
IMAGE_SIZE = 512


def elastic_transform_tf(image, mask, alpha=40, sigma=6):
    if hasattr(alpha, 'numpy'):
        alpha = float(alpha.numpy())
    if hasattr(sigma, 'numpy'):
        sigma = float(sigma.numpy())
    image_np = image.numpy()
    mask_np = mask.numpy()
    shape = image_np.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,)))
    # Image
    if image_np.ndim == 3:
        channels = [map_coordinates(image_np[..., c], indices, order=1, mode='reflect').reshape(shape)
                    for c in range(image_np.shape[-1])]
        image_deformed = np.stack(channels, axis=-1)
    else:
        image_deformed = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
    # Mask 
    if mask_np.ndim == 3:
        channels = [map_coordinates(mask_np[..., c], indices, order=0, mode='reflect').reshape(shape)
                    for c in range(mask_np.shape[-1])]
        mask_deformed = np.stack(channels, axis=-1)
    else:
        mask_deformed = map_coordinates(mask_np, indices, order=0, mode='reflect').reshape(shape)
    return image_deformed.astype(np.float32), mask_deformed.astype(np.float32)


def tf_elastic_transform(image, mask, alpha=40, sigma=6, prob=0.3):
    def _transform(image, mask):
        image_out, mask_out = tf.py_function(
            func=elastic_transform_tf,
            inp=[image, mask, alpha, sigma],
            Tout=[tf.float32, tf.float32]
        )
        return (image_out, mask_out)

    apply = tf.random.uniform([]) < prob
    image, mask = tf.cond(apply,
        lambda: _transform(image, mask),
        lambda: (image, mask))
    image.set_shape([None, None, 3])
    mask.set_shape([None, None, 1])
    return image, mask


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image = tf.cast(image, tf.float32) / 255.0 
        image = tf.where(image >= 0.5, 1.0, 0.0)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    return image

def random_translate(image, mask, max_translate=20):
    # Generate random translation values
    translate_x = tf.random.uniform(shape=[], minval=-max_translate, maxval=max_translate, dtype=tf.int32)
    translate_y = tf.random.uniform(shape=[], minval=-max_translate, maxval=max_translate, dtype=tf.int32)

    # Translate the image and mask
    image = tf.roll(image, shift=translate_x, axis=1)
    image = tf.roll(image, shift=translate_y, axis=0)
    mask = tf.roll(mask, shift=translate_x, axis=1)
    mask = tf.roll(mask, shift=translate_y, axis=0)

    return image, mask


IMAGE_SIZE = 512  

def random_crop_and_resize(image, mask, crop_size, image_size):
    stacked = tf.concat([image, mask], axis=-1)
    crop = tf.image.random_crop(stacked, size=[crop_size, crop_size, tf.shape(stacked)[-1]])
    image_c = crop[..., :image.shape[-1]]
    mask_c = crop[..., image.shape[-1]:]
    image_c = tf.image.resize(image_c, [image_size, image_size])
    mask_c = tf.image.resize(mask_c, [image_size, image_size])
    return image_c, mask_c

def add_gaussian_noise(image, stddev=0.1):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=image.dtype)
    return image + noise

def augment_data(image, mask):
    # RandomResizedCrop
    if tf.random.uniform(()) > 0.5:
        image, mask = random_crop_and_resize(image, mask, crop_size=410, image_size=IMAGE_SIZE)

    # ElasticTransform (custom)
    image, mask = tf_elastic_transform(image, mask, alpha=40, sigma=8, prob=0.3)

    # Horizontal Flip
    if tf.random.uniform(()) > 0.6:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    # Vertical Flip
    if tf.random.uniform(()) > 0.4:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    # Rotate (0, 90, 180, 270 degrees)
    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
    # Random Brightness/Contrast
    if tf.random.uniform(()) > 0.7:
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.3)
  
    if tf.random.uniform(()) > 0.4:
        image = tf.image.random_hue(image, max_delta=0.08)
        image = tf.image.random_saturation(image, lower=0.88, upper=1.12)
        image = tf.image.random_brightness(image, max_delta=0.08)
    # Final clip, resize, binarize mask
    image = tf.clip_by_value(image, 0, 1)
    mask = tf.clip_by_value(mask, 0, 1)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    mask = tf.image.resize(mask, [IMAGE_SIZE, IMAGE_SIZE])
    mask = tf.where(mask >= 0.5, 1.0, 0.0)
    return image, mask



def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list, augment=False, batch_size=None):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset



@tf.keras.utils.register_keras_serializable(package="Custom", name="ConvBlock")
class ConvBlock(Layer):

    def __init__(self, filters=256, kernel_size=3, dilation_rate=1, dropout_rate=0.3, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, use_bias=False, kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Dropout(rate=dropout_rate) 
        ])

    def call(self, X):
        return self.net(X)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate  
        }


def AtrousSpatialPyramidPooling(X):
    B, H, W, C = X.shape

    # Image Pooling
    image_pool = AveragePooling2D(pool_size=(H, W), name="ASPP-AvgPool")(X)
    image_pool = ConvBlock(kernel_size=1, name="ASPP-ImagePool-CB")(image_pool)
    image_pool = UpSampling2D(size=(H//image_pool.shape[1], W//image_pool.shape[2]), name="ASPP-ImagePool-UpSample")(image_pool)

    # Atrous Oprtations
    conv_1  = ConvBlock(kernel_size=1, dilation_rate=1, name="ASPP-CB-1")(X)
    conv_6  = ConvBlock(kernel_size=3, dilation_rate=6, name="ASPP-CB-6")(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12, name="ASPP-CB-12")(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18, name="ASPP-CB-18")(X)

    # Combine All
    combined = Concatenate(name="ASPP-Combine")([image_pool, conv_1, conv_6, conv_12, conv_18])
    processed = ConvBlock(kernel_size=1, name="ASPP-Net")(combined)

    # Final Output
    return processed


IMAGE_SIZE = 512

def build_model():
    # Input
    InputL = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="InputLayer")

    # Base Mode
    backbone = Xception(include_top=False, weights='imagenet', input_tensor=InputL)

    # ASPP Phase
    DCNN = backbone.get_layer('block11_sepconv1_act').output
    print(DCNN.shape)
    DCNN = tf.keras.layers.Dropout(rate=0.3)(DCNN)
    ASPP = AtrousSpatialPyramidPooling(DCNN)
    ASPP = UpSampling2D(size=(IMAGE_SIZE//4//ASPP.shape[1], IMAGE_SIZE//4//ASPP.shape[2]), name="AtrousSpatial")(ASPP)
    print(ASPP.shape)
    # LLF Phase
    LLF = backbone.get_layer('block3_sepconv1_act').output
    LLF = ZeroPadding2D(padding=((0, 1), (0, 1)))(LLF)
    LLF = ConvBlock(filters=50, kernel_size=1, dropout_rate=0.3, name="LLF-ConvBlock")(LLF)
    # LLF = keras.layers.Dropout(rate=0.3)(LLF)
    print(LLF.shape)

    # Combined
    combined = Concatenate(axis=-1, name="Combine-LLF-ASPP")([ASPP, LLF])
    features = ConvBlock(dropout_rate=0.3,name="Top-ConvBlock-1")(combined)
    features = ConvBlock(name="Top-ConvBlock-2")(features)
    upsample = UpSampling2D(size=(IMAGE_SIZE//features.shape[1], IMAGE_SIZE//features.shape[1]), interpolation='bilinear', name="Top-UpSample")(features)

    # Output Mask
    PredMask = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid', use_bias=False, name="OutputMask")(upsample)

    # DeelLabV3+ Model
    model = Model(InputL, PredMask, name="DeepLabV3-Plus")

    return model


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


import tensorflow as tf

def make_bce_focal_dice_loss(
    bce_weight=0.3,
    dice_weight=0.7,
    pos_weight=20.0,
    use_focal=True,
    focal_alpha=0.25,
    focal_gamma=2.0,
    smooth=1e-6
):
    """Returns a (y_true, y_pred) -> loss function."""
    bce_weight  = float(bce_weight)
    dice_weight = float(dice_weight)
    pos_weight  = float(pos_weight)
    focal_alpha = float(focal_alpha)
    focal_gamma = float(focal_gamma)
    smooth      = float(smooth)
    use_focal   = bool(use_focal)

    def loss_fn(y_true, y_pred):
        # y_pred are probs 
        
        eps = tf.keras.backend.epsilon()
        y_prob = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        y_true = tf.cast(y_true, y_prob.dtype)

        # BCE or Focal
        # plain elementwise BCE
        bce = -(y_true * tf.math.log(y_prob) + (1.0 - y_true) * tf.math.log(1.0 - y_prob))

        if use_focal:
            # focal 
            pt = tf.exp(-bce)
            bce_term = tf.reduce_mean(focal_alpha * tf.pow(1.0 - pt, focal_gamma) * bce)
        else:
            # weighted BCE 
            bce_term = tf.reduce_mean(
                -(pos_weight * y_true * tf.math.log(y_prob) + (1.0 - y_true) * tf.math.log(1.0 - y_prob))
            )

        # Dice 
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_prob_f = tf.reshape(y_prob, [tf.shape(y_prob)[0], -1])
        inter = tf.reduce_sum(y_true_f * y_prob_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_prob_f, axis=1)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        dice_term = 1.0 - tf.reduce_mean(dice)

        return bce_weight * bce_term + dice_weight * dice_term

    return loss_fn


loss_fn = make_bce_focal_dice_loss(
    bce_weight=0.3, dice_weight=0.7,
    pos_weight=20.0, use_focal=True,
    focal_alpha=0.25, focal_gamma=2.0,
    smooth=1e-6
)

tf.keras.utils.get_custom_objects()['bce_dice_loss'] = loss_fn


num_folds = 3
BATCH_SIZE = 8
EPOCHS = 250

# Initialize KFold cross-validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
all_train_histories = []
all_val_histories = []

for fold, (train_index, val_index) in enumerate(kf.split(train_image_paths)):
    print(f"\n========= Training fold {fold + 1}/{num_folds} =========")

    # Split data for this fold
    fold_train_images = [train_image_paths[i] for i in train_index]
    fold_train_masks = [train_mask_paths[i] for i in train_index]
    fold_val_images = [train_image_paths[i] for i in val_index]
    fold_val_masks = [train_mask_paths[i] for i in val_index]

    # Build and compile a new model for this fold
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-3),
        loss=loss_fn,
        metrics=[dice_coef, "accuracy"]
    )

    # Prepare data generators
    fold_train_dataset = data_generator(fold_train_images, fold_train_masks, augment=True, batch_size=BATCH_SIZE)
    fold_val_dataset = data_generator(fold_val_images, fold_val_masks, batch_size=BATCH_SIZE)

    # Set up callbacks 
    checkpoint = ModelCheckpoint(
        f'./model_deeplab_corrosion_fold{fold+1}.keras',
        monitor='val_dice_coef',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='max',  
    )

    early_stop = EarlyStopping(
        monitor='val_dice_coef',
        patience=25,
        verbose=1,
        restore_best_weights=True,
        mode='max', 
    )

    # Fit model
    history = model.fit(
        fold_train_dataset,
        validation_data=fold_val_dataset,
        callbacks=[checkpoint, early_stop],
        epochs=EPOCHS
    )

    # Save histories
    all_train_histories.append(history.history)
    all_val_histories.append(history.history['val_loss'])

print("Training complete for all folds!")

import tensorflow as tf

IMAGE_SIZE = 512
BATCH_SIZE = 1

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1] for mask images
        image = tf.where(image >= 0.5, 1.0, 0.0)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1] for RGB images

        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    return image

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

test_dataset = data_generator(test_image_paths, test_mask_paths)

print("Test Dataset:", test_dataset)

    
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os


# Load all three models
fold_models = []
for fold in range(1, 4):
    model_path = f"./model_deeplab_corrosion_fold{fold}.keras"
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef, 'loss_fn': loss_fn })
    fold_models.append(model)
    print(f"Loaded model for fold {fold}")

# Creating a folder to save the average predicted masks
output_dir = "./average_predicted_masks"
os.makedirs(output_dir, exist_ok=True)  

def calculate_metrics(model, test_dataset):
    accuracy_scores = []
    dice_scores = []
    iou_scores = []
    tp_scores = []
    tn_scores = []
    fp_scores = []
    fn_scores = []
    all_mask_flat = []
    all_pred_flat = []

    for image, mask in test_dataset:
        pred_mask = model.predict(image, verbose=0)
        pred_mask_binary = (pred_mask > 0.5).astype(np.float32)
        mask = mask.numpy()

        # Original mask_flat for other metrics
        mask_flat = mask.flatten()
        pred_mask_flat = pred_mask_binary.flatten()
        pred_mask_prob_flat = pred_mask.flatten()  # Original probability for AUC

        # Binarized mask for AUC only
        mask_flat_auc = (mask > 0.5).astype(np.float32).flatten()  

        # Store for AUC
        all_mask_flat.append(mask_flat_auc)
        all_pred_flat.append(pred_mask_prob_flat)

        # Calculate TP, TN, FP, FN
        tp = np.sum(mask_flat * pred_mask_flat)
        tn = np.sum((1 - mask_flat) * (1 - pred_mask_flat))
        fp = np.sum((1 - mask_flat) * pred_mask_flat)
        fn = np.sum(mask_flat * (1 - pred_mask_flat))

        # Calculate accuracy, Dice score, IoU score
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        dice = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)
        iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)

        accuracy_scores.append(accuracy)
        dice_scores.append(dice)
        iou_scores.append(iou)
        tp_scores.append(tp)
        tn_scores.append(tn)
        fp_scores.append(fp)
        fn_scores.append(fn)

    # Flatten for AUC
    all_mask_flat = np.concatenate(all_mask_flat)
    all_pred_flat = np.concatenate(all_pred_flat)

    
    print("all_mask_flat shape:", all_mask_flat.shape, "unique values:", np.unique(all_mask_flat))
    print("all_pred_flat shape:", all_pred_flat.shape, "min:", np.min(all_pred_flat), "max:", np.max(all_pred_flat))
    print("NaN in all_mask_flat:", np.any(np.isnan(all_mask_flat)))
    print("NaN in all_pred_flat:", np.any(np.isnan(all_pred_flat)))

    # Compute AUC
    try:
        if len(np.unique(all_mask_flat)) > 1:  # Ensure both classes are present
            auc = roc_auc_score(all_mask_flat, all_pred_flat)
        else:
            auc = np.nan
            print("Warning: AUC cannot be computed because only one class is present in the ground truth.")
    except ValueError as e:
        auc = np.nan
        print(f"Error computing AUC: {e}")

    avg_accuracy = np.mean(accuracy_scores)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    TP = np.sum(tp_scores)
    TN = np.sum(tn_scores)
    FP = np.sum(fp_scores)
    FN = np.sum(fn_scores)

    f1 = 2 * TP / (2 * TP + FP + FN + 1e-7)  # Fixed F1 score formula
    recall = TP / (FN + TP + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    specificity = TN / (FP + TN + 1e-7)
    MCC = (TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-7)

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("Precision Score:", precision)
    print("Sensitivity/Recall Score:", recall)
    print("Specificity Score:", specificity)
    print("F1 Score:", f1)
    print("MCC:", MCC)

    return {
        'accuracy': avg_accuracy,
        'dice': avg_dice,
        'iou': avg_iou,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'mcc': MCC
    }

# Function to save average predicted masks using plt with test image filenames
def save_average_predicted_masks(fold_models, test_dataset, output_dir, test_image_filenames):
    image_idx = 0 
    for image, _ in test_dataset:
        # Get predictions from all models
        pred_masks = []
        for model in fold_models:
            pred_mask = model.predict(image, verbose=0)
            pred_masks.append(pred_mask)
        
        # Compute the average predicted mask
        avg_pred_mask = np.mean(pred_masks, axis=0) 
        avg_pred_mask = (avg_pred_mask > 0.5).astype(np.uint8)  

        # Save each image average predicted mask in the batch
        for i in range(avg_pred_mask.shape[0]):
            if image_idx >= len(test_image_filenames):
                print(f"Warning: More masks ({image_idx}) than filenames ({len(test_image_filenames)}). Skipping.")
                break
            mask_to_save = avg_pred_mask[i].squeeze()  
           
            try:
                base_filename = os.path.basename(test_image_filenames[image_idx])
                name, ext = os.path.splitext(base_filename)
                if not ext: 
                    ext = '.png'
                output_filename = f"avg_pred_mask_{name}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                plt.imsave(output_path, mask_to_save, cmap='gray')  # Save as grayscale image
                print(f"Saved average predicted mask {image_idx} to {output_path}")
            except (IndexError, TypeError, ValueError) as e:
                print(f"Error processing filename at index {image_idx}: {test_image_filenames[image_idx]}. Skipping. Error: {e}")
            image_idx += 1


# Evaluate each fold model
results = []
for i, m in enumerate(fold_models):
    print(f"\n--- Fold {i+1} Test Results ---")
    metrics = calculate_metrics(m, test_dataset)
    results.append(metrics)

# Save average predicted masks
print("\nSaving average predicted masks...")
save_average_predicted_masks(fold_models, test_dataset, output_dir, test_image_paths)

# Aggregate and print average over all folds
print("\n Average over all folds ")
keys = results[0].keys()
avg_results = {k: np.mean([r[k] for r in results]) for k in keys}
for k, v in avg_results.items():
    print(f"{k}: {v:.4f}")


