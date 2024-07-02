from imports import *
from utils import *

# GPU check -------------------------------------------------------------------------------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Call the function to configure GPUs
configure_tensorflow_gpus()

# Define the TensorFlow distribution strategy globally
strategy = tf.distribute.MirroredStrategy()

# Suppress TensorFlow logging (except for first initialization)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs shown, 1 = filter out INFO logs, 2 = filter out WARNING logs, 3 = filter out ERROR logs
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging at ERROR level

# Data preprocessing -----------------------------------------------------------------------------------------------------------------------------------------
# Setting multiple seeds for reproducibility
SEED = 111
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

PATH = 'extracted_files/Data'
batch_size = 32
image_dim = (168, 168)

# Load the entire dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    PATH,
    image_size=image_dim,
    # batch_size=batch_size,
    seed=SEED,
    shuffle=True
)

# Calculate the sizes for each split
total_size = len(full_dataset)
initial_train_size = int(0.01 * total_size)
later_train_size = int(0.79 * total_size)
test_size = total_size - initial_train_size - later_train_size

# Split the dataset
initial_train_ds = full_dataset.take(initial_train_size)
remaining_ds = full_dataset.skip(initial_train_size)
later_train_ds = remaining_ds.take(later_train_size)
test_ds = remaining_ds.skip(later_train_size)

# Print dataset sizes for verification
print(f"Initial training dataset size: {len(initial_train_ds)}")
print(f"Later training dataset size: {len(later_train_ds)}")
print(f"Test dataset size: {len(test_ds)}")

class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
inv_class_mappings = {v: k for k, v in class_mappings.items()}
class_names = list(class_mappings.keys())

num_classes = len(class_mappings.keys())
image_shape = (image_dim[0], image_dim[1], 1)

# Training epochs and batch size
epochs = 50
print(f'Number of Classes: {num_classes}')
print(f'Image shape: {image_shape}')
print(f'Epochs: {epochs}')
print(f'Batch size: {batch_size}')

def encode_labels(image, label):
    return image, tf.one_hot(label, depth=num_classes)

# Preprocess and encode labels
initial_train_ds_preprocessed = initial_train_ds.map(encode_labels, num_parallel_calls=tf.data.AUTOTUNE)
later_train_ds_preprocessed = later_train_ds.map(encode_labels, num_parallel_calls=tf.data.AUTOTUNE)
test_ds_preprocessed = test_ds.map(encode_labels, num_parallel_calls=tf.data.AUTOTUNE)

model = Sequential([
    # Input tensor shape
    Input(shape=(168, 168, 3)),
    
    # Convolutional layer 1
    Conv2D(64, (5, 5), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),

    # Convolutional layer 2
    Conv2D(64, (5, 5), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),

    # Convolutional layer 3
    Conv2D(128, (4, 4), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional layer 4
    Conv2D(128, (4, 4), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),

    # Dense layers 
    Dense(512, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# Model summary
model.summary()

# Compiling model with Adam optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define log directory for TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Callbacks
model_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=1e-4, patience=4, verbose=False)
# model_mc = ModelCheckpoint('model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=False)

# Function to get least confident samples
def get_least_confident_samples(model, dataset, num_samples):
    predictions = []
    for batch in dataset:
        images, _ = batch
        preds = model.predict(images)
        predictions.extend(preds)
    predictions = np.array(predictions)
    confidence = np.max(predictions, axis=1)
    least_confident_indices = np.argsort(confidence)[:num_samples]
    data_list = list(dataset.unbatch().as_numpy_iterator())
    least_confident_samples = [data_list[i] for i in least_confident_indices]
    return tf.data.Dataset.from_tensor_slices((np.array([sample[0] for sample in least_confident_samples]), np.array([sample[1] for sample in least_confident_samples])))

# Function to get random samples
def get_random_samples(dataset, num_samples):
    # Convert the dataset to a list of elements
    data_list = list(dataset.unbatch().as_numpy_iterator())
    # Randomly sample indices
    random_indices = np.random.choice(len(data_list), num_samples, replace=False)
    # Ensure that the samples have consistent shapes
    random_samples = [data_list[i] for i in random_indices]
    random_samples_x = np.array([sample[0] for sample in random_samples])
    random_samples_y = np.array([sample[1] for sample in random_samples])
    return tf.data.Dataset.from_tensor_slices((random_samples_x, random_samples_y))

# Function to combine least confident and random samples
def get_least_confident_and_random_samples(model, dataset, num_samples):
    num_least_confident = num_samples // 2
    num_random = num_samples - num_least_confident
    least_confident_samples = get_least_confident_samples(model, dataset, num_least_confident)
    random_samples = get_random_samples(dataset, num_random)
    combined_samples = least_confident_samples.concatenate(random_samples)
    return combined_samples

# Training with Active Learning Loop
epochs_per_iteration = 5
iterations = epochs // epochs_per_iteration
num_samples_to_add = 10

# Least confidence sampling -----------------------------------------------------------------------------------------------------------------------------------------
print("Least confidence sampling:")
initial_train_ds_preprocessed_least_confident = initial_train_ds_preprocessed
later_train_ds_preprocessed_least_confident = later_train_ds_preprocessed
for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}")
    history = model.fit(
        initial_train_ds_preprocessed_least_confident,
        epochs=epochs_per_iteration,
        validation_data=test_ds_preprocessed,
        callbacks=[model_rlr, tensorboard_callback],
        verbose=True
    )
    new_samples = get_least_confident_samples(model, later_train_ds_preprocessed_least_confident, num_samples_to_add)
    initial_train_ds_preprocessed_least_confident = initial_train_ds_preprocessed_least_confident.concatenate(new_samples)
    later_train_ds_preprocessed_least_confident = later_train_ds_preprocessed_least_confident.skip(num_samples_to_add)

# Random sampling -----------------------------------------------------------------------------------------------------------------------------------------
print("Random sampling:")
initial_train_ds_preprocessed_random = initial_train_ds_preprocessed
later_train_ds_preprocessed_random = later_train_ds_preprocessed
for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}")
    history = model.fit(
        initial_train_ds_preprocessed_random,
        epochs=epochs_per_iteration,
        validation_data=test_ds_preprocessed,
        callbacks=[model_rlr, tensorboard_callback],
        verbose=True
    )
    new_samples = get_random_samples(later_train_ds_preprocessed_random, num_samples_to_add)
    initial_train_ds_preprocessed_random = initial_train_ds_preprocessed_random.concatenate(new_samples)
    later_train_ds_preprocessed_random = later_train_ds_preprocessed_random.skip(num_samples_to_add)

# Hybrid (Least confidence + Random sampling) -----------------------------------------------------------------------------------------------------------------------------------------
print("Hybrid sampling (Least confidence + Random):")
initial_train_ds_preprocessed_hybrid = initial_train_ds_preprocessed
later_train_ds_preprocessed_hybrid = later_train_ds_preprocessed
for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}")
    history = model.fit(
        initial_train_ds_preprocessed_hybrid,
        epochs=epochs_per_iteration,
        validation_data=test_ds_preprocessed,
        callbacks=[model_rlr, tensorboard_callback],
        verbose=True
    )
    new_samples = get_least_confident_and_random_samples(model, later_train_ds_preprocessed_hybrid, num_samples_to_add)
    initial_train_ds_preprocessed_hybrid = initial_train_ds_preprocessed_hybrid.concatenate(new_samples)
    later_train_ds_preprocessed_hybrid = later_train_ds_preprocessed_hybrid.skip(num_samples_to_add)

# Final training of the model
history = model.fit(
    initial_train_ds_preprocessed_hybrid,
    epochs=epochs,
    validation_data=test_ds_preprocessed,
    callbacks=[model_rlr, tensorboard_callback],
    verbose=True
)
