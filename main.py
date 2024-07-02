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
    predictions = model.predict(dataset)
    confidence = np.max(predictions, axis=1)
    least_confident_indices = np.argsort(confidence)[:num_samples]
    return least_confident_indices

# Function to get random samples
def get_random_samples(dataset, num_samples):
    # Convert the dataset to a list of elements
    data_list = list(dataset.as_numpy_iterator())
    # Randomly sample indices
    random_indices = np.random.choice(len(data_list), num_samples, replace=False)
    # Ensure that the samples have consistent shapes
    random_samples = [data_list[i] for i in random_indices]
    random_samples_x = np.array([sample[0] for sample in random_samples])
    random_samples_y = np.array([sample[1] for sample in random_samples])
    return tf.data.Dataset.from_tensor_slices((random_samples_x, random_samples_y))

# Function to get diverse samples using KMeans clustering
def get_diverse_samples(dataset, num_clusters, num_samples_per_cluster):
    # Convert the dataset to a list of elements
    data_list = list(dataset.as_numpy_iterator())
    data_x = np.array([sample[0] for sample in data_list])
    
    # Print the shape of the data for debugging
    print("Original data shape:", data_x.shape)  # Expected shape: (535, 32, 168, 168, 3)
    
    # Combine the first two dimensions (num_samples and batch_size)
    num_samples, batch_size, height, width, channels = data_x.shape
    data_x = data_x.reshape(num_samples * batch_size, height * width * channels)
    
    # Print the new shape of the data for debugging
    print("Reshaped data shape:", data_x.shape)  # Expected shape: (535 * 32, 168 * 168 * 3)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=SEED).fit(data_x)
    cluster_labels = kmeans.labels_
    
    diverse_indices = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            # Select random samples from each cluster
            selected_indices = np.random.choice(cluster_indices, num_samples_per_cluster, replace=False)
            diverse_indices.extend(selected_indices)
    
    return diverse_indices

# Function to combine least confident and diverse sampling
def get_least_confident_and_diverse_samples(model, dataset, num_samples, num_clusters):
    least_confident_indices = get_least_confident_samples(model, dataset, num_samples)
    diverse_indices = get_random_samples(dataset, num_samples)
    combined_indices = list(set(least_confident_indices) | set(diverse_indices))
    return combined_indices[:num_samples]

# Training with Active Learning Loop
epochs_per_iteration = 5
iterations = epochs // epochs_per_iteration
num_samples_to_add = 30
num_clusters = 4


# Least confidence sampling -----------------------------------------------------------------------------------------------------------------------------------------
print(len(initial_train_ds_preprocessed))

# for i in range(iterations):
    # print(f"Iteration {i + 1}/{iterations}")

    # # Train the model
    # history = model.fit(
    #     initial_train_ds_preprocessed,
    #     epochs=epochs_per_iteration,
    #     validation_data=test_ds_preprocessed,
    #     callbacks=[model_rlr, tensorboard_callback],
    #     verbose=False
    # )

    # # Get least confident samples from the pool of unlabeled data
    # least_confident_indices = get_least_confident_samples(model, later_train_ds_preprocessed, num_samples_to_add)

    # # Select the least confident samples
    # new_samples = later_train_ds_preprocessed.take(num_samples_to_add)

    # # Add these samples to the training dataset
    # initial_train_ds_preprocessed = initial_train_ds_preprocessed.concatenate(new_samples)

    # # Remove added samples from the pool
    # later_train_ds_preprocessed = later_train_ds_preprocessed.skip(num_samples_to_add)

# 506/506 [==============================] - 20s 39ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 0.4201 - val_accuracy: 0.9853 - lr: 1.0000e-04

# Random sampling -----------------------------------------------------------------------------------------------------------------------------------------
print(len(initial_train_ds_preprocessed))
for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}")

    # Train the model
    history = model.fit(
        initial_train_ds_preprocessed,
        epochs=epochs_per_iteration,
        validation_data=test_ds_preprocessed,
        callbacks=[model_rlr, tensorboard_callback],
        # verbose=True
    )

    # Get random samples from the pool of unlabeled data
    new_samples = get_random_samples(later_train_ds_preprocessed, num_samples_to_add)

    # Add these samples to the training dataset
    initial_train_ds_preprocessed = initial_train_ds_preprocessed.concatenate(new_samples)

    # Remove added samples from the pool
    later_train_ds_preprocessed = later_train_ds_preprocessed.skip(num_samples_to_add)

# 106/106 [==============================] - 4s 41ms/step - loss: 8.7861e-10 - accuracy: 1.0000 - val_loss: 0.7185 - val_accuracy: 0.9589 - lr: 1.0000e-04


# Least confidence + diversity sampling -----------------------------------------------------------------------------------------------------------------------------------------

# Final training of the model

print(len(initial_train_ds_preprocessed))
history = model.fit(
    initial_train_ds_preprocessed,
    epochs=epochs,
    validation_data=test_ds_preprocessed,
    callbacks=[model_rlr, tensorboard_callback],
    verbose=True
)

# history = model.fit(
#     later_train_ds_preprocessed,
#     epochs=epochs,
#     validation_data=test_ds_preprocessed,
#     callbacks=[model_rlr, tensorboard_callback],
#     verbose=True
# )

# tensorboard --logdir logs/fit