from imports import *

def setup_gpu(device_id='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)

    print("Available devices:", tf.config.list_physical_devices())
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow will be able to use the GPU!")
    else:
        print("Make sure TensorFlow can detect the GPU with the updated drivers!")

def setup_logging(tf_log_level='3'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level  # Suppress most TensorFlow logging
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging at ERROR level
    # optuna.logging.set_verbosity(optuna.logging.WARNING)

def setup_strategy(strategy_type='MirroredStrategy'):
    if strategy_type == 'MirroredStrategy':
        print(f"Using strategy: Mirrored Strategy")
        return tf.distribute.MirroredStrategy()
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False