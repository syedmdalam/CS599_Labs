import tensorflow as tf
from tensorflow.keras.utils import get_file
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

tf.random.set_seed(0)
np.random.seed(0)



def load_data(file_name='notMNIST_small.mat'):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    
    # 1. Check if file exists locally
    if not os.path.exists(file_name):
        print(f"\n[ERROR] File not found: {file_name}")
        print(f"Python is looking in this folder: {current_dir}")
        print("Please move 'notMNIST_small.mat' into this folder.\n")
        # Return None to signal failure
        return None

    print(f"Loading data from local file: {file_path}...")
    
    try:
        data = loadmat(file_name)
    except Exception as e:
        print(f"[ERROR] Could not read the .mat file. Details: {e}")
        return None

    X = data['images']  # Shape: (28, 28, N)
    y = data['labels']  # Shape: (N,)
    
    X = np.transpose(X, (2, 0, 1))
    X = X.astype('float32') / 255.0
    y = y.astype('int32')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data loaded. Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

def create_dataset(x, y, batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


########################### GRU CELL IMPLEMENTATION ###########################

class GRU_Cell(object):
    def __init__(self, input_units, hidden_units):
        self.input_units = input_units
        self.hidden_units = hidden_units
        
        # Helper to create weights
        def weight(shape):
            return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))
        def bias(shape):
            return tf.Variable(tf.zeros(shape))

        self.Wz = weight([input_units, hidden_units])
        self.Uz = weight([hidden_units, hidden_units])
        self.bz = bias([hidden_units])

        self.Wr = weight([input_units, hidden_units])
        self.Ur = weight([hidden_units, hidden_units])
        self.br = bias([hidden_units])

        self.Wh = weight([input_units, hidden_units])
        self.Uh = weight([hidden_units, hidden_units])
        self.bh = bias([hidden_units])

        self.trainable_variables = [
            self.Wz, self.Uz, self.bz,
            self.Wr, self.Ur, self.br,
            self.Wh, self.Uh, self.bh
        ]

    def __call__(self, prev_state, x):
        z = tf.sigmoid(tf.matmul(x, self.Wz) + tf.matmul(prev_state, self.Uz) + self.bz)
        r = tf.sigmoid(tf.matmul(x, self.Wr) + tf.matmul(prev_state, self.Ur) + self.br)
        reset_hidden = r * prev_state
        s_tilde = tf.tanh(tf.matmul(x, self.Wh) + tf.matmul(reset_hidden, self.Uh) + self.bh)
        s = (1 - z) * prev_state + z * s_tilde
        return s


############################ MGU CELL IMPLEMENTATION ############################

class MGU_Cell(object):
    def __init__(self, input_units, hidden_units):
        self.input_units = input_units
        self.hidden_units = hidden_units
        
        def weight(shape):
            return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))
        def bias(shape):
            return tf.Variable(tf.zeros(shape))
        
        self.Wf = weight([input_units, hidden_units])
        self.Uf = weight([hidden_units, hidden_units])
        self.bf = bias([hidden_units])

        self.Wh = weight([input_units, hidden_units])
        self.Uh = weight([hidden_units, hidden_units])
        self.bh = bias([hidden_units])

        self.trainable_variables = [
            self.Wf, self.Uf, self.bf,
            self.Wh, self.Uh, self.bh
        ]

    def __call__(self, prev_state, x):
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_state, self.Uf) + self.bf)
        forgotten_hidden = f * prev_state
        s_tilde = tf.tanh(tf.matmul(x, self.Wh) + tf.matmul(forgotten_hidden, self.Uh) + self.bh)
        s = (1 - f) * prev_state + f * s_tilde
        return s


############################ Training and Eval Functions ############################

# def train_and_evaluate(cell_class, x_train, y_train, x_test, y_test, 
#                        hidden_units=128, num_layers=1, epochs=10, batch_size=64):
    
#     input_dim = x_train.shape[2] # 28
#     cells = []
#     for i in range(num_layers):
#         in_dim = input_dim if i == 0 else hidden_units
#         cells.append(cell_class(in_dim, hidden_units))
    
#     # Output Layer
#     output_classes = 10
#     Wo = tf.Variable(tf.random.truncated_normal([hidden_units, output_classes], stddev=0.1))
#     bo = tf.Variable(tf.zeros([output_classes]))
    
#     optimizer = tf.optimizers.Adam(learning_rate=0.001)
#     train_dataset = create_dataset(x_train, y_train, batch_size)
#     test_dataset = create_dataset(x_test, y_test, batch_size, shuffle=False)
    
#     history = {'loss': [], 'test_error': []}
    
#     for epoch in range(epochs):
#         total_loss = 0
#         steps = 0
#         for x_batch, y_batch in train_dataset:
#             with tf.GradientTape() as tape:
#                 states = [tf.zeros([batch_size, hidden_units]) for _ in range(num_layers)]
#                 for t in range(x_batch.shape[1]):
#                     current_input = x_batch[:, t, :]
#                     new_states = []
#                     for i, cell in enumerate(cells):
#                         # cell(prev_state, input)
#                         h = cell(states[i], current_input)
#                         new_states.append(h)
#                         current_input = h 
                    
#                     states = new_states
#                 logits = tf.matmul(states[-1], Wo) + bo
#                 loss = tf.reduce_mean(
#                     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=logits)
#                 )
#             all_vars = [Wo, bo]
#             for cell in cells:
#                 all_vars += cell.trainable_variables
            
#             grads = tape.gradient(loss, all_vars)
#             optimizer.apply_gradients(zip(grads, all_vars))
#             total_loss += loss
#             steps += 1
            
#         avg_loss = total_loss / steps
#         history['loss'].append(avg_loss.numpy())
#         test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
#         for x_batch_test, y_batch_test in test_dataset:
#             states = [tf.zeros([batch_size, hidden_units]) for _ in range(num_layers)]
#             for t in range(x_batch_test.shape[1]):
#                 current_input = x_batch_test[:, t, :]
#                 new_states = []
#                 for i, cell in enumerate(cells):
#                     h = cell(states[i], current_input)
#                     new_states.append(h)
#                     current_input = h
#                 states = new_states
            
#             logits = tf.matmul(states[-1], Wo) + bo
#             test_acc_metric.update_state(y_batch_test, logits)
            
#         test_error = 1.0 - test_acc_metric.result().numpy()
#         history['test_error'].append(test_error)
#         print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Error={test_error:.4f}")

#     return history




def train_and_evaluate(cell_class, x_train, y_train, x_test, y_test, 
                       hidden_units=128, epochs=10, batch_size=64):
    
    input_dim = x_train.shape[2] # 28 columns
    cell = cell_class(input_dim, hidden_units)

    output_classes = 10
    Wo = tf.Variable(tf.random.truncated_normal([hidden_units, output_classes], stddev=0.1))
    bo = tf.Variable(tf.zeros([output_classes]))
    
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    train_dataset = create_dataset(x_train, y_train, batch_size)
    test_dataset = create_dataset(x_test, y_test, batch_size, shuffle=False)
    
    history = {'loss': [], 'test_acc': [], 'test_error': []}
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                state = tf.zeros([batch_size, hidden_units])
                for t in range(x_batch.shape[1]):
                    state = cell(state, x_batch[:, t, :])
                
                logits = tf.matmul(state, Wo) + bo
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=logits)
                )
            vars_to_update = cell.trainable_variables + [Wo, bo]
            grads = tape.gradient(loss, vars_to_update)
            optimizer.apply_gradients(zip(grads, vars_to_update))
            
            total_loss += loss
            steps += 1
            
        avg_loss = total_loss / steps
        history['loss'].append(avg_loss.numpy())
        test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        for x_batch_test, y_batch_test in test_dataset:
            state = tf.zeros([batch_size, hidden_units])
            for t in range(x_batch_test.shape[1]):
                state = cell(state, x_batch_test[:, t, :])
            logits = tf.matmul(state, Wo) + bo
            test_acc_metric.update_state(y_batch_test, logits)
            
        test_acc = test_acc_metric.result().numpy()
        test_error = 1.0 - test_acc 
        
        history['test_acc'].append(test_acc)
        history['test_error'].append(test_error)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={test_acc:.4f}, Error={test_error:.4f}")

    return history

##################### Run the Experiments #####################

# def run_comparison():
#     data = load_data('notMNIST_small.mat')
#     if data is None: return

#     (x_train, y_train), (x_test, y_test) = data
#     configs = [
#         {'name': 'Baseline (1 Layer, 128 Units)', 'units': 128, 'layers': 1},
#         {'name': 'Deep (2 Layers, 64 Units)',    'units': 64,  'layers': 2},
#     ]
    
#     trials = 1 # Set to 3 for final report
#     results = {} 

#     for conf in configs:
#         conf_name = conf['name']
#         u = conf['units']
#         l = conf['layers']
        
#         print(f"\n=== Running Config: {conf_name} ===")
        
#         # Run GRU
#         print(f"Training GRU ({u} units, {l} layers)...")
#         gru_hist = train_and_evaluate(GRU_Cell, x_train, y_train, x_test, y_test, 
#                                       hidden_units=u, num_layers=l, epochs=10)
        
#         # Run MGU
#         print(f"Training MGU ({u} units, {l} layers)...")
#         mgu_hist = train_and_evaluate(MGU_Cell, x_train, y_train, x_test, y_test, 
#                                       hidden_units=u, num_layers=l, epochs=10)
        
#         results[conf_name] = {'GRU': gru_hist, 'MGU': mgu_hist}

#     return results









def run_experiment():
    data_train, data_test = load_data('notMNIST_small.mat')
    if data_train is None: return

    trials = 3 # [cite: 37, 39]
    results = {'GRU': [], 'MGU': []}
    
    print("\n--- Starting GRU Experiments ---")
    for t in range(trials):
        print(f"\nGRU Trial {t+1}/{trials}")
        hist = train_and_evaluate(GRU_Cell, *data_train, *data_test, epochs=10) # Reduced epochs for testing speed
        results['GRU'].append(hist)

    print("\n--- Starting MGU Experiments ---")
    for t in range(trials):
        print(f"\nMGU Trial {t+1}/{trials}")
        hist = train_and_evaluate(MGU_Cell, *data_train, *data_test, epochs=10)
        results['MGU'].append(hist)
        
    return results


########################### Plotting Function ###########################


def plot_comparison(results):
    if not results: return
    
    plt.figure(figsize=(14, 6))
    
    # Plot Error
    plt.subplot(1, 2, 1)
    colors = ['b', 'g', 'm', 'c'] # Colors for different configs
    
    for idx, (conf_name, metrics) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        epochs = range(1, len(metrics['GRU']['test_error']) + 1)
        
        plt.plot(epochs, metrics['GRU']['test_error'], color=c, linestyle='--', label=f'GRU - {conf_name}')
        plt.plot(epochs, metrics['MGU']['test_error'], color=c, linestyle='-',  label=f'MGU - {conf_name}')
        
    plt.title('Test Error Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    for idx, (conf_name, metrics) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        epochs = range(1, len(metrics['GRU']['loss']) + 1)
        
        plt.plot(epochs, metrics['GRU']['loss'], color=c, linestyle='--', label=f'GRU - {conf_name}')
        plt.plot(epochs, metrics['MGU']['loss'], color=c, linestyle='-',  label=f'MGU - {conf_name}')

    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()




# def plot_results(results):
#     epochs = range(1, len(results['GRU'][0]['test_error']) + 1)
    
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     for i, hist in enumerate(results['GRU']):
#         plt.plot(epochs, hist['test_error'], 'b--', label=f'GRU Trial {i+1}')
#     for i, hist in enumerate(results['MGU']):
#         plt.plot(epochs, hist['test_error'], 'r-', label=f'MGU Trial {i+1}')
    
#     plt.title('Test Classification Error over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Error Rate')
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(1, 2, 2)
#     for i, hist in enumerate(results['GRU']):
#         plt.plot(epochs, hist['loss'], 'b--', label=f'GRU Trial {i+1}')
#     for i, hist in enumerate(results['MGU']):
#         plt.plot(epochs, hist['loss'], 'r-', label=f'MGU Trial {i+1}')
        
#     plt.title('Training Loss over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    results = run_comparison()
    plot_comparison(results)
