# ==============================================================================
# CNN-LSTM Model for Signal Classification
# ==============================================================================
# Architecture:
#   - Input 1: Waveform signals (batches of 5 time steps, 3000 points each)
#   - Input 2: Time differences (Δt) between consecutive signals
#   - Feature extractor submodel (Conv2D + Dense) applied via TimeDistributed
#   - LSTM layer (stateful) for temporal sequence processing
#   - Dense layers for final binary classification (sigmoid output)
#
# Raw input format:
#   - signals: list of N numpy arrays, each of shape (3000,)
#   - times:   list of N floats (occurrence time of each signal)
# ==============================================================================

import numpy as np
import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense,
    Masking, TimeDistributed, Concatenate, LSTM
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SIGNAL_LENGTH = 3000      # Number of data points per signal
TIME_STEPS = 5            # Number of signals per sequence (window size)
BATCH_SIZE = 10           # Fixed batch size (required for stateful LSTM)
MASK_VALUE = -10          # Padding value for incomplete sequences
T_THRESHOLD = 0.70        # Time threshold fraction for labeling (70% of T_max)
CUTOFF_SECONDS = 120      # Omit the last N seconds of signals per experiment
HDF5_FILE = "data.hdf5"   # Path to the HDF5 file
MAX_GROUPS = None         # Max number of groups to load from HDF5 (None = all)
TRAIN_GROUPS = 15         # Number of groups used for training (rest = testing)


# ==============================================================================
# 1. HDF5 Data Loading
# ==============================================================================
# Reads experiments from an HDF5 file. The expected structure is:
#
# file.hdf5
# ├── Group_1/                        ← experiment group
# │   ├── attrs["flashover_status"]   ← "Hay Flashover" to include
# │   └── Signals/                    ← subgroup containing signal datasets
# │       ├── signal_0001             ← dataset, shape (3000,)
# │       │   └── attrs["timestamp_s"]← occurrence time (absolute seconds)
# │       ├── signal_0002
# │       │   └── attrs["timestamp_s"]
# │       └── ...
# ├── Group_2/
# │   ├── attrs["flashover_status"]
# │   └── Signals/
# │       └── ...
# └── ...

def load_from_hdf5(filepath=HDF5_FILE, max_groups=MAX_GROUPS):
    """
    Load experiment data from an HDF5 file.

    Only groups with flashover_status == 'Hay Flashover' are loaded.
    Timestamps are shifted so each experiment starts at t=0.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    max_groups : int or None
        Maximum number of groups to load. None = load all valid groups.

    Returns
    -------
    experiments : list of dict
        Each dict has:
        - "name":    str, group name in the HDF5 file
        - "signals": list of np.ndarray, each shape (SIGNAL_LENGTH,)
        - "times":   list of float, occurrence times referenced to 0
        - "t_max":   float, total experiment duration (last_time - first_time)
    """

    experiments = []

    with h5py.File(filepath, "r") as f:
        # Iterate over top-level groups in the file
        group_names = list(f.keys())
        print(f"HDF5 file: {filepath}")
        print(f"  Total groups found: {len(group_names)}")

        loaded = 0
        skipped = 0

        for group_name in group_names:
            group = f[group_name]

            # --- Check flashover_status attribute ---
            flashover_status = group.attrs.get("flashover_status", "")
            if flashover_status != "Hay Flashover":
                skipped += 1
                continue

            # --- Stop if max_groups limit reached ---
            if max_groups is not None and loaded >= max_groups:
                break

            # --- Access the Signals subgroup ---
            if "Signals" not in group:
                print(f"  WARNING: Group '{group_name}' has no 'Signals' subgroup, skipping.")
                skipped += 1
                continue

            signals_group = group["Signals"]

            signals = []
            times = []

            # --- Read each signal dataset and its timestamp ---
            for dataset_name in signals_group:
                dataset = signals_group[dataset_name]
                signal_data = dataset[:]  # Read the signal array

                # Read the timestamp from the dataset's attributes
                timestamp = dataset.attrs.get("timestamp_s", None)
                if timestamp is None:
                    print(f"  WARNING: '{group_name}/Signals/{dataset_name}' "
                          f"has no 'timestamp_s' attribute, skipping signal.")
                    continue

                signals.append(signal_data.astype(np.float32))
                times.append(float(timestamp))

            n_original = len(signals)

            if len(signals) == 0:
                print(f"  WARNING: Group '{group_name}' has no valid signals, skipping.")
                skipped += 1
                continue

            # --- Shift timestamps so the experiment starts at t=0 ---
            t_min = min(times)
            times = [t - t_min for t in times]
            t_max = max(times)

            # --- Remove signals in the last CUTOFF_SECONDS ---
            # Signals near the very end of the experiment (sustained flashover)
            # are excluded because they are not useful for prediction.
            if CUTOFF_SECONDS > 0 and t_max > CUTOFF_SECONDS:
                cutoff_time = t_max - CUTOFF_SECONDS
                filtered = [
                    (s, t) for s, t in zip(signals, times)
                    if t <= cutoff_time
                ]
                if len(filtered) == 0:
                    print(f"  WARNING: Group '{group_name}' has no signals "
                          f"after removing last {CUTOFF_SECONDS}s, skipping.")
                    skipped += 1
                    continue
                signals, times = zip(*filtered)
                signals = list(signals)
                times = list(times)
                # Recalculate t_max based on the remaining signals
                t_max = max(times)
                print(f"  [{group_name}]: removed {n_original - len(signals)} signals "
                      f"in last {CUTOFF_SECONDS}s (kept {len(signals)}, "
                      f"new T_max={t_max:.1f}s)")
            else:
                n_original = len(signals)

            experiments.append({
                "name": group_name,
                "signals": signals,
                "times": times,
                "t_max": t_max
            })

            loaded += 1

    print(f"  Groups loaded:  {loaded} (flashover_status = 'Hay Flashover')")
    print(f"  Groups skipped: {skipped}")
    if max_groups is not None:
        print(f"  Max groups limit: {max_groups}")

    return experiments


# ==============================================================================
# 2. Data Preprocessing
# ==============================================================================
# Takes the raw input (two lists) and converts it into the batched tensors
# that the model expects, including Δt computation and padding.

def generate_labels(times, t_max, t_threshold=T_THRESHOLD, time_steps=TIME_STEPS):
    """
    Generate binary labels for each sequence based on a time threshold.

    The labeling follows the criterion:
        y = 1  if  t >= T_max * t_threshold   (high danger of flashover)
        y = 0  otherwise                      (low danger)

    All experiments ended with sustained flashover, so signals near the end
    are labeled as high danger and signals at the beginning as low danger.
    The threshold (default 70%) defines the transition point.

    Parameters
    ----------
    times : list of float
        Sorted occurrence times of all signals in the experiment.
    t_max : float
        Total experiment duration (seconds).
    t_threshold : float
        Fraction of t_max to define the class boundary. Default: 0.70.
    time_steps : int
        Number of signals per group. Default: 5.

    Returns
    -------
    labels : list of int
        One label (0 or 1) per group of `time_steps` signals.
    """
    threshold_time = t_max * t_threshold
    labels = []

    for start in range(0, len(times) - time_steps + 1, time_steps):
        # Use the time of the last signal in the group as reference
        group_time = times[start + time_steps - 1]

        if group_time >= threshold_time:
            labels.append(1)  # High danger of flashover
        else:
            labels.append(0)  # Low danger

    return labels


def preprocess_data(signals, times, t_max,
                    t_threshold=T_THRESHOLD,
                    time_steps=TIME_STEPS,
                    signal_length=SIGNAL_LENGTH,
                    batch_size=BATCH_SIZE,
                    mask_value=MASK_VALUE):
    """
    Convert raw signal lists into model-ready batched arrays.
    Labels are generated automatically using the time threshold criterion.

    Parameters
    ----------
    signals : list of np.ndarray
        List of N numpy arrays, each of shape (signal_length,).
        Example: [np.array([...3000 pts...]), np.array([...3000 pts...]), ...]
    times : list of float
        List of N occurrence times (one per signal), in the same order.
        Example: [0.0, 0.0012, 0.0031, 0.0045, ...]
    t_max : float
        Total experiment duration (seconds).
    t_threshold : float
        Fraction of t_max to define the class boundary. Default: 0.70.
    time_steps : int
        Number of signals per sequence (window size). Default: 5.
    signal_length : int
        Number of data points per signal. Default: 3000.
    batch_size : int
        Fixed batch size for stateful LSTM. Default: 10.
    mask_value : float
        Value used to pad incomplete sequences. Default: -10.

    Returns
    -------
    X_signals : np.ndarray, shape (N_padded, time_steps, 1, signal_length, 1)
        Batched signal sequences, padded to be a multiple of batch_size.
    X_dt : np.ndarray, shape (N_padded, time_steps, 1)
        Batched Δt sequences, padded to be a multiple of batch_size.
    y : np.ndarray, shape (N_padded, 1)
        Padded labels (padding labels are set to 0).
    n_real : int
        Number of real (non-padded) sequences.
    """

    N = len(signals)
    assert len(times) == N, "signals and times must have the same length"

    # --- Sort signals by occurrence time ---
    sorted_indices = np.argsort(times)
    signals = [signals[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]

    # --- Build sequences using a sliding window of size `time_steps` ---
    seq_signals = []   # Each element: list of `time_steps` signal arrays
    seq_dt = []        # Each element: list of `time_steps` Δt values

    for start in range(0, N - time_steps + 1, time_steps):
        window_signals = []
        window_dt = []

        for j in range(time_steps):
            idx = start + j
            sig = signals[idx]

            # Ensure each signal has the correct length
            assert len(sig) == signal_length, (
                f"Signal {idx} has length {len(sig)}, expected {signal_length}"
            )
            window_signals.append(sig)

            # Compute Δt: time difference to the previous signal
            # First signal in the window gets Δt = 0
            if j == 0:
                window_dt.append(0.0)
            else:
                dt = times[start + j] - times[start + j - 1]
                window_dt.append(dt)

        seq_signals.append(window_signals)
        seq_dt.append(window_dt)

    n_sequences = len(seq_signals)
    n_real = n_sequences

    # --- Generate labels automatically from time threshold ---
    labels = generate_labels(times, t_max, t_threshold, time_steps)
    labels = np.array(labels).flatten()

    # --- Pad to make the number of sequences a multiple of batch_size ---
    remainder = n_sequences % batch_size
    n_pad = (batch_size - remainder) % batch_size
    n_padded = n_sequences + n_pad

    # --- Build numpy arrays ---
    # Signals: (n_padded, time_steps, 1, signal_length, 1)
    X_signals = np.full(
        (n_padded, time_steps, 1, signal_length, 1),
        fill_value=mask_value,
        dtype=np.float32
    )

    # Δt: (n_padded, time_steps, 1)
    X_dt = np.full(
        (n_padded, time_steps, 1),
        fill_value=mask_value,
        dtype=np.float32
    )

    # Labels: (n_padded, 1)
    y = np.zeros((n_padded, 1), dtype=np.float32)

    # Fill real data into the arrays
    for i in range(n_sequences):
        for j in range(time_steps):
            X_signals[i, j, 0, :, 0] = seq_signals[i][j]
            X_dt[i, j, 0] = seq_dt[i][j]
        y[i, 0] = labels[i]

    # --- Class distribution ---
    n_class_0 = int(np.sum(y[:n_sequences] == 0))
    n_class_1 = int(np.sum(y[:n_sequences] == 1))

    print(f"Preprocessing complete:")
    print(f"  Total signals:     {N}")
    print(f"  Sequences created: {n_sequences} (window={time_steps}, stride={time_steps})")
    print(f"  Threshold:         {t_threshold*100:.0f}% of T_max ({t_max:.1f}s) = {t_max*t_threshold:.1f}s")
    print(f"  Class 0 (low):     {n_class_0} ({n_class_0/n_sequences*100:.1f}%)")
    print(f"  Class 1 (high):    {n_class_1} ({n_class_1/n_sequences*100:.1f}%)")
    print(f"  Padded to:         {n_padded} (batch_size={batch_size})")
    print(f"  X_signals shape:   {X_signals.shape}")
    print(f"  X_dt shape:        {X_dt.shape}")
    print(f"  y shape:           {y.shape}")

    return X_signals, X_dt, y, n_real


# ==============================================================================
# 3. Feature Extractor Submodel
# ==============================================================================
# Processes each individual signal (1 time step).
# Extracts spatial features from the raw waveform using Conv2D,
# followed by pooling, dropout, flattening, and a dense layer.
# Input shape per signal: (1, 3000, 1) → Output shape: (20,)

def build_feature_extractor(input_shape=(1, SIGNAL_LENGTH, 1)):
    """Build the CNN-based feature extractor for a single waveform signal."""

    inp = Input(shape=input_shape, name="feature_extractor_input")

    # Convolutional layer: 5 filters, kernel size (1, 100), ReLU activation
    x = Conv2D(
        filters=5,
        kernel_size=(1, 100),
        activation="relu",
        name="conv2d"
    )(inp)

    # Max pooling: pool size (1, 10) — reduces temporal dimension
    x = MaxPooling2D(
        pool_size=(1, 10),
        name="max_pool2d"
    )(x)

    # Dropout: 50% for regularization
    x = Dropout(0.5, name="dropout_cnn")(x)

    # Flatten: convert 2D feature maps to 1D vector
    x = Flatten(name="flatten")(x)

    # Dense layer: 20 units with ReLU activation
    x = Dense(
        units=20,
        activation="relu",
        name="dense_features"
    )(x)

    # Dropout: 50% after dense layer
    x = Dropout(0.5, name="dropout_dense")(x)

    model = Model(inputs=inp, outputs=x, name="feature_extractor")
    return model


# ==============================================================================
# 4. Complete Model
# ==============================================================================
# Combines the feature extractor (applied via TimeDistributed to each of the
# 5 time steps), concatenates with Δt, and passes through LSTM + Dense layers.

def build_complete_model(batch_size=BATCH_SIZE,
                         time_steps=TIME_STEPS,
                         signal_length=SIGNAL_LENGTH):
    """Build the full CNN-LSTM model for signal classification."""

    # --- Input 1: Waveform signals ---
    # Shape: (batch_size, time_steps, 1, signal_length, 1)
    input_signals = Input(
        batch_shape=(batch_size, time_steps, 1, signal_length, 1),
        name="input_signals"
    )

    # --- Input 2: Time differences (Δt) between signals ---
    # Shape: (batch_size, time_steps, 1)
    input_dt = Input(
        batch_shape=(batch_size, time_steps, 1),
        name="input_delta_t"
    )

    # --- Masking layer (only on Δt) ---
    # Masking is applied only to the Δt input (3D tensor).
    # Applying Masking to the 5D signal tensor causes shape conflicts with
    # TimeDistributed + Concatenate. The Δt mask is sufficient: the LSTM
    # will ignore padded timesteps based on the Δt mask propagation.
    masked_dt = Masking(
        mask_value=MASK_VALUE,
        name="masking_delta_t"
    )(input_dt)

    # --- Feature extraction via TimeDistributed ---
    # Apply the CNN feature extractor independently to each time step
    feature_extractor = build_feature_extractor(
        input_shape=(1, signal_length, 1)
    )

    # Output shape: (batch_size, time_steps, 20)
    features = TimeDistributed(
        feature_extractor,
        name="time_distributed_features"
    )(input_signals)

    # --- Concatenation ---
    # Merge features (20 dims) with Δt (1 dim) → (batch_size, 5, 21)
    concatenated = Concatenate(
        axis=-1,
        name="concatenate"
    )([features, masked_dt])

    # --- LSTM layer ---
    # Stateful: maintains hidden state across batches
    # return_sequences=False: output only the last time step
    x = LSTM(
        units=120,
        return_sequences=False,
        stateful=True,
        name="lstm"
    )(concatenated)

    # --- Dense layers for classification ---
    x = Dense(units=500, activation="relu", name="dense_500")(x)
    x = Dense(units=100, activation="relu", name="dense_100")(x)

    # Dropout: 50% regularization before output
    x = Dropout(0.5, name="dropout_output")(x)

    # --- Output layer ---
    # Single unit with sigmoid for binary classification
    output = Dense(units=1, activation="sigmoid", name="output")(x)

    # --- Build and compile ---
    model = Model(
        inputs=[input_signals, input_dt],
        outputs=output,
        name="cnn_lstm_signal_classifier"
    )

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==============================================================================
# 5. Training Loop (Stateful LSTM)
# ==============================================================================

def train_model(model, X_signals, X_dt, y, n_real, epochs=20):
    """
    Train the stateful LSTM model.

    Parameters
    ----------
    model : keras.Model
        Compiled model from build_complete_model().
    X_signals, X_dt, y : np.ndarray
        Preprocessed arrays from preprocess_data().
    n_real : int
        Number of real (non-padded) sequences.
    epochs : int
        Number of training epochs.
    """

    # Get the LSTM layer by name to reset its states
    lstm_layer = model.get_layer("lstm")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train on all data (including padding, which is masked)
        history = model.fit(
            [X_signals, X_dt], y,
            batch_size=BATCH_SIZE,
            epochs=1,
            shuffle=False,     # Required for stateful LSTM
            verbose=1
        )

        # Reset LSTM hidden states after each epoch
        lstm_layer.reset_states()

    return model


# ==============================================================================
# 6. Main
# ==============================================================================

def build_arrays(experiments_list):
    """
    Convert a list of experiment dicts into batched numpy arrays.

    Parameters
    ----------
    experiments_list : list of dict
        Each dict has "signals", "times", "t_max".

    Returns
    -------
    X_signals, X_dt, y : np.ndarray
        Batched arrays ready for the model (padded to batch_size multiple).
    n_real : int
        Number of real (non-padded) sequences.
    """
    all_signals = []
    all_times = []
    all_labels = []

    for i, exp in enumerate(experiments_list):
        sorted_idx = np.argsort(exp["times"])
        sorted_times = [exp["times"][j] for j in sorted_idx]
        sorted_signals = [exp["signals"][j] for j in sorted_idx]

        labels = generate_labels(sorted_times, exp["t_max"], T_THRESHOLD)

        n_usable = len(labels) * TIME_STEPS
        all_signals.extend(sorted_signals[:n_usable])
        all_times.extend(sorted_times[:n_usable])
        all_labels.extend(labels)

        n0 = labels.count(0)
        n1 = labels.count(1)
        print(f"  [{exp['name']}]: {len(exp['signals'])} signals, "
              f"T_max={exp['t_max']:.1f}s, "
              f"{len(labels)} groups (class 0: {n0}, class 1: {n1})")

    N = len(all_signals)
    n_sequences = len(all_labels)
    remainder = n_sequences % BATCH_SIZE
    n_pad = (BATCH_SIZE - remainder) % BATCH_SIZE
    n_padded = n_sequences + n_pad

    X_signals = np.full(
        (n_padded, TIME_STEPS, 1, SIGNAL_LENGTH, 1),
        fill_value=MASK_VALUE, dtype=np.float32
    )
    X_dt = np.full(
        (n_padded, TIME_STEPS, 1),
        fill_value=MASK_VALUE, dtype=np.float32
    )
    y = np.zeros((n_padded, 1), dtype=np.float32)

    for i in range(n_sequences):
        for j in range(TIME_STEPS):
            idx = i * TIME_STEPS + j
            X_signals[i, j, 0, :, 0] = all_signals[idx]
            if j == 0:
                X_dt[i, j, 0] = 0.0
            else:
                X_dt[i, j, 0] = all_times[idx] - all_times[idx - 1]
        y[i, 0] = all_labels[i]

    n_real = n_sequences
    n0 = int(np.sum(y[:n_real] == 0))
    n1 = int(np.sum(y[:n_real] == 1))
    print(f"  Total: {n_real} sequences, "
          f"class 0: {n0} ({n0/n_real*100:.1f}%), "
          f"class 1: {n1} ({n1/n_real*100:.1f}%), "
          f"padded to: {n_padded}")

    return X_signals, X_dt, y, n_real


# ==============================================================================
# 7. Prediction & Plotting
# ==============================================================================

def predict_and_plot(model, X_signals, X_dt, y_true, n_real,
                     experiments_list, title_prefix=""):
    """
    Run predictions on a dataset and generate plots.

    The model outputs a probability (0 to 1) via the sigmoid activation.
    Values close to 1 = high danger of flashover, close to 0 = low danger.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    X_signals, X_dt, y_true : np.ndarray
        Input arrays and true labels.
    n_real : int
        Number of real (non-padded) sequences.
    experiments_list : list of dict
        Original experiment dicts (for per-experiment plotting).
    title_prefix : str
        Prefix for plot titles (e.g., "Train" or "Test").
    """

    # --- Get predictions ---
    # Model output: probability (sigmoid), shape (n_padded, 1)
    y_pred_prob = model.predict(
        [X_signals, X_dt],
        batch_size=BATCH_SIZE
    ).flatten()[:n_real]

    y_true_flat = y_true.flatten()[:n_real]
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # --- Classification report ---
    print(f"\n{'='*60}")
    print(f"{title_prefix} Classification Report")
    print(f"{'='*60}")
    print(classification_report(
        y_true_flat, y_pred_class,
        target_names=["Low danger (0)", "High danger (1)"]
    ))

    # --- Plot 1: Prediction probability over all sequences ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_pred_prob, label="Predicted probability", alpha=0.7, linewidth=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", label="Decision threshold (0.5)")
    ax.fill_between(
        range(n_real), 0, 1,
        where=(y_true_flat == 1),
        alpha=0.15, color="red", label="True class 1 (high danger)"
    )
    ax.set_xlabel("Sequence index")
    ax.set_ylabel("Predicted probability")
    ax.set_title(f"{title_prefix} — Model output over all sequences")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Confusion matrix ---
    cm = confusion_matrix(y_true_flat, y_pred_class)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low (0)", "High (1)"])
    ax.set_yticklabels(["Low (0)", "High (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_prefix} — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Per-experiment prediction timeline ---
    # Shows the model output for each experiment separately
    n_experiments = len(experiments_list)
    n_cols = min(3, n_experiments)
    n_rows = (n_experiments + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_experiments == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    seq_offset = 0
    for idx, exp in enumerate(experiments_list):
        n_signals_exp = len(exp["signals"])
        n_seq_exp = n_signals_exp // TIME_STEPS

        if seq_offset + n_seq_exp > n_real:
            n_seq_exp = n_real - seq_offset

        probs = y_pred_prob[seq_offset:seq_offset + n_seq_exp]
        trues = y_true_flat[seq_offset:seq_offset + n_seq_exp]

        ax = axes[idx]
        ax.plot(probs, label="Predicted prob.", linewidth=1.0)
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8)
        ax.fill_between(
            range(n_seq_exp), 0, 1,
            where=(trues == 1),
            alpha=0.15, color="red", label="True class 1"
        )
        ax.set_title(f"{exp['name']} (T_max={exp['t_max']:.0f}s)")
        ax.set_xlabel("Sequence index")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=8)

        seq_offset += n_seq_exp

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{title_prefix} — Per-experiment predictions", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- Plot 4: Histogram of predicted probabilities ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_pred_prob[y_true_flat == 0], bins=50, alpha=0.6,
            label="True class 0 (low)", color="blue")
    ax.hist(y_pred_prob[y_true_flat == 1], bins=50, alpha=0.6,
            label="True class 1 (high)", color="red")
    ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold 0.5")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} — Distribution of predictions")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return y_pred_prob, y_pred_class


# ==============================================================================
# 8. Main
# ==============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Load experiments from HDF5 file
    # ------------------------------------------------------------------
    experiments = load_from_hdf5(filepath=HDF5_FILE, max_groups=MAX_GROUPS)

    if len(experiments) == 0:
        print("ERROR: No valid experiments found. Check the HDF5 file.")
        exit(1)

    # ------------------------------------------------------------------
    # Split into train and test sets by experiment groups
    # ------------------------------------------------------------------
    assert len(experiments) > TRAIN_GROUPS, (
        f"Not enough experiments ({len(experiments)}) for "
        f"TRAIN_GROUPS={TRAIN_GROUPS}. Need at least {TRAIN_GROUPS + 1}."
    )

    train_experiments = experiments[:TRAIN_GROUPS]
    test_experiments = experiments[TRAIN_GROUPS:]

    print(f"\n{'='*60}")
    print(f"Train/Test split: {len(train_experiments)} train, "
          f"{len(test_experiments)} test")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Build training arrays
    # ------------------------------------------------------------------
    print(f"\n--- Training data ({len(train_experiments)} experiments) ---")
    X_train_sig, X_train_dt, y_train, n_train = build_arrays(train_experiments)

    # ------------------------------------------------------------------
    # Build testing arrays
    # ------------------------------------------------------------------
    print(f"\n--- Testing data ({len(test_experiments)} experiments) ---")
    X_test_sig, X_test_dt, y_test, n_test = build_arrays(test_experiments)

    # ------------------------------------------------------------------
    # Build and train model
    # ------------------------------------------------------------------
    model = build_complete_model()
    model.summary()

    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")
    model = train_model(model, X_train_sig, X_train_dt, y_train, n_train, epochs=20)

    # ------------------------------------------------------------------
    # Evaluate and plot — Training set
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating on TRAINING set...")
    print(f"{'='*60}")
    lstm_layer = model.get_layer("lstm")
    lstm_layer.reset_states()
    y_train_prob, y_train_pred = predict_and_plot(
        model, X_train_sig, X_train_dt, y_train, n_train,
        train_experiments, title_prefix="TRAIN"
    )

    # ------------------------------------------------------------------
    # Evaluate and plot — Testing set
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating on TESTING set...")
    print(f"{'='*60}")
    lstm_layer.reset_states()
    y_test_prob, y_test_pred = predict_and_plot(
        model, X_test_sig, X_test_dt, y_test, n_test,
        test_experiments, title_prefix="TEST"
    )