import os
import sys
import time
import warnings
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Adding `import imageio` and `pip install kaleido` as it is a dependency.
import imageio

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data
    "TICKER": "NVDA",
    "PERIOD": "1y",
    "INTERVAL": "1d",

    # Neural Network
    "LAYER_SIZES": [5, 16, 32, 64, 32, 16, 8, 3],   # 8 layers, 176 nodes
    "LEARNING_RATE": 0.05,
    "MOMENTUM": 0.9,
    "TRAINING_EPOCHS": 200,
    "SEED": 42,

    # Loss Landscape
    "LANDSCAPE_RES": 80,
    "LANDSCAPE_RANGE": 4.0,

    # Output
    "RESOLUTION": (1920, 1080),
    "OUTPUT_FILE": "ML_Training_Visualization.png",
    "LOG_FILE": "ml_training_pipeline.log",

    # Rendering
    "MAX_EDGES_PER_LAYER": 80,
}

THEME = {
    "BG": "#0b0b0b",
    "GRID": "#1a1a1a",
    "TEXT": "#ffffff",
    "TEXT_DIM": "#888888",
    "FONT": "Roboto Mono",

    # Node colors per layer (8 layers)
    "NODE_COLORS": [
        "#ff1493",   # Deep Pink        (Input: 5 features)
        "#9932cc",   # Dark Orchid
        "#4169e1",   # Royal Blue
        "#00bfff",   # Deep Sky Blue
        "#00fa9a",   # Medium Spring Green
        "#7fff00",   # Chartreuse
        "#ffffff",   # White
        "#da70d6",   # Orchid            (Output: 3 regimes)
    ],

    # Loss landscape
    "SURFACE_COLORSCALE": "Jet",
    "PATH_COLOR": "#ff1493",
    "MARKER_COLOR": "#ffffff",

    # Edges
    "EDGE_GLOW": "rgba(0, 191, 255, 0.08)",
    "EDGE_CORE": "rgba(255, 255, 255, 0.25)",
}

# Regime labels
REGIME_NAMES = ["Bull", "Bear", "Sideways"]

# =============================================================================
# UTILS
# =============================================================================


def log(msg):
    """Centralized logger."""
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    try:
        with open(CONFIG["LOG_FILE"], "a") as f:
            f.write(formatted + "\n")
    except:
        pass

# =============================================================================
# MODULE 1: DATA
# =============================================================================


def fetch_and_engineer_features():
    """
    Fetches BTC-USD data and engineers quant features for regime classification.
    Returns: X (n, 5), y (n,) with labels {0: Bull, 1: Bear, 2: Sideways}
    """
    import yfinance as yf
    import pandas as pd

    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']})...")

    try:
        df = yf.download(CONFIG["TICKER"], period=CONFIG["PERIOD"],
                         interval=CONFIG["INTERVAL"], progress=False)
    except Exception as e:
        log(f"[Error] YF Download failed: {e}. Using synthetic fallback.")
        return _generate_synthetic_fallback()

    # Handle MultiIndex columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        log("[Data] Flattened MultiIndex columns.")

    if df.empty:
        log("[Warning] Empty dataframe. Using synthetic fallback.")
        return _generate_synthetic_fallback()

    price = df["Close"].values.flatten().astype(float)

    # Feature Engineering
    log("[Data] Engineering quant features...")

    # 1. Log Returns
    log_ret = np.diff(np.log(price + 1e-9))

    # 2. Rolling Volatility (20-day)
    window = 20
    vol = np.array([np.std(log_ret[max(0, i - window):i + 1])
                    for i in range(len(log_ret))])

    # 3. Momentum (10-day ROC)
    mom_window = 10
    momentum = np.zeros(len(log_ret))
    for i in range(mom_window, len(log_ret)):
        momentum[i] = (price[i + 1] - price[i + 1 - mom_window]) / \
            (price[i + 1 - mom_window] + 1e-9)

    # 4. RSI-like oscillator (14-day)
    rsi = np.zeros(len(log_ret))
    rsi_window = 14
    for i in range(rsi_window, len(log_ret)):
        gains = np.maximum(log_ret[i - rsi_window:i], 0)
        losses = np.maximum(-log_ret[i - rsi_window:i], 0)
        avg_gain = np.mean(gains) + 1e-9
        avg_loss = np.mean(losses) + 1e-9
        rsi[i] = avg_gain / (avg_gain + avg_loss)  # Normalized 0-1

    # 5. Mean-Reversion Score (z-score of price vs 30-day MA)
    mr_window = 30
    mean_rev = np.zeros(len(log_ret))
    for i in range(mr_window, len(log_ret)):
        window_prices = price[i + 1 - mr_window:i + 2]
        ma = np.mean(window_prices)
        std = np.std(window_prices) + 1e-9
        mean_rev[i] = (price[i + 1] - ma) / std

    # Stack features
    X = np.column_stack([log_ret, vol, momentum, rsi, mean_rev])

    # Trim warmup period
    trim = max(window, mom_window, rsi_window, mr_window)
    X = X[trim:]

    # Regime Labels
    ret_trimmed = log_ret[trim:]
    sigma = np.std(ret_trimmed)
    y = np.ones(len(ret_trimmed), dtype=int) * 2   # Default: Sideways
    y[ret_trimmed > sigma] = 0                       # Bull
    y[ret_trimmed < -sigma] = 1                      # Bear

    # Normalize features
    for col in range(X.shape[1]):
        mu = np.mean(X[:, col])
        std = np.std(X[:, col]) + 1e-9
        X[:, col] = (X[:, col] - mu) / std

    log(f"[Data] {len(X)} samples | Features: 5 | "
        f"Regimes: Bull={np.sum(y==0)}, Bear={np.sum(y==1)}, "
        f"Sideways={np.sum(y==2)}")

    return X, y


def _generate_synthetic_fallback():
    """Synthetic spiral dataset if yfinance fails."""
    log("[Data] Generating synthetic spiral fallback...")
    np.random.seed(CONFIG["SEED"])
    n_per_class = 100
    n_classes = 3
    X = np.zeros((n_per_class * n_classes, 5))
    y = np.zeros(n_per_class * n_classes, dtype=int)

    for cls in range(n_classes):
        ix = range(n_per_class * cls, n_per_class * (cls + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(cls * 4.0, (cls + 1) * 4.0, n_per_class) + \
            np.random.randn(n_per_class) * 0.15
        X[ix, 0] = r * np.sin(t)
        X[ix, 1] = r * np.cos(t)
        X[ix, 2] = np.random.randn(n_per_class) * 0.3
        X[ix, 3] = np.random.randn(n_per_class) * 0.3
        X[ix, 4] = np.random.randn(n_per_class) * 0.3
        y[ix] = cls

    return X, y

# =============================================================================
# MODULE 2: NEURAL NETWORK (Pure NumPy)
# =============================================================================


class NumpyNeuralNetwork:
    """
    Pure NumPy feedforward neural network for market regime classification.
    Architecture: [5, 16, 32, 64, 32, 16, 8, 3]
    """

    def __init__(self, layer_sizes, lr=0.05, momentum=0.9, seed=42):
        np.random.seed(seed)
        self.layers = layer_sizes
        self.lr = lr
        self.momentum = momentum
        self.weights = []
        self.biases = []
        self.vel_w = []
        self.vel_b = []

        # He initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * \
                np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
            self.vel_w.append(np.zeros_like(w))
            self.vel_b.append(np.zeros_like(b))

        # Training history
        self.history = {
            "loss": [],
            "weights_snapshots": [],
            "activations_snapshots": [],
            "param_trajectory": [],
        }

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-9)

    def forward(self, X):
        self.activations = [X]
        self.pre_acts = []
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_acts.append(z)
            a = self._relu(z)
            self.activations.append(a)
        # Output
        z = a @ self.weights[-1] + self.biases[-1]
        self.pre_acts.append(z)
        a = self._softmax(z)
        self.activations.append(a)
        return a

    def _cross_entropy(self, y_pred, y_true):
        n = y_true.shape[0]
        return -np.mean(np.log(y_pred[range(n), y_true] + 1e-9))

    def backward(self, y_true):
        n = y_true.shape[0]
        delta = self.activations[-1].copy()
        delta[range(n), y_true] -= 1
        delta /= n

        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * \
                    self._relu_deriv(self.pre_acts[i - 1])
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.lr * dw
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.lr * db
            self.weights[i] += self.vel_w[i]
            self.biases[i] += self.vel_b[i]

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = self._cross_entropy(y_pred, y)
        self.backward(y)
        return loss

    def get_weight_magnitudes(self, layer_idx):
        m = np.abs(self.weights[layer_idx])
        return m / (m.max() + 1e-9)

    def get_activation_magnitudes(self):
        return [np.mean(np.abs(a), axis=0) for a in self.activations]

    def get_param_2d(self):
        return np.array([np.mean(self.weights[0]),
                         np.mean(self.weights[-1])])


def run_training(X, y):
    """Runs full training, recording snapshots every epoch."""
    log("[Training] Initializing neural network...")
    net = NumpyNeuralNetwork(
        CONFIG["LAYER_SIZES"],
        lr=CONFIG["LEARNING_RATE"],
        momentum=CONFIG["MOMENTUM"],
        seed=CONFIG["SEED"],
    )

    epochs = CONFIG["TRAINING_EPOCHS"]
    log(f"[Training] Running {epochs} epochs...")

    for epoch in range(epochs):
        loss = net.train_step(X, y)
        net.history["loss"].append(loss)
        net.history["weights_snapshots"].append(
            [w.copy() for w in net.weights]
        )
        net.history["activations_snapshots"].append(
            net.get_activation_magnitudes()
        )
        net.history["param_trajectory"].append(net.get_param_2d())

        if (epoch + 1) % 50 == 0:
            log(f"[Training] Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    log(f"[Training] Final loss: {net.history['loss'][-1]:.4f}")
    return net

# =============================================================================
# MODULE 3: RENDERING
# =============================================================================


def compute_network_layout(layer_sizes):
    """
    Arena3D-style: nodes scattered on flat horizontal planes stacked vertically.
    Each layer is a transparent plane with nodes spread across it.
    """
    np.random.seed(42)
    positions = []
    n_layers = len(layer_sizes)

    # Vertical spacing between planes — balanced with XY range
    z_spacing = 0.9
    plane_half = 2.0  # Half-width of each plane

    for li, n_nodes in enumerate(layer_sizes):
        z_val = li * z_spacing

        if n_nodes <= 3:
            # Small layers: evenly spaced in a line
            x = np.linspace(-1.0, 1.0, n_nodes)
            y = np.zeros(n_nodes)
        elif n_nodes <= 16:
            cols = int(np.ceil(np.sqrt(n_nodes)))
            rows = int(np.ceil(n_nodes / cols))
            xg = np.linspace(-plane_half * 0.7, plane_half * 0.7, cols)
            yg = np.linspace(-plane_half * 0.7, plane_half * 0.7, rows)
            xx, yy = np.meshgrid(xg, yg)
            x = xx.flatten()[:n_nodes]
            y = yy.flatten()[:n_nodes]
        else:
            cols = int(np.ceil(np.sqrt(n_nodes)))
            rows = int(np.ceil(n_nodes / cols))
            xg = np.linspace(-plane_half * 0.85, plane_half * 0.85, cols)
            yg = np.linspace(-plane_half * 0.85, plane_half * 0.85, rows)
            xx, yy = np.meshgrid(xg, yg)
            x = xx.flatten()[:n_nodes] + np.random.randn(n_nodes) * 0.1
            y = yy.flatten()[:n_nodes] + np.random.randn(n_nodes) * 0.1

        z = np.full(n_nodes, z_val)
        positions.append((x, y, z))

    return positions


# Quant-specific layer labels for the network visualization
LAYER_LABELS = [
    "Market Features",        # Input: returns, vol, momentum, RSI, mean-rev
    "Signal Extraction",      # Hidden 1
    "Pattern Recognition",    # Hidden 2
    "Regime Detection",       # Hidden 3 (largest — 64 nodes)
    "Risk Assessment",        # Hidden 4
    "Alpha Generation",       # Hidden 5
    "Position Sizing",        # Hidden 6
    "Regime Output",          # Output: Bull / Bear / Sideways
]


def generate_loss_landscape(res, rng):
    """
    Generates a dramatic multi-modal loss surface with clear peaks and valleys.
    Matches the reference: vivid rainbow surface with visible topography.
    """
    x = np.linspace(-rng, rng, res)
    y = np.linspace(-rng, rng, res)
    X, Y = np.meshgrid(x, y)

    # Dramatic multi-modal surface: Rastrigin-like + Gaussian wells
    Z = (X**2 + Y**2) / 8.0  # Base bowl
    Z += 1.5 * (np.cos(2.0 * X) + np.cos(2.0 * Y))  # Ripples (peaks/valleys)
    Z += -3.0 * np.exp(-((X - 2.5)**2 + (Y - 2.5)**2) / 1.5)  # Deep global min
    Z += -1.5 * np.exp(-((X + 1.5)**2 + (Y - 1.0)**2) / 1.0)  # Local min
    Z += -1.0 * np.exp(-((X - 0.5)**2 + (Y + 2.0)**2) /
                       0.8)  # Another local min
    Z += 2.0 * np.exp(-((X + 2.5)**2 + (Y + 2.5)**2) / 3.0)   # High plateau

    # Normalize to nice visual range
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9) * 5.0

    return X, Y, Z


def generate_gd_path(n_points, rng):
    """Synthetic gradient descent spiral path from high loss to minimum."""
    t = np.linspace(0, 4 * np.pi, n_points)
    start = np.array([-2.0, -2.0])
    end = np.array([2.5, 2.5])

    progress = t / t.max()
    ease = progress * progress * (3 - 2 * progress)

    spiral_amp = 1.5 * (1 - ease)
    px = start[0] + (end[0] - start[0]) * ease + spiral_amp * np.sin(t * 1.5)
    py = start[1] + (end[1] - start[1]) * ease + spiral_amp * np.cos(t * 1.5)

    # Clamp to landscape bounds
    px = np.clip(px, -rng + 0.1, rng - 0.1)
    py = np.clip(py, -rng + 0.1, rng - 0.1)

    return px, py


def get_path_z(path_x, path_y, lx, ly, lz):
    """Interpolate Z values on the loss surface for the GD path."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (lx[0, :], ly[:, 0]), lz.T,
        method="linear", bounds_error=False, fill_value=float(lz.min())
    )
    pts = np.column_stack([path_x, path_y])
    return interp(pts)


def _build_edge_traces(positions, layer_idx, weight_snap, max_edges):
    """
    Builds edge line coordinates between adjacent layers.
    Filters to top-N edges by weight magnitude.
    """
    pos_from = positions[layer_idx]
    pos_to = positions[layer_idx + 1]
    n_from, n_to = len(pos_from[0]), len(pos_to[0])

    if weight_snap is not None:
        w = np.abs(weight_snap)
        w_norm = w / (w.max() + 1e-9)
        flat = w_norm.flatten()
        if len(flat) > max_edges:
            thresh = np.sort(flat)[-max_edges]
        else:
            thresh = 0.0
    else:
        w_norm = None
        thresh = 0.0

    xe, ye, ze = [], [], []
    for i in range(n_from):
        for j in range(n_to):
            if w_norm is not None and w_norm[i, j] < thresh:
                continue
            xe.extend([pos_from[0][i], pos_to[0][j], None])
            ye.extend([pos_from[1][i], pos_to[1][j], None])
            ze.extend([pos_from[2][i], pos_to[2][j], None])

    return xe, ye, ze


# --------------- STATIC IMAGE RENDERER ---------------

def render_training_animation(data):
    """Creates an animated training video showing optimization progress."""

    frames = []
    tmp_dir = "frames"

    # Check if tmp_dir exists and remove it if it does to prevent errors
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir, exist_ok=True)

    log("[Render] Generating training animation frames...")

    for epoch in range(data["n_epochs"]):

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            horizontal_spacing=0.02,
        )

        # ==============================
        # LEFT PANEL (Network)
        # ==============================

        act_mags = data["activations"][epoch]

        for li in range(len(CONFIG["LAYER_SIZES"])):

            pos = data["positions"][li]
            color = THEME["NODE_COLORS"][li]

            if li < len(act_mags):
                a = np.clip(act_mags[li], 0, 1)
                sizes = 6 + 10 * a
            else:
                sizes = np.full(len(pos[0]), 8)

            fig.add_trace(
                go.Scatter3d(
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=color,
                        opacity=0.9
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )

        # ==============================
        # RIGHT PANEL (Loss Landscape)
        # ==============================

        fig.add_trace(
            go.Surface(
                x=data["landscape_X"],
                y=data["landscape_Y"],
                z=data["landscape_Z"],
                colorscale="Jet",
                opacity=0.8,
                showscale=False
            ),
            row=1, col=2
        )

        # Gradient path until this epoch
        fig.add_trace(
            go.Scatter3d(
                x=data["path_x"][:epoch+1],
                y=data["path_y"][:epoch+1],
                z=data["path_z"][:epoch+1] + 0.1,
                mode="lines",
                line=dict(color="red", width=6),
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(
            width=1920,
            height=1080,
            paper_bgcolor=THEME["BG"],
            plot_bgcolor=THEME["BG"],
            margin=dict(l=0, r=0, t=0, b=0),
        )

        frame_path = f"{tmp_dir}/frame_{epoch:04d}.png"
        fig.write_image(frame_path)
        frames.append(imageio.imread(frame_path))

        if epoch % 20 == 0:
            log(f"[Render] Frame {epoch}/{data['n_epochs']}")

    log("[Render] Creating video...")

    imageio.mimsave("training_animation.gif", frames, fps=20)

    log("[Success] Animation saved: training_animation.gif")

# =============================================================================
# MAIN
# =============================================================================


def main():
    # Install kaleido for Plotly image export. Also, install yfinance for data fetching.
    #%pip install kaleido yfinance imageio

    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    log("=" * 60)
    log("  ML TRAINING SIMULATION PIPELINE")
    log("  Market Regime Classification (Bull / Bear / Sideways)")
    log("=" * 60)

    # 1. Data
    X, y = fetch_and_engineer_features()

    # 2. Training
    net = run_training(X, y)

    # 3. Loss Landscape
    log("[Landscape] Generating 3D loss surface...")
    lx, ly, lz = generate_loss_landscape(
        CONFIG["LANDSCAPE_RES"], CONFIG["LANDSCAPE_RANGE"]
    )

    # 4. Gradient Path
    log("[Landscape] Computing gradient descent path...")
    path_x, path_y = generate_gd_path(
        CONFIG["TRAINING_EPOCHS"], CONFIG["LANDSCAPE_RANGE"]
    )
    path_z = get_path_z(path_x, path_y, lx, ly, lz)

    # 5. Network Layout
    positions = compute_network_layout(CONFIG["LAYER_SIZES"])

    # 6. Pack data for renderer
    render_data = {
        "positions": positions,
        "weights": net.history["weights_snapshots"],
        "activations": net.history["activations_snapshots"],
        "losses": net.history["loss"],
        "n_epochs": CONFIG["TRAINING_EPOCHS"],
        "landscape_X": lx,
        "landscape_Y": ly,
        "landscape_Z": lz,
        "path_x": path_x,
        "path_y": path_y,
        "path_z": path_z,
    }

    # 7. Render training animation
    render_training_animation(render_data)

    log("=" * 60)
    log("  PIPELINE FINISHED")
    log("=" * 60)


if __name__ == "__main__":
    main()