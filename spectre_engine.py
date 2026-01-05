import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import optax
import matplotlib.pyplot as plt
import time
import os

# --- 0. HARDWARE CONFIGURATION ---
# Prevent JAX from monopolizing GPU memory (Critical for Edge/Laptop stability)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- 1. MISSION PARAMETERS (SWARM MODE / STRESS TEST) ---
CONF = {
    "SAMPLE_RATE": 500e6,      # 500 MHz (Wideband / High Fidelity)
    "DURATION": 20e-6,         # 20 microseconds (Longer Pulse)
    "CENTER_FREQ": 2.4e9,      # 2.4 GHz
    "JAMMER_POWER": 50.0,      # High-Power Threat
    "LEARNING_RATE": 0.01,     
    "ITERATIONS": 200,         # Optimization Steps
    "RAPP_SMOOTHNESS": 2.0,    # SSPA Smoothness factor (p)
    "RAPP_VSAT": 1.0           # Saturation Voltage
}

N_SAMPLES = int(CONF["SAMPLE_RATE"] * CONF["DURATION"])
BATCH_SIZE = 128               # Simulating 128 parallel threat scenarios (Swarm Defense)

print(f"💀 SPECTRE ENGINE ONLINE | Device: {jax.devices()[0]}")
print(f"   Mode: SWARM DEFENSE | Bandwidth: 500MHz | Batch: {BATCH_SIZE}")

# --- 2. PHYSICS ENGINE (Rapp Model & Channel) ---

@jit
def rapp_model(signal, saturation_voltage=1.0, smoothness_factor=2.0):
    """
    Rapp Model for Solid State Power Amplifiers (SSPA).
    Preserves Phase (AM/PM = 0) while compressing Amplitude (AM/AM).
    Critical for real-world hardware deployment.
    """
    magnitude = jnp.abs(signal)
    ratio = magnitude / (saturation_voltage + 1e-9)
    # The Rapp formula: v_out = v_in / (1 + |v_in/v_sat|^(2p))^(1/2p)
    denominator = jnp.power(1 + jnp.power(ratio, 2 * smoothness_factor), 1 / (2 * smoothness_factor))
    return signal / denominator

def generate_chirp_jammer(t, sweep_bw=20e6):
    """Generates a sweeping 'Chirp' Jammer signal."""
    f_start = CONF["CENTER_FREQ"] - sweep_bw/2
    k = sweep_bw / CONF["DURATION"] 
    inst_phase = 2 * jnp.pi * (f_start * t + 0.5 * k * t**2)
    return jnp.sqrt(CONF["JAMMER_POWER"]) * jnp.exp(1j * inst_phase)

# --- 3. COGNITIVE MODULATOR ---

class SpectreModulator:
    def __init__(self, n_samples):
        self.n_samples = n_samples
    
    def init_params(self, key):
        # Initialize with low-power random noise (LPI mode)
        return jax.random.normal(key, (self.n_samples, 2)) * 0.01

    def encode(self, params):
        # 1. Map to Complex IQ
        iq = params[:, 0] + 1j * params[:, 1]
        # 2. Apply Hardware Constraints (Rapp PA Model)
        return rapp_model(iq, CONF["RAPP_VSAT"], CONF["RAPP_SMOOTHNESS"])

# --- 4. DIFFERENTIABLE OBJECTIVES ---

@jit
def simulate_channel(tx_waveform, t_axis, jammer_offset):
    # Stochastic Jammer: We don't know exactly when it starts
    j_sig = generate_chirp_jammer(t_axis + jammer_offset)
    # RX = TX + Jammer
    return tx_waveform + j_sig, j_sig

def loss_fn(params, jammer_offsets, t_axis, target_key):
    """
    Non-holomorphic loss function optimized via Wirtinger Calculus.
    """
    tx_wave = SpectreModulator(N_SAMPLES).encode(params)
    
    def single_shot_loss(j_off):
        rx, jam = simulate_channel(tx_wave, t_axis, j_off)
        
        # Objective A: Maximize Correlation with Target Key (Comms Link)
        sig_score = jnp.abs(jnp.vdot(tx_wave, target_key))**2
        
        # Objective B: Minimize Spectral Overlap with Jammer (Anti-Jam)
        # We use Parseval's theorem (energy conservation) implicitly
        jam_energy = jnp.abs(jnp.vdot(tx_wave, jam))**2
        
        return -sig_score + 0.8 * jam_energy

    # Batch optimize over multiple jammer timing offsets (Robustness)
    return jnp.mean(vmap(single_shot_loss)(jammer_offsets))

# --- 5. EXECUTION & BENCHMARKING ---

def run_mission():
    key = jax.random.PRNGKey(42) # Deterministic Seed
    t_axis = jnp.linspace(0, CONF["DURATION"], N_SAMPLES)
    
    # The "Handshake" Key (Receiver expects this)
    target_key = jax.random.normal(key, (N_SAMPLES,)) + 1j * jax.random.normal(key, (N_SAMPLES,))
    target_key = target_key / jnp.linalg.norm(target_key)

    # Init
    mod = SpectreModulator(N_SAMPLES)
    params = mod.init_params(key)

    # OPTIMIZER: SplitRealAndImaginary for Complex Stability
    # This addresses the Adam variance problem for complex numbers
    base_opt = optax.adam(CONF["LEARNING_RATE"])
    optimizer = optax.contrib.split_real_and_imaginary(base_opt) 
    opt_state = optimizer.init(params)

    @jit
    def update_step(params, opt_state, j_offsets):
        val, grads = value_and_grad(loss_fn)(params, j_offsets, t_axis, target_key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val

    # Warm-up (Compiling XLA)
    # We must warm up with the SAME batch size (128) to ensure the kernel is compiled for the loop
    print("   ...Compiling JAX Kernels (Warm-up)...")
    warmup_params, warmup_state, warmup_loss = update_step(params, opt_state, jnp.zeros((BATCH_SIZE,)))
    warmup_loss.block_until_ready()

    print("   ...Engaging Adaptive Nulling Loop...")
    start_time = time.perf_counter()
    loss_history = []
    
    for i in range(CONF["ITERATIONS"]):
        key, subkey = jax.random.split(key)
        # Simulate 128 parallel timelines (Swarm Mode)
        j_offsets = jax.random.normal(subkey, (BATCH_SIZE,)) * 1e-9 
        
        params, opt_state, loss = update_step(params, opt_state, j_offsets)
        
        if i % 20 == 0:
            loss.block_until_ready() # Sync for print
            print(f"   [Step {i}] Jammer Correlation: {loss:.4f}")
        
        loss_history.append(loss)

    end_time = time.perf_counter()
    avg_latency = ((end_time - start_time) / CONF["ITERATIONS"]) * 1000
    print(f"\n⚡ MISSION COMPLETE. Avg Kernel Latency: {avg_latency:.3f} ms")
    
    return params, t_axis, loss_history

if __name__ == "__main__":
    params, t, history = run_mission()
    
    # Visualization
    final_wave = SpectreModulator(N_SAMPLES).encode(params)
    jammer = generate_chirp_jammer(t)
    
    # Use norm='ortho' for correct energy scaling (Parseval's Theorem)
    freqs = jnp.fft.fftfreq(N_SAMPLES, d=1/CONF["SAMPLE_RATE"])
    tx_spec = jnp.abs(jnp.fft.fft(final_wave, norm="ortho"))
    jam_spec = jnp.abs(jnp.fft.fft(jammer, norm="ortho"))

    plt.style.use('dark_background')
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    ax[0].plot(history, color='#00ff00', linewidth=1.5)
    ax[0].set_title("SPECTRE: Optimization Dynamics (Swarm Mode)")
    ax[0].set_ylabel("Loss (Jammer Overlap)")
    ax[0].grid(alpha=0.3)
    
    mask = freqs > 0
    ax[1].plot(freqs[mask]/1e6, jam_spec[mask], 'r', alpha=0.6, label='Threat (Jammer)')
    ax[1].plot(freqs[mask]/1e6, tx_spec[mask], '#00ff00', label='SPECTRE Response')
    ax[1].set_title(f"Spectral Domain: Autonomous Nulling ({int(CONF['SAMPLE_RATE']/1e6)} MHz BW)")
    ax[1].set_xlabel("Frequency (MHz)")
    ax[1].set_ylabel("Power Density")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectre_result.png') 
    print("Graph saved to spectre_result.png")