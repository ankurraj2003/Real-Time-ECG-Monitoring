import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynq import Overlay, allocate
from scipy.signal import find_peaks


print("Loading Hardware Overlay...")
overlay = Overlay("ecg_filter.bit")
dma = overlay.axi_dma_0 

print("Loading noisy ECG data...")
df = pd.read_csv("noisy_continuous_ecg.csv", header=None)
noisy_data_floats = df.values.flatten()

scale_factor = 10000 
noisy_data_int16 = np.int16(noisy_data_floats * scale_factor)
data_size = len(noisy_data_int16)

# FIX 1: Put the buffer back to its normal, full size
print("Allocating DMA buffers...")
in_buffer = allocate(shape=(data_size,), dtype=np.int16)
out_buffer = allocate(shape=(data_size,), dtype=np.int16)

np.copyto(in_buffer, noisy_data_int16)

print("Streaming data through the FPGA filters...")
dma.recvchannel.transfer(out_buffer)
dma.sendchannel.transfer(in_buffer)

# Wait for the data to finish going IN
dma.sendchannel.wait()

# FIX 2: Do NOT wait for the receive channel. It will wait forever.
# Instead, just pause Python for 1 second to let the hardware finish computing.
print("Giving the hardware 1 second to finish processing...")
time.sleep(1)

# FIX 3: Manually tell the CPU to fetch the fresh data from the physical RAM
out_buffer.invalidate()
print("Hardware filtering complete!")

# Plot the Results
filtered_data_floats = np.array(out_buffer) / scale_factor
t = np.arange(data_size) / 125.0

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(t, noisy_data_floats, color='red')
plt.title("Raw Noisy ECG (Software Input)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(t, filtered_data_floats, color='blue')
plt.title("Cleaned ECG (FPGA Hardware Output)")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Clean up memory
in_buffer.freebuffer()
out_buffer.freebuffer()


# 1. Define the biological parameters for a human heartbeat
sampling_rate = 125.0

# Minimum time between beats: 0.4 seconds (equivalent to a max of 150 BPM)
# This prevents the algorithm from accidentally counting the smaller 'T-wave' as a second heartbeat
min_distance_samples = int(0.4 * sampling_rate)

# Dynamic height threshold: We only want the tall 'R' peaks.
# Let's dynamically set the threshold to 50% of the maximum peak in your specific signal
peak_threshold = np.max(filtered_data_floats) * 0.5

# 2. Find the peaks!
print("Detecting QRS complexes...")
peaks, properties = find_peaks(filtered_data_floats, height=peak_threshold, distance=min_distance_samples)

# 3. Extract Heartbeat Info (RR Intervals and BPM)
# Get the actual times (in seconds) where the peaks occurred
peak_times = t[peaks]

# Calculate the time difference between each consecutive beat (known as the RR interval)
rr_intervals = np.diff(peak_times)

# Convert the RR intervals to instantaneous Beats Per Minute (BPM)
instantaneous_bpm = 60.0 / rr_intervals
average_bpm = np.mean(instantaneous_bpm)

print(f"Total heartbeats detected: {len(peaks)}")
print(f"Average Heart Rate: {average_bpm:.1f} BPM")

# 4. Plot the final medical data
plt.figure(figsize=(15, 6))

# Plot the clean ECG from the FPGA
plt.plot(t, filtered_data_floats, color='blue', label='Cleaned ECG (FPGA)')

# Overlay a red dot on every detected R-peak
plt.plot(t[peaks], filtered_data_floats[peaks], "ro", markersize=8, label='Detected R-Peaks (Software)')

plt.title(f"Automated QRS Detection (Average HR: {average_bpm:.1f} BPM)")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# In[ ]:




