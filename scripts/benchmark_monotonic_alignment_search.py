from tts_impl.functional.monotonic_align import available_mas_algorithms, default_mas_alogirhtm, maximum_path
import torch
import time


Ts = [128, 256, 512, 1024]
B = 16

print("AVAILABLE:", available_mas_algorithms)
print("DEFAULT:", default_mas_alogirhtm)

for algorithm in available_mas_algorithms:
    print(f"benchmarking {algorithm}")
    for T in Ts:
        S = T * 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attn = torch.randn(B, T, S, device=device)
        start_time = time.perf_counter()
        maximum_path(attn, torch.ones_like(attn), algorithm)
        end_time = time.perf_counter()

        delta = end_time - start_time
        print(f"[algorithm={algorithm}, T={T}]: {delta}s")
