import timeit

import torch

from tts_impl.functional.monotonic_align import (
    available_mas_algorithms,
    default_mas_alogirhtm,
    maximum_path,
)

Ts = [32, 64, 128, 256, 512, 1024]
B = 32

print("AVAILABLE:", available_mas_algorithms)
print("DEFAULT:", default_mas_alogirhtm)

algorithms = available_mas_algorithms
algorithms.remove("naive")

for algorithm in algorithms:
    print(f"benchmarking {algorithm}")
    for T in Ts:
        S = T * 4
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            attn = torch.randn(B, T, S, device=device)
            output = maximum_path(attn, algorithm=algorithm)
            test = (output == maximum_path(attn, algorithm="cython")).all()
            delta = timeit.timeit(
                lambda: maximum_path(attn, torch.ones_like(attn), algorithm), number=5
            )
            print(
                f"[algorithm={algorithm}, B={B}, T={T}, S={S}]: {delta} s, passed_check={test}"
            )
