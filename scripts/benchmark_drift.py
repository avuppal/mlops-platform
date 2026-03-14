import time
import numpy as np
from src.drift import DriftDetector

def run_benchmark():
    num_features = [10, 100, 1000]
    num_samples = 5000
    results = []
    
    print("Benchmarking drift detection on large feature sets...")
    
    for n_feat in num_features:
        # Generate random baseline and target data
        baseline_data = np.random.randn(num_samples, n_feat)
        target_data = np.random.randn(num_samples, n_feat)
        
        # Simulate drift in target
        target_data += 0.5
        
        start = time.time()
        detector = DriftDetector(method="psi")
        for i in range(n_feat):
            detector.detect(baseline_data[:, i], target_data[:, i])
            
        elapsed = time.time() - start
        results.append({"features": n_feat, "time": elapsed})
        
    print("\nBenchmark Results:")
    print("Features | Time (s)")
    print("-" * 25)
    for r in results:
        print(f"{r['features']:<8} | {r['time']:<10.4f}")

if __name__ == "__main__":
    run_benchmark()
