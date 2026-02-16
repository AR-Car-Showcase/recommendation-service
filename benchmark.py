import time
import os
import json
import numpy as np
from recommender import Recommender

def benchmark():
    data_path = "dataset/cars_data_final.json"
    cache_dir = "cache"

    rec_init = Recommender(data_path=data_path, cache_dir=cache_dir)
    
    start_time = time.time()
    rec = Recommender(data_path=data_path, cache_dir=cache_dir)
    init_time = time.time() - start_time
    print(f"Initialization time (cached): {init_time:.4f}s")
    
    preferences = {
        "brands": ["bmw", "audi"],
        "body_types": ["suv"],
        "fuel_types": ["petrol"],
        "transmissions": ["automatic"],
        "budget": 80.0,
        "user_id": "benchmark_user"
    }
    
    latencies = []
    for _ in range(100):
        start_time = time.time()
        rec.get_personalized_ranking(preferences)
        latencies.append(time.time() - start_time)
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"Average Personalized Ranking Latency: {avg_latency*1000:.2f}ms")
    print(f"P95 Personalized Ranking Latency: {p95_latency*1000:.2f}ms")

if __name__ == "__main__":
    benchmark()
