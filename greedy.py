"""
Emergency Vehicle Dispatch Optimization
Greedy Algorithm Implementation with Experimental Validation
Generates experimental_results.png for LaTeX report
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class Request:
    """Represents an emergency service request."""
    
    def __init__(self, req_id: int, start: float, duration: float, 
                 urgency: int, deadline: float):
        self.id = req_id
        self.start = start
        self.duration = duration
        self.urgency = urgency
        self.deadline = deadline
        self.finish = start + duration
        self.score = 0.0
    
    def __repr__(self):
        return (f"Request({self.id}, [{self.start:.1f}, {self.finish:.1f}], "
                f"u={self.urgency}, score={self.score:.3f})")


def greedy_emergency_dispatch(requests: List[Request], 
                               k_ambulances: int) -> Tuple[List, float, int]:
    """
    Greedy algorithm for emergency dispatch optimization.
    
    Args:
        requests: List of Request objects
        k_ambulances: Number of available ambulances
    
    Returns:
        Tuple of (assignments, total_weight, comparisons)
        - assignments: List of (request_id, ambulance_id) tuples
        - total_weight: Sum of urgency weights for served requests
        - comparisons: Number of comparisons performed (for analysis)
    """
    # Compute urgency efficiency scores
    for req in requests:
        req.score = req.urgency / req.duration
    
    # Sort by score (descending), then deadline (ascending) for tie-breaking
    sorted_requests = sorted(
        requests,
        key=lambda r: (-r.score, r.deadline)
    )
    
    assignments = []
    total_weight = 0
    comparisons = 0
    
    # Initialize ambulance availability times (all start at time 0)
    ambulance_avail = [0.0] * k_ambulances
    
    # Greedy assignment loop
    for request in sorted_requests:
        best_ambulance = None
        earliest_completion = float('inf')
        
        # Find ambulance that can complete request earliest
        for j in range(k_ambulances):
            comparisons += 1
            
            # Determine when this ambulance can complete the request
            if ambulance_avail[j] <= request.start:
                # Ambulance is free before request start time
                completion = request.finish
            else:
                # Ambulance is busy, will serve request after current assignment
                completion = ambulance_avail[j] + request.duration
            
            # Check if this assignment is feasible and better than current best
            if completion <= request.deadline and completion < earliest_completion:
                best_ambulance = j
                earliest_completion = completion
        
        # Assign request if feasible ambulance was found
        if best_ambulance is not None:
            assignments.append((request.id, best_ambulance))
            total_weight += request.urgency
            ambulance_avail[best_ambulance] = earliest_completion
    
    return assignments, total_weight, comparisons


def generate_random_instance(n_requests: int, k_ambulances: int, 
                             seed: int = None) -> List[Request]:
    """
    Generate random test instance.
    
    Args:
        n_requests: Number of emergency requests
        k_ambulances: Number of ambulances (not used in generation but for API)
        seed: Random seed for reproducibility
    
    Returns:
        List of Request objects
    """
    if seed is not None:
        random.seed(seed)
    
    requests = []
    urgency_levels = [1, 2, 3]  # Standard, Urgent, Critical
    urgency_dist = [0.3, 0.4, 0.3]  # Distribution weights
    
    for i in range(n_requests):
        start = random.uniform(0, 500)
        duration = random.uniform(10, 60)
        urgency = random.choices(urgency_levels, weights=urgency_dist)[0]
        slack = random.uniform(0, 50)  # Extra time before deadline
        deadline = start + duration + slack
        
        requests.append(Request(i, start, duration, urgency, deadline))
    
    return requests


def run_experiment_varying_n(k: int = 5, n_values: List[int] = None, 
                             trials: int = 10) -> dict:
    """
    Experiment 1: Vary number of requests, fix number of ambulances.
    
    Args:
        k: Fixed number of ambulances
        n_values: List of request counts to test
        trials: Number of trials per configuration
    
    Returns:
        Dictionary with results
    """
    if n_values is None:
        n_values = [50, 100, 200, 300, 400, 500]
    
    results = {'n': [], 'time': [], 'comparisons': []}
    
    print(f"Experiment 1: Varying n (k={k} fixed)")
    print("-" * 50)
    for n in n_values:
        times = []
        comps = []
        
        for trial in range(trials):
            requests = generate_random_instance(n, k, seed=trial)
            
            start_time = time.perf_counter()
            _, _, comparisons = greedy_emergency_dispatch(requests, k)
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            times.append(elapsed)
            comps.append(comparisons)
        
        avg_time = np.mean(times)
        avg_comps = np.mean(comps)
        
        results['n'].append(n)
        results['time'].append(avg_time)
        results['comparisons'].append(avg_comps)
        
        print(f"  n={n:3d}: time={avg_time:6.2f}ms, comparisons={avg_comps:7.0f}")
    
    return results


def run_experiment_varying_k(n: int = 500, k_values: List[int] = None, 
                             trials: int = 10) -> dict:
    """
    Experiment 2: Vary number of ambulances, fix number of requests.
    
    Args:
        n: Fixed number of requests
        k_values: List of ambulance counts to test
        trials: Number of trials per configuration
    
    Returns:
        Dictionary with results
    """
    if k_values is None:
        k_values = [2, 4, 6, 8, 10, 12]
    
    results = {'k': [], 'time': [], 'comparisons': []}
    
    print(f"\nExperiment 2: Varying k (n={n} fixed)")
    print("-" * 50)
    for k in k_values:
        times = []
        comps = []
        
        for trial in range(trials):
            requests = generate_random_instance(n, k, seed=trial)
            
            start_time = time.perf_counter()
            _, _, comparisons = greedy_emergency_dispatch(requests, k)
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            times.append(elapsed)
            comps.append(comparisons)
        
        avg_time = np.mean(times)
        avg_comps = np.mean(comps)
        
        results['k'].append(k)
        results['time'].append(avg_time)
        results['comparisons'].append(avg_comps)
        
        print(f"  k={k:2d}: time={avg_time:6.2f}ms, comparisons={avg_comps:7.0f}")
    
    return results


def create_experimental_plots(exp1_results: dict, exp2_results: dict, 
                              filename: str = 'experimental_results.png'):
    """
    Create 4-panel figure showing experimental validation.
    Matches the style expected in academic papers.
    
    Args:
        exp1_results: Results from experiment 1 (varying n)
        exp2_results: Results from experiment 2 (varying k)
        filename: Output filename for plot
    """
    # Set style for academic paper
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Experimental Validation of Time Complexity Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # --- Plot (a): Running time vs n (k=5) ---
    ax1.plot(exp1_results['n'], exp1_results['time'], 'bo-', 
             linewidth=2, markersize=8, label='Measured')
    ax1.set_xlabel('Number of Requests (n)', fontsize=11)
    ax1.set_ylabel('Running Time (ms)', fontsize=11)
    ax1.set_title('(a) Running Time vs n (k=5)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # --- Plot (b): Comparisons vs n (k=5) ---
    ax2.plot(exp1_results['n'], exp1_results['comparisons'], 'ro-', 
             linewidth=2, markersize=8, label='Measured')
    # Add theoretical line (5n)
    n_vals = np.array(exp1_results['n'])
    ax2.plot(n_vals, 5 * n_vals, 'k--', linewidth=2, alpha=0.7,
             label='Theoretical: 5n')
    ax2.set_xlabel('Number of Requests (n)', fontsize=11)
    ax2.set_ylabel('Number of Comparisons', fontsize=11)
    ax2.set_title('(b) Comparisons vs n (k=5)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # --- Plot (c): Running time vs k (n=500) ---
    ax3.plot(exp2_results['k'], exp2_results['time'], 'go-', 
             linewidth=2, markersize=8, label='Measured')
    ax3.set_xlabel('Number of Ambulances (k)', fontsize=11)
    ax3.set_ylabel('Running Time (ms)', fontsize=11)
    ax3.set_title('(c) Running Time vs k (n=500)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # --- Plot (d): Comparisons vs k (n=500) ---
    ax4.plot(exp2_results['k'], exp2_results['comparisons'], 'mo-', 
             linewidth=2, markersize=8, label='Measured')
    # Add theoretical line (500k)
    k_vals = np.array(exp2_results['k'])
    ax4.plot(k_vals, 500 * k_vals, 'k--', linewidth=2, alpha=0.7,
             label='Theoretical: 500k')
    ax4.set_xlabel('Number of Ambulances (k)', fontsize=11)
    ax4.set_ylabel('Number of Comparisons', fontsize=11)
    ax4.set_title('(d) Comparisons vs k (n=500)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n{'='*60}")
    print(f"Graph saved as: {filename}")
    print(f"{'='*60}")
    print(f"\nYou can now upload this PNG file to Overleaf along with your LaTeX document.")
    
    # Also show the plot
    plt.show()


def demo_algorithm():
    """Demonstrate the algorithm with a small example."""
    print("\n" + "="*60)
    print("DEMO: Small Example")
    print("="*60)
    
    # Create a small test instance
    requests = [
        Request(0, start=0, duration=30, urgency=3, deadline=40),
        Request(1, start=5, duration=20, urgency=2, deadline=35),
        Request(2, start=10, duration=15, urgency=3, deadline=30),
        Request(3, start=15, duration=40, urgency=1, deadline=70),
        Request(4, start=20, duration=10, urgency=2, deadline=40),
    ]
    
    k = 2  # 2 ambulances
    
    print(f"\nRequests (n={len(requests)}):")
    for req in requests:
        print(f"  {req}")
    
    print(f"\nAmbulances: k={k}")
    
    # Run algorithm
    assignments, total_weight, comparisons = greedy_emergency_dispatch(requests, k)
    
    print(f"\nResults:")
    print(f"  Assignments made: {len(assignments)}")
    print(f"  Total urgency weight: {total_weight}")
    print(f"  Comparisons performed: {comparisons}")
    print(f"\nAssignment details:")
    for req_id, amb_id in assignments:
        req = requests[req_id]
        print(f"  Request {req_id} → Ambulance {amb_id} "
              f"(urgency={req.urgency}, duration={req.duration:.1f}, score={req.score:.3f})")


def print_summary_statistics(exp1_results: dict, exp2_results: dict):
    """Print summary statistics from experiments."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nExperiment 1 - Varying n (k=5):")
    print(f"  Number of test configurations: {len(exp1_results['n'])}")
    print(f"  Request range: {min(exp1_results['n'])} - {max(exp1_results['n'])}")
    print(f"  Time range: {min(exp1_results['time']):.2f}ms - {max(exp1_results['time']):.2f}ms")
    print(f"  Average comparisons per request: {np.mean(exp1_results['comparisons']) / np.mean(exp1_results['n']):.2f}")
    
    print("\nExperiment 2 - Varying k (n=500):")
    print(f"  Number of test configurations: {len(exp2_results['k'])}")
    print(f"  Ambulance range: {min(exp2_results['k'])} - {max(exp2_results['k'])}")
    print(f"  Time range: {min(exp2_results['time']):.2f}ms - {max(exp2_results['time']):.2f}ms")
    print(f"  Average comparisons per ambulance: {np.mean(exp2_results['comparisons']) / np.mean(exp2_results['k']):.2f}")
    
    # Verify theoretical predictions
    print("\nTheoretical Validation:")
    print(f"  Exp 1 - Expected comparisons (5n): {[5*n for n in exp1_results['n']]}")
    print(f"  Exp 1 - Measured comparisons:       {[int(c) for c in exp1_results['comparisons']]}")
    print(f"  Exp 2 - Expected comparisons (500k): {[500*k for k in exp2_results['k']]}")
    print(f"  Exp 2 - Measured comparisons:        {[int(c) for c in exp2_results['comparisons']]}")


if __name__ == "__main__":
    print("="*60)
    print("Emergency Vehicle Dispatch Optimization")
    print("Greedy Algorithm - Experimental Validation")
    print("="*60)
    
    # Run demo
    demo_algorithm()
    
    # Run experiments
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60 + "\n")
    
    # Experiment 1: Varying n with k=5
    exp1 = run_experiment_varying_n(k=5, trials=10)
    
    # Experiment 2: Varying k with n=500
    exp2 = run_experiment_varying_k(n=500, trials=10)
    
    # Print summary statistics
    print_summary_statistics(exp1, exp2)
    
    # Create and save plots
    print("\n" + "="*60)
    print("GENERATING GRAPH")
    print("="*60)
    create_experimental_plots(exp1, exp2, filename='experimental_results.png')
    
    print("\n✓ All experiments completed successfully!")
    print("✓ Graph generated: experimental_results.png")
    print("\nNext steps:")
    print("  1. Upload 'experimental_results.png' to Overleaf")
    print("  2. Place it in the same folder as your .tex file")
    print("  3. Compile your LaTeX document")
    print("  4. Verify the figure appears in Section VI")
