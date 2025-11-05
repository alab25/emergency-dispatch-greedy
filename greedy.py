import random
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Request:
    id: int
    pickup: int  # location on a line (simplified to 1D)
    hospital: int
    call_time: float
    urgency: int  # 3=critical, 2=urgent, 1=standard
    duration: float  # service time including travel
    deadline: float
    
    def score(self):
        """Urgency per unit time"""
        return self.urgency / self.duration if self.duration > 0 else 0

@dataclass
class Ambulance:
    id: int
    position: int
    available_time: float

def travel_time(from_pos: int, to_pos: int, speed: float = 1.0) -> float:
    """Calculate travel time between two positions"""
    return abs(to_pos - from_pos) / speed

def greedy_emergency_dispatch(
    requests: List[Request], 
    ambulances: List[Ambulance],
    speed: float = 1.0
) -> Tuple[List[Tuple[Request, Ambulance]], float, int]:
    """
    Greedy algorithm for emergency dispatch
    Returns: (assignments, total_weight, comparisons_count)
    """
    # Sort by score (urgency/duration), then by deadline
    sorted_requests = sorted(requests, key=lambda r: (-r.score(), r.deadline))
    
    assignments = []
    total_weight = 0
    comparisons = 0
    
    # Track ambulance states
    amb_states = {amb.id: {'position': amb.position, 'available_time': amb.available_time} 
                  for amb in ambulances}
    
    for request in sorted_requests:
        best_ambulance = None
        earliest_completion = float('inf')
        
        # Find best ambulance for this request
        for amb in ambulances:
            comparisons += 1
            state = amb_states[amb.id]
            
            # Calculate when ambulance can arrive at pickup
            travel_to_pickup = travel_time(state['position'], request.pickup, speed)
            arrival_time = state['available_time'] + travel_to_pickup
            
            # Calculate completion time
            completion_time = arrival_time + request.duration
            
            # Check feasibility and optimality
            if completion_time <= request.deadline and completion_time < earliest_completion:
                best_ambulance = amb.id
                earliest_completion = completion_time
        
        # Assign if feasible
        if best_ambulance is not None:
            assignments.append((request, best_ambulance))
            total_weight += request.urgency
            
            # Update ambulance state
            amb_states[best_ambulance]['position'] = request.hospital
            amb_states[best_ambulance]['available_time'] = earliest_completion
    
    return assignments, total_weight, comparisons

def generate_random_instance(n_requests: int, n_ambulances: int, 
                            city_size: int = 100, seed: int = None) -> Tuple[List[Request], List[Ambulance]]:
    """Generate random test instance"""
    if seed:
        random.seed(seed)
    
    requests = []
    for i in range(n_requests):
        pickup = random.randint(0, city_size)
        hospital = random.randint(0, city_size)
        call_time = random.uniform(0, 100)
        urgency = random.choice([1, 2, 3])
        
        # Duration = travel time + treatment time
        travel = abs(hospital - pickup)
        treatment = random.uniform(10, 30)
        duration = travel + treatment
        
        # Deadline based on urgency (tighter for critical)
        deadline_slack = {3: 1.5, 2: 2.0, 1: 3.0}
        deadline = call_time + duration * deadline_slack[urgency]
        
        requests.append(Request(i, pickup, hospital, call_time, urgency, duration, deadline))
    
    ambulances = [Ambulance(i, random.randint(0, city_size), 0.0) 
                  for i in range(n_ambulances)]
    
    return requests, ambulances

def time_algorithm(n_requests: int, n_ambulances: int, trials: int = 5) -> Tuple[float, int]:
    """Time the algorithm and count comparisons"""
    total_time = 0
    total_comparisons = 0
    
    for _ in range(trials):
        requests, ambulances = generate_random_instance(n_requests, n_ambulances)
        
        start = time.perf_counter()
        _, _, comparisons = greedy_emergency_dispatch(requests, ambulances)
        end = time.perf_counter()
        
        total_time += (end - start)
        total_comparisons += comparisons
    
    return total_time / trials, total_comparisons / trials

def run_experiments():
    """Run timing experiments and generate plots"""
    print("Running experiments...")
    
    # Experiment 1: Vary n (number of requests) with k=5 ambulances
    n_values = [10, 50, 100, 200, 500, 1000, 2000]
    k = 5
    
    times_1 = []
    comparisons_1 = []
    theoretical_1 = []
    
    for n in n_values:
        avg_time, avg_comp = time_algorithm(n, k, trials=5)
        times_1.append(avg_time * 1000)  # Convert to ms
        comparisons_1.append(avg_comp)
        theoretical_1.append(n * k)  # O(nk) theoretical
        print(f"n={n}, k={k}: {avg_time*1000:.2f}ms, {avg_comp:.0f} comparisons")
    
    # Experiment 2: Vary k (number of ambulances) with n=500 requests
    k_values = [1, 2, 5, 10, 20, 50]
    n = 500
    
    times_2 = []
    comparisons_2 = []
    theoretical_2 = []
    
    for k in k_values:
        avg_time, avg_comp = time_algorithm(n, k, trials=5)
        times_2.append(avg_time * 1000)
        comparisons_2.append(avg_comp)
        theoretical_2.append(n * k)
        print(f"n={n}, k={k}: {avg_time*1000:.2f}ms, {avg_comp:.0f} comparisons")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time vs n
    axes[0, 0].plot(n_values, times_1, 'bo-', label='Measured', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Requests (n)', fontsize=12)
    axes[0, 0].set_ylabel('Time (ms)', fontsize=12)
    axes[0, 0].set_title(f'Running Time vs n (k={5})', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Comparisons vs n (with theoretical O(nk))
    axes[0, 1].plot(n_values, comparisons_1, 'ro-', label='Measured', linewidth=2, markersize=8)
    axes[0, 1].plot(n_values, theoretical_1, 'g--', label='Theoretical O(nk)', linewidth=2)
    axes[0, 1].set_xlabel('Number of Requests (n)', fontsize=12)
    axes[0, 1].set_ylabel('Comparisons', fontsize=12)
    axes[0, 1].set_title(f'Comparisons vs n (k={5})', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Time vs k
    axes[1, 0].plot(k_values, times_2, 'mo-', label='Measured', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Ambulances (k)', fontsize=12)
    axes[1, 0].set_ylabel('Time (ms)', fontsize=12)
    axes[1, 0].set_title(f'Running Time vs k (n={500})', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Comparisons vs k (with theoretical O(nk))
    axes[1, 1].plot(k_values, comparisons_2, 'co-', label='Measured', linewidth=2, markersize=8)
    axes[1, 1].plot(k_values, theoretical_2, 'g--', label='Theoretical O(nk)', linewidth=2)
    axes[1, 1].set_xlabel('Number of Ambulances (k)', fontsize=12)
    axes[1, 1].set_ylabel('Comparisons', fontsize=12)
    axes[1, 1].set_title(f'Comparisons vs k (n={500})', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('experimental_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'experimental_results.png'")
    plt.show()
    
    # Test correctness with small example
    print("\n" + "="*60)
    print("CORRECTNESS TEST - Small Example")
    print("="*60)
    
    requests, ambulances = generate_random_instance(10, 3, seed=42)
    assignments, total_weight, comparisons = greedy_emergency_dispatch(requests, ambulances)
    
    print(f"\nGenerated {len(requests)} requests, {len(ambulances)} ambulances")
    print(f"Served {len(assignments)} requests")
    print(f"Total urgency weight: {total_weight}")
    print(f"Comparisons made: {comparisons}")
    print("\nAssignments:")
    for req, amb_id in assignments[:5]:  # Show first 5
        print(f"  Request {req.id} (urgency={req.urgency}, score={req.score():.3f}) -> Ambulance {amb_id}")

if __name__ == "__main__":
    run_experiments()
