# Emergency Dispatch: Greedy Algorithm

This repository contains a Python implementation designed to solve emergency dispatch routing and assignment using a greedy algorithmic approach[cite: 3].

## 📂 Repository Structure

*   **`greedy.py`**: The primary Python script containing the core logic for the greedy algorithm[cite: 3].
*   **`README.md`**: Project documentation[cite: 3].

## 🚀 Overview

In emergency dispatch scenarios, the goal is to allocate limited resources (such as ambulances, fire trucks, or police units) to incoming incidents as quickly and efficiently as possible. 

This project tackles the problem using a **Greedy Algorithm**[cite: 3]. A greedy approach makes the locally optimal choice at each step. For emergency dispatch, this typically involves immediately assigning the closest or fastest available unit to a new emergency, prioritizing immediate response times over globally calculating the absolute optimal routing for all current and future emergencies.

## 💻 Usage

Ensure you have Python 3.x installed. To run the algorithm, navigate to the project directory and execute the script from your terminal:

```bash
python greedy.py
```
