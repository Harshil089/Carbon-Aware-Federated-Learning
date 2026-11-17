# Final review — Questions and defensible answers

**Q: Have you connected four physical computers?**  
No. The lab is inaccessible because of examinations. We use four logical clients on one laptop. Physical distributed validation is pending.

**Q: Are the four computers' carbon emissions measured?**  
No. A single wall-meter observation exists for PC1: 0.065 kWh over 190 min. The associated emissions are estimated using the CEA grid factor, and other PCs are assumption-based what-if models.

**Q: Why is 20.53 W low for an i7-12700?**  
That is exactly why this is a preliminary datum. We will recheck the meter display/unit, scope, whether energy was a start/end delta, the measured load and repeated runs.

**Q: Why did you assume PC3 is faster?**  
We did not verify that; a higher modeled speed is one scenario *if* its computational tasks use the GPU or a more powerful CPU. GPU VRAM capacity and RAM do not establish speed. Assumptions are configurable, not measured.

**Q: Are you using reinforcement learning or just hard-coded selection?**  
We implement tabular Q-learning with epsilon-greedy exploration, temporal-difference updates, a state representing accuracy level and training phase, actions selecting 2–4 clients, and a weighted reward trading accuracy improvement against modeled time and electricity-related CO₂. We separately compare two fixed heuristics and FedAvg.

**Q: Did you actually train an ML model?**  
Yes, four NumPy clients each update a real multinomial logistic-regression model on separate synthetic non-IID data partitions; the server aggregates model weights with sample-weighted FedAvg. Hardware/network energy is simulation-based.

**Q: Is the RL result necessarily best?**  
No. It is an exploratory baseline. The dashboard reports all outcomes including cases where simpler heuristics win. We need tuning, ablations, multiple datasets, repeat trials and physical validation before claiming improvement.

**Q: What is the unit of CO₂ factor?**  
0.675 kg CO₂ per kWh = 0.675 t CO₂/MWh, CEA v22.0, FY2025–26 weighted Indian grid average (including renewables). This is not dynamic marginal emissions, nor a direct carbon sensor.

**Q: How do you compare policies fairly?**  
The same synthetic partitions and a common per-round random variation are used for policies with the same seed. We report both fixed-round outcomes and time/energy to an accuracy threshold. We would expand to several data generation seeds and confidence intervals in the next phase.

**Q: What is left after November 19?**  
Verify reading; obtain full CPU/GPU details; collect replicated watts/kWh for all PCs; run a true LAN/Flower version; measure network latency and idle power; run an established dataset; compare at a common accuracy threshold, plus Pareto-front and significance checks.
