# Demonstrating the project

Double-click **START_DEMO.bat**. Choose **Quick demo** and click **Run Simulation**. The app runs local softmax training, FedAvg aggregation, Q-learning, policy evaluations, sensitivity analysis, and chart generation.

## Explain the experiment

“The scheduler selects which clients train in each federated round. Each selected client trains a local model on its own data partition. FedAvg combines their model weights. Q-learning rewards accuracy improvement and penalizes modeled latency and carbon.”

## Show the trade-off

Open **Overview**. Compare accuracy and cost metrics together. The full default run gives approximately 83.3% final accuracy for FedAvg, 82.2% for Fastest two, and 78.9% for the default Q-learning policy. Fastest two has lower modeled latency and carbon in this configuration. These results show why the scheduler must be compared against simple baselines.

Target hit rate measures whether a trial reached the target at any round. A policy can reach 80% during training and finish below 80%, so this rate differs from final accuracy.

## Show the scheduler

Open **Client replay**, select **Q-learning scheduler**, and click **Play replay**. Green cards identify selected clients. Replay uses the saved per-round records.

## Show exact parameters

Open **Parameters**. Point out the rounds, episodes, local learning rate, alpha, gamma, exploration schedule, reward weights, client profiles, and formulas. The JSON download contains the exact record for that run.

## Change one setting

Change the number of rounds or the carbon penalty under **Training & reward settings**, then run again. Compare policy choices and accuracy/cost metrics. The effect depends on the learned policy; a larger penalty does not guarantee a better overall result.

Each run has its own folder and parameter record. Use **Export run ZIP** to retain the demonstrated result.

Training uses synthetic non-IID data. Infrastructure metrics use the configured simulation model. Reference inputs and source details remain under Parameters and in the provenance file.
