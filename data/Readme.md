# Custom Drift Dataset (ARFF Format)

This synthetic dataset is designed for benchmarking **adaptive concept drift detection frameworks**. It is specifically tailored for real-time streaming environments using tools like **Kafka** and **PySpark**, and is suitable for evaluating different types of **concept drift** including:

- Sudden Drift  
- Gradual Drift  
- Incremental Drift

## Dataset Overview

- **Total Records:** 200,000  
- **Format:** ARFF (Attribute-Relation File Format)  
- **Attributes:**
  - `at1` - Numeric Feature
  - `at2` - Numeric Feature
  - `at3` - Numeric Feature
  - `cl`  - Categorical Class Label {0, 1}

## Drift Injection Details

Drift is injected at **fixed intervals of 10,000 records**:

| Record Range     | Drift Type     |
|------------------|----------------|
| 1–10,000         | No Drift (Initial Stable Data) |
| 10,001–20,000    | Sudden Drift   |
| 20,001–30,000    | Gradual Drift  |
| 30,001–40,000    | Incremental Drift |
| 40,001–50,000    | Sudden Drift   |
| ...              | ...            |
| 190,001–200,000  | Random Drift (Mixed Types) |

## Use Case

This dataset is intended for use in:

- **Adaptive Drift Detection Projects**
- **Streaming ML Pipelines (Kafka + PySpark)**
- **Benchmarking Drift Detection Algorithms:**
  - KL Divergence
  - Hellinger Distance
  - Hybrid Methods

## Format Example

```arff
@relation custom_data

@attribute at1 numeric
@attribute at2 numeric
@attribute at3 numeric
@attribute cl {0,1}

@data
4.551125386,8.715686657,5.348319297,0
4.569528788,9.455497185,2.354468491,0
0.878607722,0.919708406,4.903802179,1
...

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## How to Use
Kafka Producer: Use the ARFF file as the source stream to send data in real time to a Kafka topic.

PySpark Consumer: Use sliding window and adaptive threshold logic to detect concept drift while consuming the data.

##License
This dataset is generated synthetically for academic research purposes. No real-world data is included. Free to use under the MIT License.