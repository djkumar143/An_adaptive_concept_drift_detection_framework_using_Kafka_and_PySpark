# An adaptive real-time concept drift Detection in streaming data using Kafka and PySpark

A real-time streaming framework for detecting concept drift in data streams using Apache Kafka and PySpark, with adaptive algorithms and an interactive dashboard.

## 🚀 Features and things to consider

- **Dataset compatibility for Consumer**: Although the producer is designed to stream many attributes, the consumer file is currently
 runs for three attributes and one class label. So, select three attributes only after you run the producer file.
- **Real-time Drift Detection**: Detects sudden, gradual, and incremental concept drifts
- **Adaptive Algorithms**:
  - KL Divergence for sudden drift detection
  - Hellinger Distance for gradual/incremental drift detection
- **Dynamic Window Management**: Automatically adjusts window size based on drift detection
- **Interactive Dashboard**: Real-time visualization of drift scores, model accuracy, and system metrics
- **Model Management**: Automatic model retraining and switching upon drift detection
- **Benchmark Support**: Works with standard datasets (SEA, ELEC2) and custom data

## 📋 Prerequisites
In your Linux system,
- Open terminal
  ctrl + alt + t

- type the command: `sudo apt update && sudo apt upgrade`

- Python 3.11
  install python3.11 using the command:
  `sudo apt install python3.11`, then confirm by typing `y`(whenever required)
  then verify using:
  `whereis python3.11`
  `python3.11 -m pip version`
-upgrade pip to latest version if needed
  `pip install -upgrade pip`

- Java 8 or later (for Spark), java 21 is being used in this project
- `pip install openjdk-21-jdk -y`

- check the java version using the command: 
  `java -version`

- Apache Kafka
  download kafka2.12:3.0.0 from https://kafka.apache.org/downloads

- Apache PySpark
  download the pyspark3.2.0(spark-3.2.0-bin-hadoop3.2.tgz) from the repository link: https://archive.apache.org/dist/spark/spark-3.2.0/

- Extract the PySpark file. Then set the path using:
  type the command to open the bashrc file where, 
  `sudo nano ~/.bashrc`
 then.. go to the last line and set the path to the SPARK_HOME by typing:
  export SPARK_HOME=/path/to/your/extracted/PySpark/home/
  export PATH=$PATH:$SPARK_HOME/bin

  and also for java
  export JAVA_HOME=/path/to/java/home/
  export PATH=$PATH:$JAVA_HOME/bin/

- save by: ctrl + o, then press Enter, then ctrl + x

- source the bashrc file by:
`source ~/.bashrc`

- Now check the Java version again:
`java -version`

Now, go to the project repository and create a python environment.
- create an environment using python3.11
`python3.11 -m venv myenv`

- After the virtual environment is created, then activate the environment using the command:
  `source myenv/bin/activate`

- Required Python packages:
  `pip install pyspark==3.2.0 kafka-python confluent-kafka numpy pandas matplotlib liac-arff ipywidgets`

## 🛠️ Setup
- Now, In a separate terminal tab, go to the kafka home directory
1. **Start Kafka Server**:
   # Start Zookeeper
   `bin/zookeeper-server-start.sh config/zookeeper.properties`
Now, open another terminal tab and go to kafka home directory again and
   # Start Kafka
   `bin/kafka-server-start.sh config/server.properties`
Now. open another terminal tab and go to the kafka home directory and
2. **Create Kafka Topic**:

 `bin/kafka-topics.sh --create --topic drift_stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1`
   
## 📊 Usage
In the Environment containing all the dependencies for this project,start vs code by typing the command:
`code .`
Now, select and open the kafka_producer notebook
### 1. Start the Producer
Run the Kafka producer notebook (`kafka_producer.ipynb`):
- Select kafka topic from available list of kafka topics
- Select dataset
- Select numerical features (select any three attributes only if you want to run consumer_comparison.ipynb)
- Choose the class attribute
- Start streaming data

Once we are able to produce messages to the kafka topic, then,
### 2. Run the Consumer
Execute the PySpark consumer notebook:(`consumer_comparison.ipynb`) which shows the side by side comparison of the detectors.
Execute the PySpark consumer notebook:(`consumer_benchmarking.ipynb`) which is for the purpose of benchmarking of the detectors.
```python
# The control panel will appear automatically
# Click "Start System" to begin drift detection
```

### 3. Monitor the Dashboard
The dashboard displays:
- **Real-time Status**: Current drift detection status
- **Metrics Table**: KL/Hellinger scores, thresholds, accuracy
- **Performance Summary**: Total drifts, drift rate, processing time
- **Visualizations**:
  - Drift scores over time
  - Model accuracy trends
  - Adaptive window size changes

## 🎯 Drift Detection

| Detector | Drift Type | Description |
|----------|------------|-------------|
| KL Divergence | Sudden | Detects abrupt changes in data distribution |
| Hellinger Distance | Gradual/Incremental | Identifies slow, progressive changes |
| Hybrid fusion | Sudden/Gradual/Incremental | Identifies mixed drift types |

## 📁 Output Files

Results are saved in the `results/` directory:
- `drift_detection_results_[timestamp].csv` - Drift detection events
- `drift_detection_summary_[timestamp].json` - Performance summary
- `drift_detection_metrics_[timestamp].json` - Detailed metrics

## ⚙️ Configuration

Key parameters in the consumer notebook:
```python
INITIAL_WINDOW_SIZE = 200          # Starting window size
MIN_WINDOW_SIZE = 150              # Minimum adaptive window
MAX_WINDOW_SIZE = 500             # Maximum adaptive window
BATCH_INTERVAL = "2 seconds"      # Streaming batch interval
DRIFT_THRESHOLD_MULTIPLIER = 1.0  # Sensitivity adjustment
```

## 🧪 Benchmark Testing

Run benchmark tests to evaluate detector performance:
```python
# Click "Run Benchmark" in the control panel
# Tests sudden, gradual, and incremental drift scenarios
```

## 📈 Performance Metrics

- **Drift Detection Delay (DDD)**: Time to detect drift after occurrence
- **False Positive Rate (FPR)**: Incorrect drift detections
- **Accuracy Before/After Adaptation**: Model performance metrics
- **Area Under Performance Curve (AUPC)**: Overall system performance

## 🔧 Troubleshooting

1. **Kafka Connection Error**: Ensure Kafka is running on `localhost:9092`
2. **No Data Received**: Check if producer is streaming to correct topic
3. **Memory Issues**: Adjust `spark.driver.memory` in Spark configuration
4. **Slow Processing**: Increase `BATCH_INTERVAL` or reduce `MAX_OFFSETS_PER_TRIGGER`

## 🤝 Architecture

```
Producer (Kafka) → Topic → Consumer (PySpark) → Dashboard
                              ↓
                    Drift Detection System
                    ├── KL Divergence Detector
                    ├── Hellinger Distance Detector
                    ├── Adaptive Window Manager
                    ├── Hoeffding Tree Model
                    └── Performance Tracker
```

## 👥 Contributors

- Deep Jyoti     (mde2023008@iiita.ac.in)        
- Sadhana Tiwari (rsi2018507@iiita.ac.in, sadhana)
- Sonali Agarwal (bdasp@iiita.ac.in, BDA lab)
- Ritesh Chandra (rsi2022001@iiita.ac.in, Ritesh)