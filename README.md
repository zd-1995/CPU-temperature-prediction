# CPU-temperature-prediction
CPU temperature prediction using neural network
In this project, we want to predict the temperature in 10 seconds with dual-core cpu temperature information. Using cnn and lstm neural network, we have trained a model to predict cpu temperature.

The results of the two networks in benchmarks for seconds 2 to 5 are given below:

![lstm](https://github.com/zd-1995/CPU-temperature-prediction/assets/89040004/932ff9c3-60ab-485e-ac03-8105511aab7e)
![cnn](https://github.com/zd-1995/CPU-temperature-prediction/assets/89040004/90baeb23-9621-40d0-8080-7025d59554b7)

Also, the temperature forecast in 10 seconds for the test data is as follows:
Initial test data:
[xtest.csv](https://github.com/zd-1995/CPU-temperature-prediction/files/12329152/xtest.csv)

Test data predicted by the network:
[result_pre.csv](https://github.com/zd-1995/CPU-temperature-prediction/files/12329168/result_pre.csv)

The results of the difference between the initial test data and the predicted test data:
[result_diff.csv](https://github.com/zd-1995/CPU-temperature-prediction/files/12329169/result_diff.csv)
