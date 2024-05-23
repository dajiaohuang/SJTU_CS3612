1. |        | 0.001 | 0.01   | 0.1    | 1      | 10     | 100    | 1000   |
   | ------ | ----- | ------ | ------ | ------ | ------ | ------ | ------ |
   | linear | 0.978 | 0.9805 | 0.9805 | 0.9805 | 0.9805 | 0.9805 | 0.9805 |
   | rbf    | 0.885 | 0.93   | 0.9575 | 0.974  | 0.9735 | 0.9735 | 0.9735 |
   | poly   | 0.5   | 0.502  | 0.925  | 0.969  | 0.9735 | 0.9735 | 0.9735 |

2. 对于 Linear SVM，c=0.1时，支持向量有662个
   top 5 positive：
   ![image-20240427210058730](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240427210058730.png)

   top 5 negative：
   ![image-20240427210115344](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240427210115344.png)
   5 outliers：

   ![image-20240427210128213](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240427210128213.png)

   

3. | C      | Num of Support Vectors |
   | :----- | :--------------------- |
   | 0.001  | 6000                   |
   | 0.01   | 4125                   |
   | 0.1    | 1675                   |
   | 1      | 1331                   |
   | 10.0   | 1434                   |
   | 100.0  | 1434                   |
   | 1000.0 | 1434                   |