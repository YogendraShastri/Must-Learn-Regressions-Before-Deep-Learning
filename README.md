# Must-Learn-Regressions-Before-Deep-Learning
**lets see major regression before starting with deep learning and neural networks**

Here's a list of regressions you should definitely learn. 
- Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Logistic Regression
- Bayesian Regression

Optional :
- Ridge & Lasso Regression
- Elastic Net Regression
- Quantile Regression

## What is Linear Regression
We all know the general equation of a straight line is y = mx + c, where m is the slope of the line and c is the y-intercept. So when we talk about linear regression, what we are doing is trying to draw a line that best fits the data, so we can make predictions or understand the relationship between variables.
It might sound a bit confusing, but in linear regression, we try to find a linear relationship between an independent variable (x) and a dependent variable (y).
one more thing **Linear regression** is a type of supervised machine-learning algorithm.

### Equation (Simple Linear Regression)

The equation for **Simple Linear Regression** is:

$$
y = w \cdot x + b
$$

### Where:
- **y** = Predicted value (dependent variable)  
- **x** = Input feature (independent variable)  
- **w** = Weight (slope of the line)  
- **b** = Bias (intercept)

### How Does a Machine “Learn”?
When we say a model learns, we mean, It tries to find the best line (or curve) that fits the data by minimizing the error between predicted and actual values.
To do that, it uses tools like:

- One is actual value y = mx+c
- One is predicted value y bar.
- Loss Function
- Cost Function

### Loss Function & Cost Function
- Measures the Error Per Sample, means what's the difference between actual data and predicted data
- **Loss function**: Measures the error for a single training example.

$$
Loss =  \bigl(y_i - \hat{y}_i\bigr)^2
$$

- **Cost function**: Represents the average loss over the entire training dataset, and is also known as **Mean Squared Error (MSE)**.

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - \hat{y}_i\bigr)^2
$$

**Derivation**
1. **For a single prediction**: This is just the difference between actual and predicted value.
  
$$
ERROR = y_i - \hat{y}_i
$$

2. **Square the Error** : We square it so, Negative errors don’t cancel out positive ones.

$$
SE_i = \bigl(y_i - \hat{y}_i\bigr)^2
$$

3. **Apply to All Samples** : For 𝑛 data points, compute the squared error for each one and sum them:

$$
\mathrm{SSE} = \sum_{i=1}^{n} \bigl(y_i - \hat{y}_i\bigr)^2
$$

4. **Average the Squared Errors** : To get the mean squared error (MSE), divide by the total number of samples 𝑛.

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - \hat{y}_i\bigr)^2
$$

### Gradient descent
- Now we know how to get the cost function, but our model or prediction will be good only if the cost is minimum, so we need to reduce the value of cost, That’s where **Gradient Descent** comes in.
- How can we reduce it, by changing **w** and **b** such that, the **error** decreases.
- This process is repeated many times (called epochs).

$$
w = w - \alpha \cdot \frac{\partial \text{Cost}}{\partial w}
$$

$$
b = b - \alpha \cdot \frac{\partial \text{Cost}}{\partial b}
$$

**Where:**
- alpha = Learning rate
- Partial derivative of the cost function with respect to weight (w).
- Partial derivative of the cost function with respect to bias (b).

**Partial Derivative**

- Cost function formula we already know.

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - \hat{y}_i\bigr)^2
$$

- replace the y bar value.

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (w x_i + b) \right)^2
$$

- Let’s take derivative of J with respect to 𝑤.

$$
\frac{\partial J}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} \frac{d}{dw} \left[ \bigl(y_i - (w x_i + b)\bigr)^2 \right]
$$

- The simplified derivative of the cost function with respect to \( w \) is:

$$
\frac{\partial Cost}{\partial w} = \frac{-2}{n} \sum_{i=1}^{n} x_i \left( y_i - (w x_i + b) \right)
$$

- Similary we can derive for b as well

$$
\frac{\partial Cost}{\partial b} = \frac{-2}{n} \sum_{i=1}^{n} \left( y_i - (w x_i + b) \right)
$$

- We subtract the gradient to move toward minimum.
- Final Gradient Descent Formula, lets replace the derivative value on actual formula.

$$
w = w + \frac{2\alpha}{n} \sum_{i=1}^{n} x_i \left( y_i - \hat{y}_i \right)
$$

$$
b = b + \frac{2\alpha}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)
$$

- Now choosing learning rate is a task, learning rate dictates the size of the steps taken in the direction of the negative gradient (or other optimization methods) to adjust the model's parameters (weights and biases).
- A higher learning rate means larger steps, potentially leading to faster convergence but also the risk of overshooting.
- A lower learning rate means smaller steps, leading to more stable training but also potentially slower convergence and the possibility of getting stuck in local minima. 

![image](https://github.com/user-attachments/assets/9b4363cf-79a7-4474-b320-61c9c09b6f23)

Now we understand the overall logic for linear regression. lets move to example, we can utilize sklearn library to use the linear regression.

- Download jupyter notebook : https://jupyter.org/install
- install sklearn & numpy.
- notebook attached in repo.


