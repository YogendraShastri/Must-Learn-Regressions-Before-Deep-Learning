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
- notebook attached in repo [View Notebook](linear_regression.ipynb)

## What is Multiple Linear Regression ?
- It models the relationship between one dependent variable and two or more independent variables.

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$

- Or more compactly using vector notation:

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

- Here x1, x2, x3 ... are the features, like when you are predicting the price of the house, you consider many features, like size of the house, bedrooms, age of the house etc..
-  cost function and gradient decent will be the same, just but they are applied to more features in multiple regression.

### Example :
- notebook attached in repo [View Notebook](multiple_linear_regression.ipynb)

## What is Logistic Regression?
- Logistic regression is mainly used for classfication problem.
- Linear regression predicts numbers (like a price or temperature) and used for continuous data.
- While logistic regression predicts probabilities for categories (like yes/no or spam/not spam).

### Equation same as linear regression

$$
z = \mathbf{w}^\top \mathbf{x} + b
$$

### Where:
- 𝑥: input vector
- 𝑤: weight vector
- 𝑏: bias

you might ask when logistic regression starts with the same linear combination as linear regression, whats the critical difference. you are correct. but linear regression outout is unbounded it can be 1, Negative or any other positive number, which is not suitable for classification problem. You can’t confidently label "Spam or Not Spam" if output can be 7.2.
To solve this, logistic regression passes the linear output through a sigmoid function. so lets see the sigmoid function. 

### Sigmoid Function
Sigmoid function's output always falls between 0 and 1 (exclusive), making it suitable for representing probabilities. it will be more clear once we see the equation.

**Equation** :

$$
\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-z}}
$$

Where:

$$
z = \mathbf{w}^\top \mathbf{x} + b
$$

so we can see that we are dividing 1 with a value which is greater or equal to one, that means the sigmoid will always falls between 0 to 1. This is now interpretable as a probability.

### Loss function

$$
\text{Loss}(y_i, \hat{y}_i) = -\left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

**Where**
- 𝑦 = actual class (0 or 1)
- 𝑦 bar = predicted probability from sigmoid

### Cost Function

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} \text{Loss}(y_i, \hat{y}_i)
$$

Where:

$$
\text{Loss}(y_i, \hat{y}_i) = -\left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

### Derivation 

1. **Use the Sigmoid Function as the Model**
- The logistic regression model estimates the probability (P, 1-P).
- For a true label \(y\) (where \(y\) is either 0 or 1) and a predicted probability \(p\) of the positive class (i.e., \(y=1\)), the probability of observing the outcome \(y\) can be expressed as:
    - If \(y=1\), the probability is \(p\).
    - If \(y=0\), the probability is \((1-p)\).
- These two cases can be combined into a single expression using the **probability mass function (PMF)** of the **Bernoulli distribution**:

$$
P(Y = y) = p^{y}(1 - p)^{(1 - y)}
$$

2. **Define the Likelihood for the Dataset**

Given a dataset of \( n \) independent samples, where each sample \( i \) has:
- True label: \( yi \in \{0, 1\} \)
- A predicted probability 𝑝𝑖, which comes from the sigmoid function.

The **likelihood** of the entire dataset is:

$$
L(w, b) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}
$$

To simplify optimization, we take the logarithm of the likelihood:

3. **Log-Likelihood Function**
- The log-likelihood function for a single observation \(i\) becomes:

$$
\log \big( P(Y = y_i) \big) = y_i \log(p_i) + (1 - y_i) \log(1 - p_i)
$$

- For the entire dataset:
  
$$
\log L(w, b) = \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

4. **Negative log-likelihood as a loss function**
- To convert this to a loss, we use Negative Log-Likelihood:

$$
\text{Loss} = -\log L(w, b)
$$

- So the cost function (for all samples) becomes:

$$
J(w, b) = - \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

5. **Average to Get Cost Function**

$$
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Now we understand the overall logic for logistic regression. lets move to example, we can utilize sklearn library to use the logistic regression.

- notebook attached in repo [View Notebook](Logistic_regression_tut.ipynb)

## What Is Polynomial Regression?
- **Polynomial regression** is a type of regression analysis where the relationship between the independent variable (x) and the dependent variable (y) is modeled as an nth degree polynomial.
- Unlike **linear regression**, which fits a straight line, **polynomial regression** can fit curved lines, making it better suited for modeling **non-linear patterns** in data.

  <img width="734" height="482" alt="image" src="https://github.com/user-attachments/assets/3e751124-a9aa-4bab-b0e8-0f623b1ce94a" />

### Equation:

$$
y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \cdots + w_d x^d
$$

**Where**
- y = predicted value
- x = input feature
- w0 = bias term
- d = Degree of the polynomial
- w1,w2..wn =  Weights for each power of x

### Risk :
**Overfitting** : A potential risk with polynomial regression is overfitting, where the model fits the training data too closely, including noise, and performs poorly on new, unseen data.

### Loss Function and Cost Function
Loss Function: Similar to Linear Regression, Polynomial Regression also uses Mean Squared Error (MSE) as the loss function.

### Validation Metrics
R-squared is generally used to evaluate the performance of the polynomial regression. R-squared value ranges from 0 to 1, where 0 means no relationship and 1 means 100% matched relationship.
sklearn R-squared can be used to evaluate the result.

**Example**:
- notebook attached in repo [View Notebook](polynomial_regression_tut.ipynb)


## Bayesian Regression
- Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred.
- Bayes’ theorem is stated mathematically as the following equation:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

**Where:**
- \( P(A \ B) \) = Posterior probability (probability of event \( A \) given that \( B \) has occurred)  
- \( P(B \ A) \) = Likelihood (probability of event \( B \) given that \( A \) is true)  
- \( P(A) \) = Prior probability of event \( A \)  
- \( P(B) \) = Marginal probability of event \( B \)

### How formula Derived ?
- So in probability there are two major events : 
1. Independent Events
2. Dependent Events

**Independent Evetns** : In dependent events are those events which do not effect the other event, like if you role a dice, no matter in which order you role it, the probabiliy of coming a number (1,2,3,4,5 or 6) is always 1/6. these events are independent of each other.

**Dependent Events** : Vise versa, dependent events are those events which affect the other. for example, you have a box with 4 red balls and 3 green balls, so probability of taking red balls will be 4/7, and green will be 3/7. but once you drawn the red ball once the probability of taking green ball will be 3/6 i.e. 1/2. You see now the event is affected.

#### Now lets see the formula **Steps** ::
- We will consider probability to take red ball as P(A), taking for Green ball is P(B).
- So probabiliy to take green ball, after drawn red ball. i.e probability of Event B when Event A is already occured will be.
  P(B/A).

#### Step 1: Definition of Conditional Probability

From the definition of conditional probability:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \quad \text{(1)}
$$

And similarly:

$$
P(B \mid A) = \frac{P(A \cap B)}{P(A)} \quad \text{(2)}
$$

---

#### Step 2: Express \( P(A \ B) \) from Equation (2)

Rearranging Equation (2):

$$
P(A \cap B) = P(B \mid A) \cdot P(A)
$$

---

#### Step 3: Substitute into Equation (1)

Now plug this into Equation (1):

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

---

###  Bayes’ Theorem:

$$
\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}
$$

#### Types of Naive Bayes Classifier-
1. Multinomial Naive Bayes
2. Bernoulli Naive Bayes
3. Gaussian Naive Bayes

  





