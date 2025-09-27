# 📘 Lecture 09: Linear Regression With one variable (Linear R. Model part 1)


## I. Introduction to Linear Regression
- **Linear Regression Model**: Fits a **straight line** to the data.  
- **Why Important?**  
  - One of the **most widely used ML algorithms** today.  
  - Builds foundation: concepts here apply to many other ML models.  

👉 **Key Idea**: Predict outcomes by drawing a line that best fits the training data.

---

## II. Example: Predicting House Prices 🏠

- **Dataset**: House sizes (sq. ft.) vs. house prices ($ in thousands).  
- **Problem**: Estimate price of a **1250 sq. ft. house**.  
- **Solution**: Use linear regression to draw the best-fit line.  
- **Prediction**: The line gives an approximate price of **$220,000**.  


![Graph](./images/Graph.png)

---

## III. Regression vs. Classification (Supervised Learning Context)

| Aspect | Regression | Classification |
|--------|------------|----------------|
| **Output Type** | Numbers (continuous, infinite possibilities). Example: predicting house price = $220,000. | Categories (finite, discrete). Example: cat 🐱 or dog 🐶, or 1 of 10 diseases. |
| **Model Predicts** | Real values (e.g., $1.5, -33.2, 220K). | Classes (e.g., {0, 1}, or {disease A, disease B, disease C}). |
| **Example** | Predict house price from size. | Predict whether tumor is malignant or benign. |

👉 Regression = **numbers**, Classification = **categories**.

---

## IV. Training Data Visualization

### 📊 Graph View
- Each cross (X) = 1 house (size, price).  
- Line = best-fit regression line.

### 📋 Data Table View
| Size (sq. ft.) | Price ($1000’s) |
|----------------|-----------------|
| 2104           | 400             |
| 1416           | 232             |
| 1534           | 315             |
| 852            | 178             |
| ...            | ...             |
| 3210           | 870             |

![Graph with table](./images/Graph%20with%20table.png)

---

## V. Standard Machine Learning Notation ✍️

![Terminology](./images/Terminology.png)
- **Training Set**: Data used to train the model.  
- **Input Variable ($x$)**: Feature (e.g., size of house).  
- **Output Variable ($y$)**: Target (e.g., price).  
- **Number of Training Examples ($m$)**: Size of dataset (e.g., $m=47$).  
- **Single Training Example**: $(x, y)$.  
- **$i$-th Training Example**:  
  - $(x^{(i)}, y^{(i)})$  
  - Example: $(2104, 400)$.  
  - **Note:** Superscript $i$ is **index**, not exponentiation.  

---

## VI. Key Takeaways
- Linear regression is the **first supervised learning algorithm** studied.  
- It predicts **continuous values** (numbers).  
- Uses training examples (with "right answers") to learn.  
- Clear notation ($x$, $y$, $m$, $(x^{(i)}, y^{(i)})$) is crucial for ML.

---

## VII. What’s Next?
- Learn how to take the training set and **feed it into the algorithm**.  
- The algorithm will then **learn patterns from data**.

<div style="margin-top:200px;"></div>


# 📘 Lecture 10: Linear Regression With one variable (Linear R. Model part 2)

## I. The Supervised Learning Process
- **Training Set** → fed to the learning algorithm.  
  - Includes:
    - **Input features ($x$)** (e.g., house size).  
    - **Output targets ($y$)** (e.g., house price).  
  - Targets are the **“right answers”** used to train the model.  

- **Algorithm Output**: Produces a function $f$  
  - Historically called a **hypothesis**.  
  - Now more commonly called the **model**.  

- **Prediction**:  
  - Function $f$ takes a new input $x$ and outputs an **estimate $\hat{y}$**.  
  - $\hat{y}$ is the **predicted value**, while $y$ is the **true target value**.  

![supervised learning process](./images/f.png)

---

## II. Essential Notation for Prediction
- **$\hat{y}$ (y-hat)** → The prediction (estimated $y$).  
- **$y$ (target)** → The true value from training data.  
- **$x$ (input feature)** → The input value (e.g., house size).  
- **Estimate vs True Value**:  
  - Example: Client selling a house.  
  - **True price ($y$)** unknown until sold.  
  - Model provides a **prediction ($\hat{y}$)** instead.  

---

## III. Representing the Function $f$ (Linear Regression)
- Start with a **straight-line representation**.  
- Formula:  
  $$
  f_{W,B}(x) = W \times x + B
  $$

- **Parameters**:  
  - $W$ (weight/slope).  
  - $B$ (bias/intercept).  
  - Together, they determine the value of $\hat{y}$.  

- **Notation**:  
  - $f_{W,B}(x)$ → explicit about parameters.  
  - $f(x)$ → shorthand, same meaning.  

---

## IV. Model Classification
- **Linear Regression Model** = straight-line model.  
- **With one variable** → called **Univariate Linear Regression**.  
  - "Uni" = one.  
  - Input = a single feature (e.g., house size).  

👉 *Later*: We’ll extend to **multiple features** (e.g., size, bedrooms, location).  

---

## V. Visualizing the Model
- Plot:  
  - $x$ (size) on horizontal axis.  
  - $y$ (price) on vertical axis.  
- Linear regression draws a **best-fit line** through training data.  
- Function: $f_{W,B}(x) = Wx + B$ → used for predictions.  

👉 *Insert figure here (Line fitting house price data)*

---

## VI. Practical Next Steps (Optional Lab)
- **Optional lab** → define a straight-line function in Python.  
- Tasks:  
  1. Choose values for $W$ and $B$.  
  2. See how well the line fits the data.  
- No coding required; just run and explore.  

---

## VII. Transition to the Cost Function
- To train the model properly, we need a **cost function**.  
- Cost function = **core concept in ML**.  
  - Used in **linear regression**.  
  - Also in **many advanced AI models**.  

👉 The next lecture will explain **how to construct the cost function**.
