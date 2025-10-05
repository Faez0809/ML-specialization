# 📘 Lecture 31: Classification & Logistic Regression Motivation

This lecture introduces **classification problems** and explains why linear regression is inadequate for solving them.  
It sets the stage for the **logistic regression algorithm**.

---

## 🔹 1. Classification and Binary Classification

👉 In classification tasks, the output variable $y$ can only take one of a few discrete values — not any real number.

### 📨 Examples of Classification
- Is an email **spam**?  
- Is a transaction **fraudulent**?  
- Is a tumor **malignant**?

![](./images(w3)/31.1.png)

---

### 🟦 Binary Classification Terminology
- Called **binary classification** when $y$ can only be **two values**.
- Synonyms for categories:  
  - No / Yes  
  - False / True  
  - **0 / 1** (preferred in ML since computers handle numbers).

### ✅ Positive vs Negative Classes
- **Negative Class (0 / False):** Represents **absence**. (e.g., email not spam).  
- **Positive Class (1 / True):** Represents **presence**. (e.g., email is spam).  

⚠ Note: *Negative ≠ bad*, *Positive ≠ good*. They simply indicate **absence vs presence**.

---

## 🔹 2. Why Linear Regression Fails for Classification
![](./images(w3)/31.2.png)

### ⚡ Attempt: Linear Regression for Tumor Classification
- Input: tumor size ($x$)  
- Output: malignant (1) or benign (0)  

Fit a straight line:
$$
f(x) = w x + b
$$

#### Problems
1. Predicts values outside [0, 1].  
   - But we only need 0 or 1.  
2. Need to set a **threshold** (e.g., 0.5):  
   - If $f(x) < 0.5 \Rightarrow \hat{y} = 0$  
   - If $f(x) \geq 0.5 \Rightarrow \hat{y} = 1$  

![](./images(w3)/31.3.png)
---

### ⚠ Sensitivity to Outliers
- A single outlier (e.g., very large tumor) can **shift the line** drastically.  
- This moves the decision boundary (threshold = 0.5) → **wrong classifications**.  
- Shows linear regression is **unstable for classification**.

---

## 🔹 3. Logistic Regression Introduction

👉 Solution: **Logistic Regression**  
- Outputs are always between **0 and 1**.  
- Naturally fits **binary classification problems**.  
- Despite its name, it is **not regression**, but a **classification algorithm**.

✅ Next step: Learn logistic regression and why it works better than linear regression.

---
# 📘 Lecture 32 – Logistic Regression (Sigmoid Function & Interpretation)

This lecture introduces **logistic regression**, explains the **sigmoid (logistic) function**, and shows how the model output is interpreted as probabilities.

---

## 1️⃣ Introduction to Logistic Regression

- Logistic Regression is **one of the most widely used classification algorithms** in the world.  
- Commonly applied to tasks like:
  - Tumor classification (malignant vs benign 🧬)  
  - Spam detection (spam or not 📧)  
  - Fraud detection (fraudulent vs safe 💳)

⚡ Unlike Linear Regression:
- Logistic regression **always outputs values between 0 and 1**, making it perfect for classification.  
- The curve fitted is **S-shaped (sigmoid curve)**.

📌 **Example:**  
If the algorithm outputs **0.7** for a given tumor size → There is a **70% probability** the tumor is malignant.

---

## 2️⃣ The Sigmoid Function (Logistic Function) ➰

The **sigmoid function** is the core of logistic regression.

**Formula (use block math with `$$` on their own lines):**

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

where:
- $z$ = input to the function (from a linear combination of features and weights)  
- $e \approx 2.7$ is Euler’s number  

### 🔎 Behavior of the Sigmoid Function
- If $z \to +\infty$, then $g(z) \to 1$  
- If $z \to -\infty$, then $g(z) \to 0$  
- At $z = 0$, $g(0) = 0.5$


✅ Outputs are **probabilities between 0 and 1**.

![Sigmoid Function](./images%28w3%29/32.1.png)

---

## 3️⃣ Building the Logistic Regression Model ⚙️

**Step 1: Linear function**

$$
z = w \cdot x + b
$$

**Step 2: Apply sigmoid**

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

**Final model**

$$
f(x) = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$

![Logistic Regression Model](./images%28w3%29/32.2.png)

---

## 4️⃣ Interpretation of Logistic Regression Output 🎯

Treat \(f(x)\) as a **probability**:

$$
f(x) = P(y = 1 \mid x; w, b)
$$

### 🧪 Example: Tumor Classification
- Input \(x\) = tumor size  
- If \(f(x) = 0.7\) → **70% chance** malignant (class 1)  
- Then benign (class 0) = **0.3**  

Also:

$$
P(y=0) + P(y=1) = 1
$$

![Interpretation](./images%28w3%29/32.3.png)

---

## ✅ Summary

- Logistic regression uses the **sigmoid** to map predictions to \([0,1]\).  
- Outputs are **probabilities** (e.g., 0.3 = 30%, 0.7 = 70%).  
- More suitable than linear regression for classification tasks.  
- Widely used in practice (ads, medical diagnosis, fraud detection).

---
# 📘 Lecture 33 – Logistic Regression (Decision Boundary)

This lecture explains the **decision boundary** in logistic regression: how the probabilistic output is converted into a binary prediction (0 or 1), and how more complex decision boundaries can be formed using polynomial features.

---

## 1️⃣ Recap: Logistic Regression Output

**Step 1 – Linear part:**

$$
z = \vec{w} \cdot \vec{x} + b
$$

**Step 2 – Apply Sigmoid:**

$$
f_{\vec{w},b}(x) = g(z) = \frac{1}{1 + e^{-z}}
$$

- $f_{\vec{w},b}(x)$ = probability that $y=1$ (positive class).  
- Example: if $f(x) = 0.7$ → **70% chance** that $y=1$.  


![](./images(w3)/33.1.png)

---

## 2️⃣ Threshold for Prediction

To make a binary decision:

- If $f(x) \geq 0.5$ → predict $\hat{y} = 1$  
- If $f(x) < 0.5$ → predict $\hat{y} = 0$  

Since the sigmoid crosses $0.5$ at $z = 0$:

$$
f(x) \geq 0.5 \;\;\Longleftrightarrow\;\; z \geq 0 \;\;\Longleftrightarrow\;\; \vec{w} \cdot \vec{x} + b \geq 0
$$

 


---

## 3️⃣ The Decision Boundary (Linear Case)

The **decision boundary** is the line (or surface in higher dimensions) where $f(x) = 0.5$, i.e., where:

$$
\vec{w} \cdot \vec{x} + b = 0
$$

This separates the space into regions predicting $y=0$ and $y=1$.

### Example with Two Features

If parameters are $w_1 = 1$, $w_2 = 1$, $b = -3$:

$$
z = w_1 x_1 + w_2 x_2 + b = x_1 + x_2 - 3
$$  

So the decision boundary is:

$$
x_1 + x_2 = 3
$$  

- Region **$x_1 + x_2 \geq 3$** → predict $\hat{y} = 1$  
- Region **$x_1 + x_2 < 3$** → predict $\hat{y} = 0$  

![](./images(w3)/33.2.png) 


---

## 4️⃣ Non-Linear Decision Boundaries

If we include **polynomial features**, the boundary can become **curved**.

### Example: Circular Decision Boundary

Let:

$$
z = w_1 x_1^2 + w_2 x_2^2 + b
$$

If $w_1 = 1$, $w_2 = 1$, $b = -1$:

$$
x_1^2 + x_2^2 - 1 = 0
$$

This is the equation of a **circle** with radius 1.  

- Inside circle ($x_1^2 + x_2^2 < 1$) → predict $\hat{y}=0$  
- Outside circle ($x_1^2 + x_2^2 \geq 1$) → predict $\hat{y}=1$  

![](./images(w3)/33.3.png) 


---

## 5️⃣ More Complex Boundaries

By including higher-order polynomial terms like $x_1^2$, $x_1 x_2$, $x_2^3$, etc., logistic regression can model **ellipses** and **irregular shapes** as decision boundaries.
![](./images(w3)/33.4.png)
---

## ✅ Summary

- Logistic regression predicts probabilities with the sigmoid function.  
- A **threshold** (commonly 0.5) converts probabilities into binary outputs.  
- The **decision boundary** is where $\vec{w} \cdot \vec{x} + b = 0$.  
- With polynomial features, logistic regression can capture **non-linear decision boundaries**.  

---

# 📘 Lecture 34 – Cost Function for Logistic Regression

This lecture explains why squared error is not used for logistic regression and shows the correct **logistic loss**.

---

## 1️⃣ Training set setup

- Each example has features (e.g., tumor size, age) and a binary label.
- Label \(y \in \{0,1\}\).

**Model**

$$
f_{\mathbf{w},b}(x)=\frac{1}{1+e^{-(\mathbf{w}^{\top}\mathbf{x}+b)}}
$$

 
![](./images(w3)/34.1.png)

---

## 2️⃣ Why not squared error?

Squared error (good for linear regression):

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}\big(f(x^{(i)})-y^{(i)}\big)^2
$$

- ✅ Linear regression → **convex** cost (easy for gradient descent).
- ❌ Logistic regression → **non-convex** cost (many bumps).

 
![](./images(w3)/34.2.png)

---

## 3️⃣ Logistic loss (correct choice)

For one example \((x^{(i)},y^{(i)})\) with \(f^{(i)}=f_{\mathbf{w},b}(x^{(i)})\):

**Piecewise form**

$$
L(f^{(i)},y^{(i)})=
\begin{cases}
-\log\!\big(f^{(i)}\big), & \text{if } y^{(i)}=1 \\[6pt]
-\log\!\big(1-f^{(i)}\big), & \text{if } y^{(i)}=0
\end{cases}
$$

**Single-line form (handy for code and notes)**

$$
L(f^{(i)},y^{(i)})=-\,y^{(i)}\log\!\big(f^{(i)}\big)-\big(1-y^{(i)}\big)\log\!\big(1-f^{(i)}\big)
$$

📌 Screenshots:  
![](./images(w3)/34.3.png)  
![](./images(w3)/34.4.png)

**Intuition (simple):**
- If \(y=1\) → want \(f\approx1\) → loss small; \(f\approx0\) → loss huge.  
- If \(y=0\) → want \(f\approx0\) → loss small; \(f\approx1\) → loss huge.

---

## 4️⃣ Overall cost (average loss)

$$
J(\mathbf{w},b)=\frac{1}{m}\sum_{i=1}^{m}L\!\big(f^{(i)},y^{(i)}\big),
\quad \text{where } f^{(i)}=f_{\mathbf{w},b}(x^{(i)})
$$

This choice makes the cost **convex** for logistic regression → gradient descent works well.

  
![](./images(w3)/34.5.png)

---

## ✅ Summary

- Squared error → non-convex for logistic regression.  
- Use **logistic loss**: \( -y\log(f) -(1-y)\log(1-f) \).  
- Final cost \(J\) = average loss over all examples.  
- Train \(\mathbf{w},b\) by minimizing \(J\) with gradient descent.
