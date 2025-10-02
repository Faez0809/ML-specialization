# 📘 Multiple Linear Regression  

This lecture explains **Multiple Linear Regression**, which allows us to use **multiple features (inputs)** to make better predictions compared to using only one feature.  
It also introduces **notations** 📑 and a **compact vectorized form** ⚡.

---

## 1️⃣ From Single Feature to Multiple Features

### 🔹 Single Feature Model (Univariate Regression)
\[
f_{w,b}(x) = wx + b
\]

- Uses only **one input feature**.  
- Example: Predicting house price using only **size of the house (sq. feet)**.

📌 *Insert Screenshot Here ➡️ (Single feature table)*

---

### 🔹 Multiple Features Model (Multiple Linear Regression)
- Uses **several input features** → more accurate predictions.  
- Example: House price prediction using:  
  1. 🏠 Size of the house (\(x_1\))  
  2. 🛏️ Number of bedrooms (\(x_2\))  
  3. 🏢 Number of floors (\(x_3\))  
  4. 📅 Age of the home in years (\(x_4\))  

📌 *Insert Screenshot Here ➡️ (Multiple features table)*

---

## 2️⃣ Notation for Multiple Features ✍️

To handle multiple inputs, we use structured notation:

| Symbol | Meaning | Example (if \(n=4\)) |
|--------|---------|-----------------------|
| \(n\) | Number of features | \(n=4\) |
| \(x_j\) | The \(j^{th}\) feature | \(x_1\) = size, \(x_4\) = age |
| \(\vec{x}\) | Vector of all features | \([x_1, x_2, x_3, x_4]\) |
| \(x^{(i)}\) | The \(i^{th}\) training example (a vector) | \(x^{(2)} = [1416, 3, 2, 40]\) |
| \(x^{(i)}_j\) | Value of the \(j^{th}\) feature in the \(i^{th}\) example | \(x^{(2)}_3 = 2\) |

📌 *Insert Screenshot Here ➡️ (Notation with training examples)*

---

## 3️⃣ The Multiple Linear Regression Model 🧮

The model learns **parameters (weights)** for each feature + a bias term.

### 🔹 Expanded Form
\[
f_{\vec{w}, b}(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b
\]

For \(n\) features:
\[
f_{\vec{w}, b}(\vec{x}) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
\]

---

### 🔹 Example with House Prices
\[
f_{\vec{w}, b}(\vec{x}) = 0.1x_1 + 4x_2 + 10x_3 - 2x_4 + 80
\]

- 📏 **0.1** → Each extra square foot adds **$100**  
- 🛏️ **4** → Each bedroom adds **$4000**  
- 🏢 **10** → Each floor adds **$10,000**  
- 📉 **-2** → Each year reduces price by **$2000**  
- 💵 **80** → Base price = **$80,000**  

📌 *Insert Screenshot Here ➡️ (Model with example equation)*

---

## 4️⃣ Vectorized Notation (Compact Form) ⚡

Instead of writing many terms, we use the **dot product**:

\[
f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b
\]

- \(\vec{w} = [w_1, w_2, \dots, w_n]\) → parameters (weights)  
- \(\vec{x} = [x_1, x_2, \dots, x_n]\) → features  

### 🔹 Dot Product Definition:
\[
\vec{w} \cdot \vec{x} = w_1x_1 + w_2x_2 + \dots + w_nx_n
\]

✔ Compact form = **shorter, cleaner, faster** to compute.  

📌 *Insert Screenshot Here ➡️ (Dot product vector form)*

---

## 5️⃣ Important Terminology 📖

- **Multiple Linear Regression** → Linear regression with **multiple input features**.  
- **Univariate Regression** → Linear regression with **one feature only**.  
- **Multivariate Regression** → Different concept (predicting multiple outputs), not covered here.  

---

## 6️⃣ Key Insights 🌟

✅ Adding more features improves prediction accuracy.  
✅ Each weight (\(w_j\)) shows how much that feature affects the result.  
✅ Positive weight → increases output, Negative weight → decreases output.  
✅ Vectorization makes the model **easier** and **faster** to implement in code.  

---

## 🚀 Next Step
Learn about **vectorization techniques** to compute predictions for **many examples at once** efficiently.

---
