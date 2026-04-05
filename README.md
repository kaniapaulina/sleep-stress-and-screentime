# sleep-stress-and-screentime

### how do your daily habbits define your health?

---

A collection of NumPy-based neural networks designed to analyze the impact of digital habits on mental well-being. Built from **scratch** without high-level ML libraries like TensorFlow or PyTorch.

---

## Models Included:

1. **Sleep Regressor**: Predicts the total hours of sleep based on digital load and stress levels.
    - Architecture: 3-layer MLP (Leaky ReLU).
    - Features: Includes custom interaction terms.

2. **Health Classifier**: Identifies high-risk mental health profiles (Binary Classification).
    - Architecture: 3-layer MLP (Sigmoid output).
    - Metric: Cross-Entropy Loss with Accuracy tracking.

## Technical Elements:

- He Initialization: Weights are scaled using $n_{in}$ to ensure stable gradients during the first few epochs.
- Vectorized Backpropagation: All partial derivatives ($dW$, $db$) are calculated using matrix dot products for high efficiency.
- Data Pipeline: Includes automated One-Hot Encoding for categorical variables (Gender, Location) and custom Min-Max scaling.

---

### Data

Source: [Kaggle - Impact of screentime on mental health](https://www.kaggle.com/datasets/khushikyad001/impact-of-screen-time-on-mental-health?fbclid=IwY2xjawQ9De1leHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeHtQUcS4Sf-yvCWPszcldwjyaYreAHy_vLcOWPGxGBA-CmURlIY2Mce1lVXs_aem_xcqXiQAV7pSzU2fuKBflNw)

---

### Quick Start
```Bash
git clone https://github.com/kaniapaulina/sleep-stress-and-screentime.git
cd sleep-stress-and-screentime
```
Ensure you have the dataset named digital_diet_mental_health.csv in the root folder.

```Bash
# To run the Sleep Regression model
python sleep_model.py

# To run the Mental Health Classification model
python health_model.py
```