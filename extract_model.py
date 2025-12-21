import sys
import pickle
sys.path.append('src/models')
from baselines import LinearRegressionBaseline

# Load the wrapped model
with open('models/baselines/linear_regression_baseline.pkl', 'rb') as f:
    model_wrapper = pickle.load(f)

# Extract and save the sklearn model directly
with open('models/baselines/linear_model_sklearn.pkl', 'wb') as f:
    pickle.dump(model_wrapper.model, f)

print('✅ Sklearn LinearRegression model extracted and saved!')
print(f'Model type: {type(model_wrapper.model)}')
