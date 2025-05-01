import pandas as pd

# Load your CSV file
df = pd.read_csv('brain_stroke.csv')  # Replace with your actual file path

# Option 1: Interpolation (recommended for numerical data like 'bmi')
df['bmi'] = df['bmi'].interpolate(method='linear')

# Option 2 (alternative): Fill missing values forward then backward
# df['bmi'] = df['bmi'].fillna(method='ffill').fillna(method='bfill')

# Save cleaned file (optional)
df.to_csv('cleaned_brain_stroke.csv', index=False)
