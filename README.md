#!/usr/bin/env python3
"""
House Price Prediction using Linear Regression
Dataset: Housing Prices Dataset from Kaggle
URL: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Colab-compatible styling
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
    sns.set_style("whitegrid")

sns.set_palette("husl")

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_data(self):
        """
        Load and create synthetic housing dataset
        (In real scenario, you would load from Kaggle CSV file)
        """
        # Creating synthetic data similar to the Kaggle housing dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        square_footage = np.random.normal(2000, 800, n_samples)
        square_footage = np.clip(square_footage, 500, 5000)
        
        bedrooms = np.random.poisson(3, n_samples) + 1
        bedrooms = np.clip(bedrooms, 1, 6)
        
        bathrooms = np.random.poisson(2, n_samples) + 1
        bathrooms = np.clip(bathrooms, 1, 4)
        
        # Generate price with realistic relationships
        # Price increases with square footage, bedrooms, and bathrooms
        price = (
            square_footage * 120 +  # $120 per sq ft
            bedrooms * 15000 +      # $15k per bedroom
            bathrooms * 10000 +     # $10k per bathroom
            np.random.normal(0, 25000, n_samples)  # Random noise
        )
        price = np.clip(price, 100000, 1000000)  # Realistic price range
        
        # Create DataFrame
        data = pd.DataFrame({
            'square_footage': square_footage,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'price': price
        })
        
        return data
    
    def explore_data(self, data):
        """Perform exploratory data analysis"""
        print("=" * 60)
        print("HOUSING DATASET EXPLORATION")
        print("=" * 60)
        print(f"Dataset shape: {data.shape}")
        print(f"\nBasic Statistics:")
        print(data.describe())
        
        print(f"\nDataset Info:")
        print(data.info())
        
        # Check for missing values
        print(f"\nMissing Values:")
        print(data.isnull().sum())
        
        # Correlation matrix
        print(f"\nCorrelation Matrix:")
        correlation_matrix = data.corr()
        print(correlation_matrix)
        
        return correlation_matrix
    
    def visualize_data(self, data, correlation_matrix):
        """Create visualizations for data exploration"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Housing Data Exploration', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0, 0].hist(data['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Square footage vs Price
        axes[0, 1].scatter(data['square_footage'], data['price'], alpha=0.6, color='green')
        axes[0, 1].set_title('Square Footage vs Price')
        axes[0, 1].set_xlabel('Square Footage')
        axes[0, 1].set_ylabel('Price ($)')
        
        # 3. Bedrooms vs Price (boxplot)
        sns.boxplot(data=data, x='bedrooms', y='price', ax=axes[0, 2])
        axes[0, 2].set_title('Price by Number of Bedrooms')
        axes[0, 2].set_xlabel('Number of Bedrooms')
        axes[0, 2].set_ylabel('Price ($)')
        
        # 4. Bathrooms vs Price (boxplot)
        sns.boxplot(data=data, x='bathrooms', y='price', ax=axes[1, 0])
        axes[1, 0].set_title('Price by Number of Bathrooms')
        axes[1, 0].set_xlabel('Number of Bathrooms')
        axes[1, 0].set_ylabel('Price ($)')
        
        # 5. Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Heatmap')
        
        # 6. Square footage distribution
        axes[1, 2].hist(data['square_footage'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 2].set_title('Square Footage Distribution')
        axes[1, 2].set_xlabel('Square Footage')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_features(self, data):
        """Prepare features and target variables"""
        # Features (independent variables)
        X = data[['square_footage', 'bedrooms', 'bathrooms']]
        # Target (dependent variable)
        y = data['price']
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the linear regression model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features for better performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Store for evaluation
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.y_train_pred, self.y_test_pred = y_train_pred, y_test_pred
        
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        
        print("=" * 60)
        print("MODEL PERFORMANCE EVALUATION")
        print("=" * 60)
        print(f"Training Set Performance:")
        print(f"  Mean Squared Error (MSE): ${train_mse:,.2f}")
        print(f"  Root Mean Squared Error (RMSE): ${np.sqrt(train_mse):,.2f}")
        print(f"  Mean Absolute Error (MAE): ${train_mae:,.2f}")
        print(f"  R² Score: {train_r2:.4f}")
        
        print(f"\nTesting Set Performance:")
        print(f"  Mean Squared Error (MSE): ${test_mse:,.2f}")
        print(f"  Root Mean Squared Error (RMSE): ${np.sqrt(test_mse):,.2f}")
        print(f"  Mean Absolute Error (MAE): ${test_mae:,.2f}")
        print(f"  R² Score: {test_r2:.4f}")
        
        # Model coefficients
        feature_names = ['Square Footage', 'Bedrooms', 'Bathrooms']
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        print(f"\nModel Coefficients:")
        print(f"  Intercept: ${intercept:,.2f}")
        for name, coef in zip(feature_names, coefficients):
            print(f"  {name}: ${coef:,.2f}")
        
        return {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae
        }
    
    def visualize_results(self):
        """Visualize model results"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Visualization', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted (Training)
        axes[0, 0].scatter(self.y_train, self.y_train_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                       [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_title('Training Set: Actual vs Predicted Prices')
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        
        # 2. Actual vs Predicted (Testing)
        axes[0, 1].scatter(self.y_test, self.y_test_pred, alpha=0.6, color='green')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_title('Testing Set: Actual vs Predicted Prices')
        axes[0, 1].set_xlabel('Actual Price ($)')
        axes[0, 1].set_ylabel('Predicted Price ($)')
        
        # 3. Residuals (Training)
        train_residuals = self.y_train - self.y_train_pred
        axes[1, 0].scatter(self.y_train_pred, train_residuals, alpha=0.6, color='blue')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Training Set: Residual Plot')
        axes[1, 0].set_xlabel('Predicted Price ($)')
        axes[1, 0].set_ylabel('Residuals ($)')
        
        # 4. Residuals (Testing)
        test_residuals = self.y_test - self.y_test_pred
        axes[1, 1].scatter(self.y_test_pred, test_residuals, alpha=0.6, color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Testing Set: Residual Plot')
        axes[1, 1].set_xlabel('Predicted Price ($)')
        axes[1, 1].set_ylabel('Residuals ($)')
        
        plt.tight_layout()
        plt.show()
    
    def predict_price(self, square_footage, bedrooms, bathrooms):
        """Predict price for new house"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        
        # Create feature array
        features = np.array([[square_footage, bedrooms, bathrooms]])
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        predicted_price = self.model.predict(features_scaled)[0]
        
        print(f"\nHOUSE PRICE PREDICTION")
        print(f"=" * 30)
        print(f"Square Footage: {square_footage:,}")
        print(f"Bedrooms: {bedrooms}")
        print(f"Bathrooms: {bathrooms}")
        print(f"Predicted Price: ${predicted_price:,.2f}")
        
        return predicted_price
    
    def feature_importance(self):
        """Display feature importance based on coefficients"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return
        
        feature_names = ['Square Footage', 'Bedrooms', 'Bathrooms']
        coefficients = np.abs(self.model.coef_)  # Absolute values for importance
        
        # Create importance plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(feature_names, coefficients, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Feature Importance (Absolute Coefficients)', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Absolute Coefficient Value')
        
        # Add value labels on bars
        for bar, coef in zip(bars, coefficients):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(coefficients)*0.01,
                    f'${coef:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the house price prediction"""
    print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
    print("=" * 60)
    print("Dataset Source: Housing Prices Dataset from Kaggle")
    print("URL: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load data
    print("\n1. Loading dataset...")
    data = predictor.load_data()
    print("✓ Dataset loaded successfully!")
    
    # Explore data
    print("\n2. Exploring dataset...")
    correlation_matrix = predictor.explore_data(data)
    
    # Visualize data
    print("\n3. Creating data visualizations...")
    predictor.visualize_data(data, correlation_matrix)
    
    # Prepare features
    print("\n4. Preparing features...")
    X, y = predictor.prepare_features(data)
    print("✓ Features prepared!")
    
    # Train model
    print("\n5. Training linear regression model...")
    X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = predictor.train_model(X, y)
    print("✓ Model trained successfully!")
    
    # Evaluate model
    print("\n6. Evaluating model performance...")
    metrics = predictor.evaluate_model()
    
    # Visualize results
    print("\n7. Creating performance visualizations...")
    predictor.visualize_results()
    
    # Show feature importance
    print("\n8. Analyzing feature importance...")
    predictor.feature_importance()
    
    # Make sample predictions
    print("\n9. Making sample predictions...")
    sample_houses = [
        (1500, 2, 1),  # Small house
        (2500, 3, 2),  # Medium house
        (4000, 5, 3),  # Large house
    ]
    
    for sqft, beds, baths in sample_houses:
        predictor.predict_price(sqft, beds, baths)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Model interpretation
    print("\nMODEL INTERPRETATION:")
    print("- R² Score indicates how well the model explains price variance")
    print("- RMSE shows average prediction error in dollars")
    print("- Coefficients show the impact of each feature on price")
    print("- Residual plots help identify model assumptions")

if __name__ == "__main__":
    main()
