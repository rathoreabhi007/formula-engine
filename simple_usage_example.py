#!/usr/bin/env python3
"""
Simple usage example of the Hybrid Formula Engine
Shows the minimal code needed to use the engine with your DataFrame and formula
"""

import pandas as pd
import numpy as np
from hybrid_formula_engine import Formula  # This is all you need to import!

def main():
    print("🚀 SIMPLE HYBRID ENGINE USAGE")
    print("=" * 50)
    
    # Example 1: Your DataFrame and formula
    print("\n1. Basic Usage Example:")
    print("-" * 30)
    
    # Create your DataFrame (replace this with your actual data)
    df = pd.DataFrame({
        'Source': ['Valid Source 1', 'A and B test', None, 'Source 4'],
        'CCY': ['USDx', 'EUR', 'USDx', 'GBP'],
        'TRN': ['ABC', 'XYZ', 'ABC', 'ABC'],
        'ABC': ['ABC', 'ABC', 'ABC', 'ABC'],
        'price': [100.5, 200.0, 150.75, 300.0],
        'quantity': [2, 3, 1, 4]
    })
    
    print("Your DataFrame:")
    print(df)
    
    # Your complex formula
    formula = 'True if IsNotNull(Source) and CCY.endswith("x") and TRN == ABC or Source.__contains__("A and B") else False'
    
    print(f"\nYour formula: {formula}")
    
    # THIS IS ALL YOU NEED - 2 lines of code!
    engine = Formula(df)  # Initialize with your DataFrame
    result = engine.evaluate(formula, 'result')  # Evaluate your formula
    
    print(f"\nResult:")
    print(result[['Source', 'CCY', 'TRN', 'result']])
    
    # Example 2: Different formula types
    print(f"\n2. Different Formula Types:")
    print("-" * 30)
    
    formulas_to_test = [
        ('price * quantity', 'NumExpr will be used'),
        ('price * 1.1 if quantity > 2 else price', 'Optimized engine will be used'),
        ('Source.upper() if IsNotNull(Source) else "NULL"', 'Optimized engine will be used')
    ]
    
    for formula, explanation in formulas_to_test:
        print(f"\nFormula: {formula}")
        print(f"Expected: {explanation}")
        
        # Same simple usage pattern
        engine = Formula(df.copy())
        result = engine.evaluate(formula, 'output')
        print(f"Result: {result['output'].tolist()}")

def production_usage_pattern():
    """Production-ready usage pattern"""
    
    print(f"\n\n🏭 PRODUCTION USAGE PATTERN")
    print("=" * 50)
    
    # Simulate your real data
    df = pd.DataFrame({
        'Source': ['Data1', 'A and B match', None] * 1000,
        'CCY': ['USDx', 'EUR', 'GBPx'] * 1000,
        'TRN': ['ABC', 'XYZ', 'ABC'] * 1000,
        'ABC': ['ABC'] * 3000,
        'price': np.random.uniform(10, 1000, 3000),
        'quantity': np.random.randint(1, 10, 3000)
    })
    
    print(f"Dataset size: {len(df):,} rows")
    
    # Your actual formula
    your_formula = 'True if IsNotNull(Source) and CCY.endswith("x") and TRN == ABC or Source.__contains__("A and B") else False'
    
    # Production usage with error handling
    try:
        # Initialize engine
        engine = Formula(df)
        
        # Evaluate formula
        import time
        start_time = time.perf_counter()
        result = engine.evaluate(your_formula, 'validation_result')
        end_time = time.perf_counter()
        
        # Show results
        execution_time = end_time - start_time
        rows_per_second = len(df) / execution_time
        
        print(f"✅ Success!")
        print(f"⏱️  Execution time: {execution_time:.4f} seconds")
        print(f"🚀 Performance: {rows_per_second:,.0f} rows/second")
        print(f"✔️  True count: {result['validation_result'].sum():,}")
        print(f"❌ False count: {(~result['validation_result']).sum():,}")
        
        # Your result DataFrame is ready to use
        # result now contains all original columns + 'validation_result'
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
    production_usage_pattern()
    
    print(f"\n💡 SUMMARY:")
    print("Just 2 lines of code:")
    print("  engine = Formula(df)")
    print("  result = engine.evaluate(formula, 'output_column')")
    print("\n🎯 That's it! The hybrid engine handles everything automatically.")