#!/usr/bin/env python3

import time
import pandas as pd
import numpy as np
from hybrid_formula_engine import HybridFormulaEngine, Formula

def test_user_complex_expression():
    """Test the user's specific complex expression with the hybrid engine"""
    
    print("🎯 TESTING USER'S COMPLEX EXPRESSION WITH HYBRID ENGINE")
    print("=" * 70)
    
    formula = "True if IsNotNull(Source) and CCY.endswith('x') and TRN == ABC or Source.__contains__('A and B') else False"
    print(f"Formula: {formula}")
    
    # Test with various dataset sizes
    sizes = [1000, 10000, 100000, 400000]
    
    for size in sizes:
        print(f"\n📊 Testing with {size:,} rows:")
        print("-" * 40)
        
        # Create test data
        np.random.seed(42)
        data = {
            'Source': [
                f'Valid Source {i}' if i % 5 != 0 
                else f'A and B test {i}' if i % 10 == 0
                else None if i % 20 == 0
                else f'Source {i}'
                for i in range(size)
            ],
            'CCY': [f'USD{"x" if i % 3 == 0 else ""}' for i in range(size)],
            'TRN': ['ABC' if i % 2 == 0 else 'XYZ' for i in range(size)],
            'ABC': ['ABC'] * size
        }
        
        df = pd.DataFrame(data)
        
        # Test hybrid engine
        try:
            engine = HybridFormulaEngine(df.copy())
            
            start_time = time.perf_counter()
            result = engine.evaluate(formula, 'result', debug=True)
            hybrid_time = time.perf_counter() - start_time
            
            print(f"  ✅ Hybrid Engine: {hybrid_time:.4f}s ({size/hybrid_time:,.0f} rows/s)")
            print(f"  📊 True count: {result['result'].sum():,} / {len(result):,}")
            
        except Exception as e:
            print(f"  ❌ Hybrid Engine failed: {e}")

def test_numerical_expressions_scaling():
    """Test pure numerical expressions with different dataset sizes"""
    
    print(f"\n⚡ NUMERICAL EXPRESSIONS SCALING TEST")
    print("=" * 70)
    
    # Pure numerical formulas that should benefit from NumExpr
    numerical_formulas = [
        'price * quantity',
        'price * quantity * (1 + tax_rate)',
        'x + y + z',
        'x ** 2 + y ** 2',
        '(price - cost) * quantity',
    ]
    
    sizes = [10000, 100000, 1000000]
    
    for size in sizes:
        print(f"\n📊 Dataset size: {size:,} rows")
        print("-" * 40)
        
        # Create numerical test data
        np.random.seed(42)
        data = {
            'price': np.random.uniform(10.0, 1000.0, size),
            'quantity': np.random.randint(1, 100, size),
            'tax_rate': np.random.uniform(0.05, 0.25, size),
            'cost': np.random.uniform(5.0, 500.0, size),
            'x': np.random.uniform(1.0, 100.0, size),
            'y': np.random.uniform(1.0, 100.0, size),
            'z': np.random.uniform(1.0, 100.0, size),
        }
        
        df = pd.DataFrame(data)
        
        for formula in numerical_formulas[:2]:  # Test first 2 formulas
            print(f"\n  🧮 Formula: {formula}")
            
            try:
                engine = HybridFormulaEngine(df.copy())
                
                # Benchmark to see engine selection
                benchmark = engine.benchmark_expression(formula, iterations=3)
                
                if benchmark['recommendation']:
                    rec_engine = benchmark['recommendation']
                    speedup = benchmark.get('speedup', 1)
                    
                    print(f"    🎯 Best engine: {rec_engine}")
                    print(f"    ⚡ Speedup: {speedup:.2f}x")
                    
                    if 'numexpr' in benchmark['engines_tested'] and benchmark['engines_tested']['numexpr']['success']:
                        ne_time = benchmark['engines_tested']['numexpr']['avg_time']
                        rate = size / ne_time
                        print(f"    ⚡ NumExpr: {ne_time:.6f}s ({rate:,.0f} rows/s)")
                    
                    if 'optimized' in benchmark['engines_tested'] and benchmark['engines_tested']['optimized']['success']:
                        opt_time = benchmark['engines_tested']['optimized']['avg_time']
                        rate = size / opt_time
                        print(f"    🚀 Optimized: {opt_time:.6f}s ({rate:,.0f} rows/s)")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")

def test_hybrid_vs_individual_engines():
    """Compare hybrid engine against individual engines"""
    
    print(f"\n🏆 HYBRID vs INDIVIDUAL ENGINES COMPARISON")
    print("=" * 70)
    
    # Create comprehensive test data
    size = 100000
    np.random.seed(42)
    
    data = {
        # Numerical columns
        'price': np.random.uniform(10.0, 1000.0, size),
        'quantity': np.random.randint(1, 100, size),
        'x': np.random.uniform(1.0, 100.0, size),
        'y': np.random.uniform(1.0, 100.0, size),
        
        # String columns  
        'name': [f'user_{i}' for i in range(size)],
        'category': np.random.choice(['A', 'B', 'C'], size),
        
        # Mixed columns
        'score': np.random.randint(1, 1000, size),
        'active': np.random.choice([True, False], size),
    }
    
    df = pd.DataFrame(data)
    
    test_formulas = [
        # Pure numerical (should favor NumExpr on large data)
        ('price * quantity', 'Numerical'),
        ('x + y', 'Numerical'),
        
        # Mixed operations (should favor optimized)
        ('"High" if score > 500 else "Low"', 'Conditional'),
        ('name.upper()', 'String'),
        
        # Complex operations (should favor optimized)
        ('price * 1.1 if active else price', 'Mixed Logic'),
    ]
    
    print(f"Dataset: {size:,} rows")
    
    for formula, formula_type in test_formulas:
        print(f"\n🧪 {formula_type}: {formula}")
        print("-" * 50)
        
        # Test with hybrid engine
        try:
            hybrid_engine = HybridFormulaEngine(df.copy())
            
            start_time = time.perf_counter()
            result_hybrid = hybrid_engine.evaluate(formula, 'result')
            hybrid_time = time.perf_counter() - start_time
            
            print(f"  🔗 Hybrid Engine: {hybrid_time:.6f}s ({size/hybrid_time:,.0f} rows/s)")
            
            # Show which engine was actually used
            is_numerical = hybrid_engine._is_numerical_expression(formula)
            print(f"    Detected as numerical: {is_numerical}")
            
        except Exception as e:
            print(f"  ❌ Hybrid Engine failed: {e}")
            continue

def showcase_intelligent_selection():
    """Showcase the intelligent engine selection in action"""
    
    print(f"\n🧠 INTELLIGENT ENGINE SELECTION SHOWCASE")
    print("=" * 70)
    
    # Create test data
    size = 50000
    np.random.seed(42)
    
    data = {
        'price': np.random.uniform(10.0, 1000.0, size),
        'quantity': np.random.randint(1, 100, size),
        'tax_rate': np.random.uniform(0.05, 0.25, size),
        'name': [f'product_{i}' for i in range(size)],
        'category': np.random.choice(['Electronics', 'Books', 'Clothing'], size),
        'score': np.random.randint(1, 1000, size),
        'active': np.random.choice([True, False], size),
    }
    
    df = pd.DataFrame(data)
    
    test_cases = [
        {
            'formula': 'price * quantity',
            'expected_engine': 'NumExpr',
            'reason': 'Pure numerical multiplication'
        },
        {
            'formula': 'price * quantity * (1 + tax_rate)',
            'expected_engine': 'NumExpr', 
            'reason': 'Complex numerical expression'
        },
        {
            'formula': 'name.upper()',
            'expected_engine': 'Optimized',
            'reason': 'String method operation'
        },
        {
            'formula': '"Premium" if score > 800 else "Standard"',
            'expected_engine': 'Optimized',
            'reason': 'Conditional string logic'
        },
        {
            'formula': 'price * 1.1 if active else price',
            'expected_engine': 'Optimized',
            'reason': 'Mixed numerical and boolean logic'
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        formula = test_case['formula']
        expected = test_case['expected_engine']
        reason = test_case['reason']
        
        print(f"\n{i}. {formula}")
        print(f"   Expected: {expected} ({reason})")
        
        try:
            engine = HybridFormulaEngine(df.copy())
            
            # Check what the engine detects
            is_numerical = engine._is_numerical_expression(formula)
            predicted_engine = "NumExpr" if is_numerical and engine.numexpr_available else "Optimized"
            
            print(f"   Detected: {predicted_engine} ({'✅' if predicted_engine == expected else '❌'})")
            
            # Benchmark to see actual performance
            benchmark = engine.benchmark_expression(formula, iterations=2)
            
            if benchmark['recommendation']:
                actual_best = benchmark['recommendation'].capitalize()
                speedup = benchmark.get('speedup', 1)
                print(f"   Best performance: {actual_best} ({speedup:.2f}x speedup)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🔗 COMPREHENSIVE HYBRID ENGINE TESTING")
    print("Intelligently combines NumExpr + Optimized engines")
    print("=" * 70)
    
    # Test user's complex expression
    test_user_complex_expression()
    
    # Test numerical expressions scaling
    test_numerical_expressions_scaling()
    
    # Compare hybrid vs individual engines
    test_hybrid_vs_individual_engines()
    
    # Showcase intelligent selection
    showcase_intelligent_selection()
    
    print(f"\n🎉 COMPREHENSIVE TESTING COMPLETE!")
    print("=" * 70)