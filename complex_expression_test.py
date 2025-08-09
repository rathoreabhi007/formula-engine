#!/usr/bin/env python3

import time
import pandas as pd
import numpy as np
from fo import Formula as OriginalFormula
from fo_optimized import FormulaOptimized
import numexpr as ne

def test_your_complex_expression():
    """Test the specific complex expression mentioned by the user"""
    
    print("🎯 TESTING YOUR COMPLEX EXPRESSION")
    print("=" * 70)
    
    # Your specific formula
    formula = "True if IsNotNull(Source) and CCY.endswith('x') and TRN == ABC or Source.__contains__('A and B') else False"
    
    print(f"Formula: {formula}")
    print()
    
    # Create 400K test data that matches your use case
    rows = 400000
    print(f"🔧 Generating {rows:,} rows of test data...")
    
    np.random.seed(42)
    data = {
        'Source': [
            f'Valid Source {i}' if i % 5 != 0 
            else f'A and B test {i}' if i % 10 == 0
            else None if i % 20 == 0
            else f'Source {i}'
            for i in range(rows)
        ],
        'CCY': [f'USD{"x" if i % 3 == 0 else ""}' for i in range(rows)],
        'TRN': ['ABC' if i % 2 == 0 else 'XYZ' for i in range(rows)],
        'ABC': ['ABC'] * rows
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Generated DataFrame: {df.shape} - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Test Original Formula Engine
    print(f"\n🐌 Original Formula Engine:")
    try:
        times = []
        for i in range(5):  # More iterations for better accuracy
            test_df = df.copy()
            engine = OriginalFormula(test_df)
            
            start_time = time.perf_counter()
            result = engine.evaluate(formula, 'result')
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        orig_avg_time = sum(times) / len(times)
        orig_min_time = min(times)
        orig_max_time = max(times)
        orig_result = result['result']
        
        print(f"  ⏱️  Average time: {orig_avg_time:.4f}s")
        print(f"  📊 Range: {orig_min_time:.4f}s - {orig_max_time:.4f}s")
        print(f"  🚀 Rate: {rows/orig_avg_time:,.0f} rows/second")
        print(f"  📊 Sample results: {orig_result.head(5).tolist()}")
        print(f"  📈 True count: {orig_result.sum():,} / {len(orig_result):,}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return
    
    # Test Optimized Formula Engine
    print(f"\n🚀 Optimized Formula Engine:")
    try:
        times = []
        for i in range(5):
            test_df = df.copy()
            engine = FormulaOptimized(test_df)
            
            start_time = time.perf_counter()
            result = engine.evaluate(formula, 'result')
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        opt_avg_time = sum(times) / len(times)
        opt_min_time = min(times)
        opt_max_time = max(times)
        opt_result = result['result']
        
        print(f"  ⏱️  Average time: {opt_avg_time:.4f}s")
        print(f"  📊 Range: {opt_min_time:.4f}s - {opt_max_time:.4f}s")
        print(f"  🚀 Rate: {rows/opt_avg_time:,.0f} rows/second")
        print(f"  📊 Sample results: {opt_result.head(5).tolist()}")
        print(f"  📈 True count: {opt_result.sum():,} / {len(opt_result):,}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return
    
    # Verify results match
    print(f"\n🔍 Results Verification:")
    try:
        results_match = orig_result.equals(opt_result)
        if results_match:
            print("  ✅ Results are identical")
        else:
            print("  ❌ Results differ!")
            diff_count = (orig_result != opt_result).sum()
            print(f"    Differences: {diff_count:,} out of {len(orig_result):,} rows")
    except Exception as e:
        print(f"  ⚠️  Could not compare: {e}")
        results_match = False
    
    # Performance comparison
    print(f"\n📊 Performance Analysis:")
    speedup = orig_avg_time / opt_avg_time
    improvement = ((orig_avg_time - opt_avg_time) / orig_avg_time) * 100
    
    print(f"  🏆 Speedup: {speedup:.2f}x faster")
    print(f"  📈 Improvement: {improvement:+.1f}%")
    print(f"  ⚡ Time saved per execution: {(orig_avg_time - opt_avg_time)*1000:.1f}ms")
    
    # Scaling analysis
    print(f"\n📈 Scaling Analysis:")
    print(f"  Original engine: {orig_avg_time/rows*1000000:.2f} μs per row")
    print(f"  Optimized engine: {opt_avg_time/rows*1000000:.2f} μs per row")
    
    # Extrapolation to larger datasets
    for scale in [1_000_000, 10_000_000]:
        orig_projected = (orig_avg_time / rows) * scale
        opt_projected = (opt_avg_time / rows) * scale
        time_saved = orig_projected - opt_projected
        
        print(f"  📊 {scale:,} rows:")
        print(f"    Original: {orig_projected:.2f}s")
        print(f"    Optimized: {opt_projected:.2f}s")
        print(f"    Time saved: {time_saved:.2f}s")
    
    return {
        'formula': formula,
        'rows': rows,
        'original_time': orig_avg_time,
        'optimized_time': opt_avg_time,
        'speedup': speedup,
        'improvement': improvement,
        'results_match': results_match
    }

def test_numexpr_suitable_expressions():
    """Test expressions that are suitable for numexpr optimization"""
    
    print(f"\n⚡ NUMEXPR-OPTIMIZED EXPRESSIONS")
    print("=" * 70)
    
    # Create numerical test data
    rows = 400000
    np.random.seed(42)
    
    data = {
        'price': np.random.uniform(10.0, 1000.0, rows),
        'quantity': np.random.randint(1, 100, rows),
        'tax_rate': np.random.uniform(0.05, 0.25, rows),
        'discount': np.random.uniform(0.0, 0.3, rows),
        'cost': np.random.uniform(5.0, 500.0, rows),
    }
    
    df = pd.DataFrame(data)
    
    # Test pure numerical expressions that numexpr excels at
    numerical_formulas = [
        'price * quantity',
        'price * quantity * (1 + tax_rate)',
        'price * quantity * (1 - discount) * (1 + tax_rate)',
        '(price - cost) * quantity',
        'price ** 2 + quantity ** 2',
    ]
    
    for formula in numerical_formulas:
        print(f"\n🧮 Testing: {formula}")
        print("-" * 50)
        
        # Test with pandas eval (baseline)
        pandas_times = []
        for _ in range(3):
            test_df = df.copy()
            start = time.perf_counter()
            test_df['result_pandas'] = test_df.eval(formula)
            pandas_time = time.perf_counter() - start
            pandas_times.append(pandas_time)
        
        pandas_avg = sum(pandas_times) / len(pandas_times)
        print(f"  🐼 Pandas eval: {pandas_avg:.4f}s ({rows/pandas_avg:,.0f} rows/s)")
        
        # Test with numexpr
        try:
            # Convert formula to use column values directly
            local_dict = {col: df[col].values for col in df.columns if col in formula}
            
            numexpr_times = []
            for _ in range(3):
                start = time.perf_counter()
                result_numexpr = ne.evaluate(formula, local_dict=local_dict)
                numexpr_time = time.perf_counter() - start
                numexpr_times.append(numexpr_time)
            
            numexpr_avg = sum(numexpr_times) / len(numexpr_times)
            print(f"  ⚡ Numexpr: {numexpr_avg:.4f}s ({rows/numexpr_avg:,.0f} rows/s)")
            
            # Compare performance
            speedup = pandas_avg / numexpr_avg
            improvement = ((pandas_avg - numexpr_avg) / pandas_avg) * 100
            print(f"  🚀 Speedup: {speedup:.2f}x faster ({improvement:+.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Numexpr failed: {e}")

if __name__ == "__main__":
    # Test your specific complex expression
    result = test_your_complex_expression()
    
    # Test numexpr-suitable expressions
    test_numexpr_suitable_expressions()
    
    print(f"\n🎉 COMPLEX EXPRESSION TESTING COMPLETE!")
    print("=" * 70)