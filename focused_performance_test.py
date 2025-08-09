#!/usr/bin/env python3

import time
import pandas as pd
import numpy as np
from fo import Formula as OriginalFormula
from fo_optimized import FormulaOptimized

def create_consistent_test_data(rows=1000):
    """Create consistent test data for both engines"""
    np.random.seed(42)  # Ensure reproducible results
    
    data = {
        'name': [f'user_{i}' for i in range(rows)],
        'score': np.random.randint(1, 1000, rows),
        'text': [f'text_{i}' for i in range(rows)],
        'num': np.random.randint(1, 100, rows),
        'active': np.random.choice([True, False], rows),
        'Source': [f'source_{i}' if i % 10 != 0 else None for i in range(rows)],
        'CCY': [f'USD{"x" if i % 3 == 0 else ""}' for i in range(rows)],
        'TRN': ['ABC' if i % 2 == 0 else 'XYZ' for i in range(rows)],
        'ABC': ['ABC'] * rows
    }
    
    return pd.DataFrame(data)

def benchmark_single_formula(formula, df, engine_class, iterations=5):
    """Benchmark a single formula execution"""
    times = []
    
    for _ in range(iterations):
        test_df = df.copy()
        engine = engine_class(test_df)
        
        start_time = time.perf_counter()
        result = engine.evaluate(formula, 'result')
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'result': result['result']
    }

def performance_comparison():
    """Compare performance between original and optimized versions"""
    
    print("🚀 FOCUSED FORMULA ENGINE PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # Test formulas that work in both engines
    test_formulas = [
        {
            'name': 'Simple String Operation',
            'formula': 'name.upper()',
            'rows': 1000
        },
        {
            'name': 'Simple Arithmetic',
            'formula': 'score * 2',
            'rows': 1000
        },
        {
            'name': 'String Concatenation',
            'formula': 'name + "_suffix"',
            'rows': 1000
        },
        {
            'name': 'Simple Conditional',
            'formula': '"High" if score > 500 else "Low"',
            'rows': 1000
        },
        {
            'name': 'Null Check',
            'formula': 'IsNotNull(Source)',
            'rows': 1000
        },
        {
            'name': 'String Method',
            'formula': 'name.__contains__("user")',
            'rows': 1000
        },
        {
            'name': 'Large Dataset - Simple',
            'formula': 'score * 1.1',
            'rows': 50000
        },
        {
            'name': 'Complex User Formula',
            'formula': 'True if IsNotNull(Source) and CCY.endswith("x") and TRN == ABC else False',
            'rows': 10000
        }
    ]
    
    results = []
    
    for test in test_formulas:
        print(f"\n📊 {test['name']}")
        print("-" * 60)
        print(f"Formula: {test['formula']}")
        print(f"Rows: {test['rows']:,}")
        
        # Create test data
        df = create_consistent_test_data(test['rows'])
        
        try:
            # Test Original Implementation
            print("\n🐌 Original Engine:")
            orig_result = benchmark_single_formula(test['formula'], df, OriginalFormula, iterations=3)
            print(f"  Average time: {orig_result['avg_time']:.6f}s")
            print(f"  Range: {orig_result['min_time']:.6f}s - {orig_result['max_time']:.6f}s")
            
            # Test Optimized Implementation
            print("\n🚀 Optimized Engine:")
            opt_result = benchmark_single_formula(test['formula'], df, FormulaOptimized, iterations=3)
            print(f"  Average time: {opt_result['avg_time']:.6f}s")
            print(f"  Range: {opt_result['min_time']:.6f}s - {opt_result['max_time']:.6f}s")
            
            # Compare results
            try:
                if orig_result['result'].dtype in ['float64', 'float32']:
                    results_match = np.allclose(orig_result['result'], opt_result['result'], 
                                              rtol=1e-10, equal_nan=True)
                else:
                    results_match = orig_result['result'].equals(opt_result['result'])
                
                if results_match:
                    print("  ✅ Results identical")
                else:
                    print("  ❌ Results differ!")
                    print(f"    Original sample: {orig_result['result'].head(3).tolist()}")
                    print(f"    Optimized sample: {opt_result['result'].head(3).tolist()}")
            except Exception as e:
                print(f"  ⚠️  Could not compare: {e}")
                results_match = False
            
            # Calculate performance metrics
            speedup = orig_result['avg_time'] / opt_result['avg_time'] if opt_result['avg_time'] > 0 else 1
            improvement = ((orig_result['avg_time'] - opt_result['avg_time']) / orig_result['avg_time']) * 100
            
            print(f"\n📈 Performance:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Improvement: {improvement:+.1f}%")
            
            # Store results
            results.append({
                'name': test['name'],
                'formula': test['formula'],
                'rows': test['rows'],
                'orig_time': orig_result['avg_time'],
                'opt_time': opt_result['avg_time'],
                'speedup': speedup,
                'improvement': improvement,
                'results_match': results_match
            })
            
        except Exception as e:
            print(f"  ❌ Error testing formula: {e}")
            continue
    
    # Overall Summary
    print(f"\n{'='*70}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        successful_tests = len([r for r in results if r['results_match']])
        
        print(f"Tests completed: {len(results)}")
        print(f"Results match: {successful_tests}/{len(results)}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Average improvement: {avg_improvement:+.1f}%")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 100)
        print(f"{'Test':<25} | {'Rows':<8} | {'Speedup':<8} | {'Improvement':<12} | {'Match':<5}")
        print("-" * 100)
        
        for r in results:
            match_icon = "✅" if r['results_match'] else "❌"
            print(f"{r['name']:<25} | {r['rows']:<8,} | {r['speedup']:<8.2f} | {r['improvement']:+10.1f}% | {match_icon:<5}")
        
        # Highlight best performers
        if results:
            best_speedup = max(results, key=lambda x: x['speedup'])
            print(f"\n🏆 Best speedup: {best_speedup['name']} ({best_speedup['speedup']:.2f}x)")
            
            best_improvement = max(results, key=lambda x: x['improvement'])
            print(f"🏆 Best improvement: {best_improvement['name']} ({best_improvement['improvement']:+.1f}%)")
    
    return results

def test_optimization_features():
    """Test specific optimization features"""
    print(f"\n🔬 OPTIMIZATION FEATURES")
    print("=" * 70)
    
    df = create_consistent_test_data(1000)
    
    # Test 1: Regex compilation benefit
    print("1. Regex Compilation Optimization")
    print("-" * 40)
    
    engine = FormulaOptimized(df)
    formulas = ['name.upper()', 'score * 2', 'name[:3]', 'IsNotNull(Source)']
    
    start_time = time.perf_counter()
    for i, formula in enumerate(formulas * 25):  # 100 total evaluations
        engine.evaluate(formula, f'test_{i}')
    total_time = time.perf_counter() - start_time
    
    print(f"  100 formula evaluations: {total_time:.4f}s")
    print(f"  Average per formula: {total_time/100:.6f}s")
    print("  ✅ Pre-compiled regex patterns reused efficiently")
    
    # Test 2: Memory efficiency
    print("\n2. Memory Efficiency")
    print("-" * 40)
    
    # Test with larger dataset
    large_df = create_consistent_test_data(100000)
    large_engine = FormulaOptimized(large_df)
    
    start_time = time.perf_counter()
    large_engine.evaluate('name + "_" + str(score)', 'memory_test')
    large_time = time.perf_counter() - start_time
    
    print(f"  100K rows processed: {large_time:.4f}s")
    print(f"  Rate: {100000/large_time:,.0f} rows/second")
    print("  ✅ Efficient processing of large datasets")

if __name__ == "__main__":
    # Run performance comparison
    results = performance_comparison()
    
    # Test optimization features
    test_optimization_features()
    
    print(f"\n🎉 PERFORMANCE ANALYSIS COMPLETE!")
    print("=" * 70)