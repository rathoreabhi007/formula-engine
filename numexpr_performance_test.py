#!/usr/bin/env python3

import time
import pandas as pd
import numpy as np
from fo import Formula as OriginalFormula
from fo_optimized import FormulaOptimized

# Check if numexpr is available
try:
    import numexpr as ne
    NUMEXPR_AVAILABLE = True
    print("✅ numexpr library found and imported successfully")
except ImportError:
    NUMEXPR_AVAILABLE = False
    print("❌ numexpr library not found. Install with: pip install numexpr")

def create_large_test_data(rows=400000):
    """Create large test dataset for performance testing"""
    print(f"🔧 Generating {rows:,} rows of test data...")
    np.random.seed(42)
    
    data = {
        # Numerical columns for numexpr testing
        'x': np.random.uniform(1.0, 10000.0, rows),
        'y': np.random.uniform(1.0, 1000.0, rows), 
        'z': np.random.uniform(0.1, 100.0, rows),
        'score': np.random.randint(1, 1000, rows),
        'amount': np.random.uniform(100.0, 50000.0, rows),
        'count': np.random.randint(1, 100, rows),
        
        # String columns for formula engine testing
        'name': [f'user_{i}' for i in range(rows)],
        'Source': [f'source_{i}' if i % 10 != 0 else None for i in range(rows)],
        'CCY': [f'USD{"x" if i % 3 == 0 else ""}' for i in range(rows)],
        'TRN': ['ABC' if i % 2 == 0 else 'XYZ' for i in range(rows)],
        'ABC': ['ABC'] * rows,
        
        # Boolean column
        'active': np.random.choice([True, False], rows),
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Generated DataFrame: {df.shape} - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df

class NumexprFormulaEngine:
    """Formula engine that uses numexpr for numerical expressions"""
    
    def __init__(self, df):
        self.df = df
        
    def evaluate(self, formula, output_col):
        """Evaluate formula using numexpr when possible"""
        
        # Simple numerical expressions that numexpr can handle
        numexpr_patterns = {
            # Basic arithmetic
            'x + y': lambda df: ne.evaluate('x + y', local_dict={'x': df['x'].values, 'y': df['y'].values}),
            'x * y': lambda df: ne.evaluate('x * y', local_dict={'x': df['x'].values, 'y': df['y'].values}),
            'x - y': lambda df: ne.evaluate('x - y', local_dict={'x': df['x'].values, 'y': df['y'].values}),
            'x / y': lambda df: ne.evaluate('x / y', local_dict={'x': df['x'].values, 'y': df['y'].values}),
            'x ** 2': lambda df: ne.evaluate('x ** 2', local_dict={'x': df['x'].values}),
            'x * 2': lambda df: ne.evaluate('x * 2', local_dict={'x': df['x'].values}),
            'x * 1.1': lambda df: ne.evaluate('x * 1.1', local_dict={'x': df['x'].values}),
            'score * 2': lambda df: ne.evaluate('score * 2', local_dict={'score': df['score'].values}),
            'amount * 1.1': lambda df: ne.evaluate('amount * 1.1', local_dict={'amount': df['amount'].values}),
            'x * y / 100': lambda df: ne.evaluate('x * y / 100', local_dict={'x': df['x'].values, 'y': df['y'].values}),
            
            # Complex numerical expressions
            'x * 2 + y * 3': lambda df: ne.evaluate('x * 2 + y * 3', local_dict={'x': df['x'].values, 'y': df['y'].values}),
            'x ** 2 + y ** 2': lambda df: ne.evaluate('x ** 2 + y ** 2', local_dict={'x': df['x'].values, 'y': df['y'].values}),
            '(x + y) * z': lambda df: ne.evaluate('(x + y) * z', local_dict={'x': df['x'].values, 'y': df['y'].values, 'z': df['z'].values}),
            'x * y + z * count': lambda df: ne.evaluate('x * y + z * count', local_dict={'x': df['x'].values, 'y': df['y'].values, 'z': df['z'].values, 'count': df['count'].values}),
            
            # Boolean expressions
            'x > 5000': lambda df: ne.evaluate('x > 5000', local_dict={'x': df['x'].values}),
            'score > 500': lambda df: ne.evaluate('score > 500', local_dict={'score': df['score'].values}),
            '(x > 1000) & (y < 500)': lambda df: ne.evaluate('(x > 1000) & (y < 500)', local_dict={'x': df['x'].values, 'y': df['y'].values}),
        }
        
        if formula in numexpr_patterns:
            result = numexpr_patterns[formula](self.df)
            self.df[output_col] = result
        else:
            # Fallback to pandas eval for expressions numexpr can't handle
            try:
                self.df[output_col] = self.df.eval(formula)
            except:
                # Final fallback to Python eval (not recommended for large data)
                self.df[output_col] = eval(formula, {}, {'df': self.df, 'np': np})
                
        return self.df

def benchmark_engines(formula, df, iterations=3):
    """Benchmark formula across all available engines"""
    results = {}
    
    print(f"\n🧪 Testing formula: {formula}")
    print("-" * 60)
    
    # Test Original Formula Engine
    print("🐌 Original Formula Engine:")
    try:
        times = []
        for _ in range(iterations):
            test_df = df.copy()
            engine = OriginalFormula(test_df)
            
            start_time = time.perf_counter()
            result = engine.evaluate(formula, 'result_orig')
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        results['original'] = {
            'time': avg_time,
            'success': True,
            'result': result['result_orig'],
            'error': None
        }
        print(f"  ⏱️  Average time: {avg_time:.4f}s")
        print(f"  📊 Rate: {len(df)/avg_time:,.0f} rows/second")
        
    except Exception as e:
        results['original'] = {'time': float('inf'), 'success': False, 'result': None, 'error': str(e)}
        print(f"  ❌ Failed: {e}")
    
    # Test Optimized Formula Engine
    print("\n🚀 Optimized Formula Engine:")
    try:
        times = []
        for _ in range(iterations):
            test_df = df.copy()
            engine = FormulaOptimized(test_df)
            
            start_time = time.perf_counter()
            result = engine.evaluate(formula, 'result_opt')
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        results['optimized'] = {
            'time': avg_time,
            'success': True,
            'result': result['result_opt'],
            'error': None
        }
        print(f"  ⏱️  Average time: {avg_time:.4f}s")
        print(f"  📊 Rate: {len(df)/avg_time:,.0f} rows/second")
        
    except Exception as e:
        results['optimized'] = {'time': float('inf'), 'success': False, 'result': None, 'error': str(e)}
        print(f"  ❌ Failed: {e}")
    
    # Test Numexpr Engine (if available)
    if NUMEXPR_AVAILABLE:
        print("\n⚡ Numexpr Engine:")
        try:
            times = []
            for _ in range(iterations):
                test_df = df.copy()
                engine = NumexprFormulaEngine(test_df)
                
                start_time = time.perf_counter()
                result = engine.evaluate(formula, 'result_numexpr')
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            results['numexpr'] = {
                'time': avg_time,
                'success': True,
                'result': result['result_numexpr'],
                'error': None
            }
            print(f"  ⏱️  Average time: {avg_time:.4f}s")
            print(f"  📊 Rate: {len(df)/avg_time:,.0f} rows/second")
            
        except Exception as e:
            results['numexpr'] = {'time': float('inf'), 'success': False, 'result': None, 'error': str(e)}
            print(f"  ❌ Failed: {e}")
    else:
        results['numexpr'] = {'time': float('inf'), 'success': False, 'result': None, 'error': 'numexpr not available'}
    
    # Compare results
    print(f"\n📊 Performance Comparison:")
    successful_engines = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_engines) > 1:
        # Find fastest
        fastest = min(successful_engines.keys(), key=lambda k: successful_engines[k]['time'])
        print(f"  🏆 Fastest: {fastest.capitalize()} ({successful_engines[fastest]['time']:.4f}s)")
        
        # Calculate speedups relative to original
        if 'original' in successful_engines and successful_engines['original']['time'] > 0:
            orig_time = successful_engines['original']['time']
            
            for engine, data in successful_engines.items():
                if engine != 'original':
                    speedup = orig_time / data['time']
                    improvement = ((orig_time - data['time']) / orig_time) * 100
                    print(f"  🚀 {engine.capitalize()} vs Original: {speedup:.2f}x faster ({improvement:+.1f}%)")
        
        # Verify results consistency
        print(f"\n🔍 Results Verification:")
        reference_result = None
        reference_engine = None
        
        for engine, data in successful_engines.items():
            if data['result'] is not None:
                if reference_result is None:
                    reference_result = data['result']
                    reference_engine = engine
                else:
                    try:
                        if hasattr(reference_result, 'dtype') and reference_result.dtype in ['float64', 'float32']:
                            match = np.allclose(reference_result, data['result'], rtol=1e-10, equal_nan=True)
                        else:
                            match = np.array_equal(reference_result, data['result'])
                        
                        if match:
                            print(f"  ✅ {engine.capitalize()} matches {reference_engine}")
                        else:
                            print(f"  ❌ {engine.capitalize()} differs from {reference_engine}")
                            print(f"    Sample {reference_engine}: {reference_result[:3] if hasattr(reference_result, '__getitem__') else reference_result}")
                            print(f"    Sample {engine}: {data['result'][:3] if hasattr(data['result'], '__getitem__') else data['result']}")
                    except Exception as e:
                        print(f"  ⚠️  Could not compare {engine} with {reference_engine}: {e}")
    
    return results

def run_comprehensive_performance_test():
    """Run comprehensive performance test with 400K rows"""
    
    print("🚀 COMPREHENSIVE FORMULA ENGINE PERFORMANCE TEST")
    print("=" * 80)
    print("Dataset: 400,000 rows")
    
    # Create large test dataset
    df = create_large_test_data(400000)
    
    # Test formulas - mix of numerical and complex expressions
    test_formulas = [
        # Pure numerical (ideal for numexpr)
        'x * 2',
        'x + y',
        'x * y / 100',
        'x ** 2',
        'amount * 1.1',
        'score * 2',
        '(x + y) * z',
        'x * y + z * count',
        
        # Boolean expressions
        'x > 5000',
        'score > 500',
        
        # Complex expressions (may require formula engines)
        '"High" if score > 500 else "Low"',  # String conditional
        'True if (x > 1000) and (y < 500) else False',  # Boolean conditional
    ]
    
    # Special test: Your original complex formula
    if NUMEXPR_AVAILABLE:
        test_formulas.append('True if Source.__contains__("source") and CCY.endswith("x") else False')
    
    all_results = []
    
    for formula in test_formulas:
        try:
            results = benchmark_engines(formula, df, iterations=3)
            results['formula'] = formula
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Error testing formula '{formula}': {e}")
            continue
    
    # Overall summary
    print(f"\n{'='*80}")
    print("📊 OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    # Calculate averages for successful tests
    engine_stats = {'original': [], 'optimized': [], 'numexpr': []}
    
    for result in all_results:
        for engine in engine_stats.keys():
            if engine in result and result[engine]['success']:
                engine_stats[engine].append(result[engine]['time'])
    
    print(f"{'Engine':<15} | {'Tests':<8} | {'Avg Time':<12} | {'Avg Rate':<15}")
    print("-" * 60)
    
    for engine, times in engine_stats.items():
        if times:
            avg_time = sum(times) / len(times)
            avg_rate = 400000 / avg_time
            print(f"{engine.capitalize():<15} | {len(times):<8} | {avg_time:<12.4f} | {avg_rate:<15,.0f} rows/s")
    
    # Best performers by category
    print(f"\n🏆 BEST PERFORMERS:")
    
    numerical_formulas = ['x * 2', 'x + y', 'x * y / 100', 'x ** 2', 'amount * 1.1']
    numerical_results = [r for r in all_results if r['formula'] in numerical_formulas]
    
    if numerical_results:
        print(f"\n📊 Numerical Expressions (best for numexpr):")
        for result in numerical_results[:3]:  # Show top 3
            successful = {k: v for k, v in result.items() if k != 'formula' and v.get('success', False)}
            if successful:
                fastest = min(successful.keys(), key=lambda k: successful[k]['time'])
                time_val = successful[fastest]['time']
                rate = 400000 / time_val
                print(f"  {result['formula']:<20} → {fastest.capitalize():<12} ({time_val:.4f}s, {rate:,.0f} rows/s)")

def test_numexpr_installation():
    """Test if numexpr can be installed and used"""
    print("\n🔧 TESTING NUMEXPR INSTALLATION")
    print("=" * 50)
    
    if not NUMEXPR_AVAILABLE:
        print("❌ numexpr not found. Installing...")
        print("💡 Run: pip install numexpr")
        return False
    
    # Test basic numexpr functionality
    try:
        import numexpr as ne
        
        # Test basic operation
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 3, 4, 5, 6])
        result = ne.evaluate('a * 2 + b')
        expected = a * 2 + b
        
        if np.array_equal(result, expected):
            print("✅ numexpr basic functionality test passed")
            
            # Test performance
            large_a = np.random.uniform(0, 100, 100000)
            large_b = np.random.uniform(0, 100, 100000)
            
            start = time.perf_counter()
            np_result = large_a * 2 + large_b
            np_time = time.perf_counter() - start
            
            start = time.perf_counter()
            ne_result = ne.evaluate('large_a * 2 + large_b')
            ne_time = time.perf_counter() - start
            
            speedup = np_time / ne_time
            print(f"✅ numexpr performance test: {speedup:.2f}x faster than numpy")
            
            return True
        else:
            print("❌ numexpr functionality test failed")
            return False
            
    except Exception as e:
        print(f"❌ numexpr test failed: {e}")
        return False

if __name__ == "__main__":
    # Test numexpr installation
    numexpr_working = test_numexpr_installation()
    
    if numexpr_working or NUMEXPR_AVAILABLE:
        # Run comprehensive performance test
        run_comprehensive_performance_test()
    else:
        print("\n💡 To get the full performance comparison, install numexpr:")
        print("   pip install numexpr")
        print("\n🔄 Running test with available engines only...")
        
        # Run limited test without numexpr
        df = create_large_test_data(400000)
        simple_formulas = ['x * 2', 'x + y', 'score > 500']
        
        for formula in simple_formulas:
            benchmark_engines(formula, df, iterations=3)
    
    print(f"\n🎉 PERFORMANCE TESTING COMPLETE!")
    print("=" * 80)