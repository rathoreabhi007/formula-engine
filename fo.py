import re
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

class Formula:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ----------- NULL CHECKS -----------
    @staticmethod
    def is_null(series: pd.Series) -> pd.Series:
        """Custom null detection"""
        return (
            series.isna()
            | (series.astype(str).str.strip() == "")
            | (series.astype(str).str.lower() == "null")
        )

    @staticmethod
    def is_not_null(series: pd.Series) -> pd.Series:
        """Custom not-null detection"""
        return ~Formula.is_null(series)

    # ----------- COLUMN REPLACEMENT -----------
    def _replace_column_refs(self, formula: str) -> str:
        """Replace bare column names with df['col']"""
        col_pattern = r'\b({})\b'.format('|'.join(map(re.escape, self.df.columns)))
        return re.sub(col_pattern, lambda m: f"df[{repr(m.group(0))}]", formula)
    
    # ----------- STR() FUNCTION HANDLING -----------
    def _process_str_function(self, formula: str) -> str:
        """Convert str(column) to column.astype(str) for proper element-wise conversion"""
        # Match str(df['col']) or str(column_name) patterns
        str_pattern = re.compile(r'str\((df\[[\'"]?\w+[\'"]?\])\)')
        return str_pattern.sub(r'\1.astype(str)', formula)
    
    # ----------- PARENTHESES BALANCING -----------
    def _balance_parentheses(self, formula: str) -> str:
        """Ensure parentheses are properly balanced and fix common issues"""
        # Count parentheses to detect imbalances
        open_count = formula.count('(')
        close_count = formula.count(')')
        
        # If more opens than closes, add missing closes at the end
        if open_count > close_count:
            formula += ')' * (open_count - close_count)
        
        return formula

    # ----------- SLICE HANDLING -----------
    def _process_slices(self, formula: str) -> str:
        """
        Convert (df['col'])[:3] or (df['col'])[1:5] to df['col'].str.slice(start, stop)
        Also handle slicing on astype(str) operations
        """
        # Handle regular column slicing
        slice_pattern = re.compile(r"(df\[['\"]?\w+['\"]?\])\s*\[\s*(\d*)\s*:\s*(\d*)\s*\]")
        
        def repl(match):
            col_ref = match.group(1)
            start = match.group(2) if match.group(2) != "" else "None"
            stop = match.group(3) if match.group(3) != "" else "None"
            return f"{col_ref}.str.slice({start}, {stop})"
        
        formula = slice_pattern.sub(repl, formula)
        
        # Handle slicing on astype(str) operations: df['col'].astype(str)[:2]
        astype_slice_pattern = re.compile(r"(df\[['\"]?\w+['\"]?\]\.astype\(str\))\s*\[\s*(\d*)\s*:\s*(\d*)\s*\]")
        
        def repl_astype(match):
            col_ref = match.group(1)
            start = match.group(2) if match.group(2) != "" else "None"
            stop = match.group(3) if match.group(3) != "" else "None"
            return f"{col_ref}.str.slice({start}, {stop})"
        
        formula = astype_slice_pattern.sub(repl_astype, formula)
        
        return formula

    # ----------- NULL CHECKS REPLACEMENT -----------
    def _process_null_checks(self, formula: str) -> str:
        """Replace IsNull / IsNotNull with custom calls"""
        formula = re.sub(
            r"IsNull\((df\[['\"]?\w+['\"]?\])\)",
            r"Formula.is_null(\1)",
            formula
        )
        formula = re.sub(
            r"IsNotNull\((df\[['\"]?\w+['\"]?\])\)",
            r"Formula.is_not_null(\1)",
            formula
        )
        return formula

    # ----------- STRING METHODS HANDLING -----------
    def _process_str_methods(self, formula: str) -> str:
        """Ensure string methods and __contains__ are prefixed with .str."""
        str_methods = set(dir(pd.Series([], dtype="object").str))

        # Handle __contains__ → .str.contains with na=False to handle nulls properly
        # Including parenthesized column references
        formula = re.sub(
            r"(\(df\[['\"]?\w+['\"]?\]\)|df\[['\"]?\w+['\"]?\])\.__contains__\(",
            r"\1.str.contains(",
            formula
        )
        
        # Clean up any extra parentheses around column references for str.contains
        formula = re.sub(
            r"\((df\[['\"]?\w+['\"]?\])\)\.str\.contains\(",
            r"\1.str.contains(",
            formula
        )
        
        # Add na=False parameter to str.contains calls to handle null values properly
        formula = re.sub(
            r"(df\[['\"]?\w+['\"]?\])\.str\.contains\(([^)]+)\)",
            r"\1.str.contains(\2, na=False)",
            formula
        )

        # Handle standard string methods - including parenthesized column references
        method_pattern = re.compile(r"(\(df\[['\"]?\w+['\"]?\]\)|df\[['\"]?\w+['\"]?\])\.(\w+)\(")

        def repl(match):
            col_ref = match.group(1)
            method = match.group(2)
            if method in str_methods:
                # Remove parentheses if they exist around the column reference
                clean_col_ref = col_ref.strip('()')
                return f"{clean_col_ref}.str.{method}("
            else:
                return match.group(0)

        return method_pattern.sub(repl, formula)
    
    # ----------- LOGICAL OPERATORS HANDLING -----------
    def _process_logical_operators(self, formula: str) -> str:
        """Convert Python logical operators to pandas-compatible ones"""
        
        # Replace 'and' with '&' and 'or' with '|' for pandas Series operations
        # But be careful not to replace them inside strings and handle precedence
        
        def replace_logical_ops(text):
            # Track whether we're inside quotes to avoid replacing 'and'/'or' in strings
            result = []
            i = 0
            in_quotes = False
            quote_char = None
            
            while i < len(text):
                char = text[i]
                
                # Handle quote tracking
                if char in ['"', "'"] and (i == 0 or text[i-1] != '\\'):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                
                # Only replace operators outside of quotes
                if not in_quotes:
                    # Check for 'and' with word boundaries
                    if (i <= len(text) - 3 and 
                        text[i:i+3] == 'and' and 
                        (i == 0 or not text[i-1].isalnum()) and 
                        (i+3 >= len(text) or not text[i+3].isalnum())):
                        result.append(' & ')
                        i += 3
                        continue
                    
                    # Check for 'or' with word boundaries
                    elif (i <= len(text) - 2 and 
                          text[i:i+2] == 'or' and 
                          (i == 0 or not text[i-1].isalnum()) and 
                          (i+2 >= len(text) or not text[i+2].isalnum())):
                        result.append(' | ')
                        i += 2
                        continue
                
                result.append(char)
                i += 1
            
            return ''.join(result)
        
        # Only process if we have pandas column references
        if 'df[' in formula:
            formula = replace_logical_ops(formula)
            
            # Add parentheses around complex logical expressions to ensure proper precedence
            # Look for patterns like: condition1 & condition2 & condition3 | condition4
            # And wrap the parts before | in parentheses
            
            # Find all | operators not in quotes and not inside function calls like np.select([...])
            or_positions = []
            in_quotes = False
            quote_char = None
            paren_depth = 0
            bracket_depth = 0
            
            for i, char in enumerate(formula):
                if char in ['"', "'"] and (i == 0 or formula[i-1] != '\\'):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif not in_quotes:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    elif char == '[':
                        bracket_depth += 1
                    elif char == ']':
                        bracket_depth -= 1
                    elif char == '|' and paren_depth == 0 and bracket_depth == 0:
                        or_positions.append(i)
            
            # If we have OR operators at the top level (not inside np.select), add parentheses around & chains
            if or_positions and not formula.strip().startswith('np.select'):
                parts = []
                start = 0
                for pos in or_positions:
                    part = formula[start:pos].strip()
                    # If this part contains & operators, wrap it in parentheses
                    if ' & ' in part and not (part.startswith('(') and part.endswith(')')):
                        part = f"({part})"
                    parts.append(part)
                    start = pos + 1
                
                # Handle the last part
                last_part = formula[start:].strip()
                if ' & ' in last_part and not (last_part.startswith('(') and last_part.endswith(')')):
                    last_part = f"({last_part})"
                parts.append(last_part)
                
                formula = ' | '.join(parts)
            
            # Special handling for conditions inside np.select
            elif formula.strip().startswith('np.select'):
                # Extract the condition part from np.select([condition], ...)
                # Use bracket matching to properly extract the condition
                start_pos = formula.find('np.select([') + len('np.select([')
                bracket_count = 1
                end_pos = start_pos
                
                for i in range(start_pos, len(formula)):
                    char = formula[i]
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i
                            break
                
                condition = formula[start_pos:end_pos] if bracket_count == 0 else None
                
                if condition:
                    
                    # Apply precedence fixes to the condition
                    # Find | operators in the condition and add parentheses around & chains
                    or_positions = []
                    in_quotes = False
                    quote_char = None
                    paren_depth = 0
                    bracket_depth = 0
                    
                    for i, char in enumerate(condition):
                        if char in ['"', "'"] and (i == 0 or condition[i-1] != '\\'):
                            if not in_quotes:
                                in_quotes = True
                                quote_char = char
                            elif char == quote_char:
                                in_quotes = False
                                quote_char = None
                        elif not in_quotes:
                            if char == '(':
                                paren_depth += 1
                            elif char == ')':
                                paren_depth -= 1
                            elif char == '[':
                                bracket_depth += 1
                            elif char == ']':
                                bracket_depth -= 1
                            elif char == '|' and paren_depth == 0 and bracket_depth == 0:
                                or_positions.append(i)
                    
                    if or_positions:
                        parts = []
                        start = 0
                        for pos in or_positions:
                            part = condition[start:pos].strip()
                            # Add parentheses around & chains AND around comparison operations
                            part = self._add_comparison_parentheses(part)
                            if ' & ' in part and not (part.startswith('(') and part.endswith(')')):
                                part = f"({part})"
                            parts.append(part)
                            start = pos + 1
                        
                        last_part = condition[start:].strip()
                        last_part = self._add_comparison_parentheses(last_part)
                        if ' & ' in last_part and not (last_part.startswith('(') and last_part.endswith(')')):
                            last_part = f"({last_part})"
                        parts.append(last_part)
                        
                        new_condition = ' | '.join(parts)
                        formula = formula.replace(condition, new_condition)
                    else:
                        # Even if no OR operators, we still need to fix precedence for & and ==
                        new_condition = self._add_comparison_parentheses(condition)
                        formula = formula.replace(condition, new_condition)
        
        return formula
    
    # ----------- COMPARISON PARENTHESES -----------
    def _add_comparison_parentheses(self, expression: str) -> str:
        """Add parentheses around comparison operations to fix operator precedence"""
        
        # Pattern to find comparison operations (==, !=, <, >, <=, >=)
        # More precise pattern that handles complex expressions
        comparison_pattern = re.compile(r'(\w+(?:\([^)]*\))?(?:\[[^\]]+\])?(?:\.[^\s&|<>=!]+(?:\([^)]*\))?)*)\s*(==|!=|<=|>=|<|>)\s*(\w+(?:\([^)]*\))?(?:\[[^\]]+\])?(?:\.[^\s&|<>=!]+(?:\([^)]*\))?)*)')
        
        def add_parens(match):
            left = match.group(1)
            operator = match.group(2)
            right = match.group(3)
            # Only add parentheses if not already present
            full_comparison = f"{left} {operator} {right}"
            if not (expression.find(f"({full_comparison})") >= 0):
                return f"({left} {operator} {right})"
            return full_comparison
        
        # Replace comparison operations with parenthesized versions
        result = comparison_pattern.sub(add_parens, expression)
        
        return result

    # ----------- VECTORIZED CONDITIONALS -----------
    def _convert_pandas_conditionals(self, formula: str) -> str:
        """Convert pandas-style conditionals to np.where or np.select"""
        
        # Handle parenthesized if-else expressions for concatenation
        if_else_in_parens = re.compile(r'\(([^()]+)\s+if\s+([^()]+)\s+else\s+([^()]+)\)')
        
        def replace_conditional(match):
            value_expr = match.group(1).strip()
            condition = match.group(2).strip() 
            else_value = match.group(3).strip()
            return f"np.where({condition}, {value_expr}, {else_value})"
        
        # Replace parenthesized if-else with np.where
        result = if_else_in_parens.sub(replace_conditional, formula)
        
        return result
    
    # ----------- COMPLEX CONCATENATION HANDLING -----------
    def _handle_complex_concatenation(self, formula: str) -> str:
        """Handle complex string concatenation with conditionals"""
        
        # Check if this is a complex concatenation with if-else
        if '+' in formula and ' if ' in formula and ' else ' in formula:
            # Pattern to identify: part1 + part2 + (conditional_part)
            complex_concat_pattern = re.compile(
                r'^(.+?)\s*\+\s*(.+?)\s*\+\s*\((.+?)\s+if\s+(.+?)\s+else\s+(.+?)\)$'
            )
            
            match = complex_concat_pattern.match(formula.strip())
            if match:
                part1 = match.group(1).strip()
                part2 = match.group(2).strip()
                conditional_true = match.group(3).strip()
                condition = match.group(4).strip()
                conditional_false = match.group(5).strip()
                
                # Use np.where to handle the conditional part, then concatenate
                return f"{part1} + {part2} + np.where({condition}, {conditional_true}, {conditional_false})"
        
        return formula

    # ----------- INLINE IF-ELSE TO np.select -----------
    def _convert_ifelse(self, formula: str) -> str:
        """Convert inline if-else to np.select with proper nested handling"""
        
        # First handle vectorized conditionals within parentheses
        formula = self._convert_pandas_conditionals(formula)
        
        # Don't convert if-else when it's part of a complex concatenation expression
        # that contains multiple + operations outside of the if-else
        plus_count_outside_parens = 0
        paren_depth = 0
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(formula):
            if char in ['"', "'"] and (i == 0 or formula[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            elif not in_quotes:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '+' and paren_depth == 0:
                    plus_count_outside_parens += 1
        
        # If there are multiple + operations outside parentheses, avoid conversion
        if plus_count_outside_parens > 1:
            return formula
        
        # Parse nested if-else expressions properly
        # Handle patterns like: "A" if cond1 else "B" if cond2 else "C"
        
        def parse_nested_ifelse(text):
            """Parse nested if-else into conditions and values"""
            conditions = []
            values = []
            current = text.strip()
            
            # Pattern to match: value if condition else rest
            pattern = re.compile(r'^(.+?)\s+if\s+(.+?)\s+else\s+(.+)$')
            
            while True:
                match = pattern.match(current)
                if not match:
                    # No more if-else, this is the final default value
                    default = current.strip()
                    break
                    
                value, condition, rest = match.groups()
                conditions.append(condition.strip())
                values.append(value.strip())
                current = rest.strip()
            
            return conditions, values, default
        
        # Check if this looks like a nested if-else
        if ' if ' in formula and ' else ' in formula:
            try:
                conditions, values, default = parse_nested_ifelse(formula)
                
                if conditions:  # We found some conditions
                    # Ensure values are properly formatted (remove any stray parentheses)
                    clean_values = []
                    for val in values:
                        clean_val = val.strip()
                        # Remove any leading parentheses that shouldn't be there
                        while clean_val.startswith('(') and not clean_val.endswith(')'):
                            clean_val = clean_val[1:]
                        clean_values.append(clean_val)
                    
                    return f"np.select([{', '.join(conditions)}], [{', '.join(clean_values)}], default={default})"
            except:
                # If parsing fails, fall back to original formula
                pass
        
        return formula

    # ----------- MAIN EVALUATOR -----------
    def evaluate(self, formula: str, output_col: str, debug: bool = False):
        if debug:
            print(f"Original: {formula}")
        
        safe_formula = self._replace_column_refs(formula)
        if debug:
            print(f"After column refs: {safe_formula}")
            
        safe_formula = self._process_str_function(safe_formula)
        if debug:
            print(f"After str function: {safe_formula}")
            
        safe_formula = self._process_slices(safe_formula)
        if debug:
            print(f"After slices: {safe_formula}")
            
        safe_formula = self._process_null_checks(safe_formula)
        if debug:
            print(f"After null checks: {safe_formula}")
            
        safe_formula = self._process_str_methods(safe_formula)
        if debug:
            print(f"After str methods: {safe_formula}")
            
        safe_formula = self._handle_complex_concatenation(safe_formula)
        if debug:
            print(f"After complex concatenation: {safe_formula}")
            
        if " if " in safe_formula:
            safe_formula = self._convert_ifelse(safe_formula)
            if debug:
                print(f"After ifelse: {safe_formula}")
            
        safe_formula = self._process_logical_operators(safe_formula)
        if debug:
            print(f"After logical operators: {safe_formula}")
            
        safe_formula = self._balance_parentheses(safe_formula)
        if debug:
            print(f"After parentheses: {safe_formula}")

        local_dict = {
            "df": self.df,
            "np": np,
            "Formula": Formula
        }
        
        if debug:
            print(f"Final formula: {safe_formula}")
            
        self.df[output_col] = eval(safe_formula, {}, local_dict)
        return self.df


def generate_test_dataframe(columns_config: dict, num_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a randomized DataFrame for benchmarking the Formula class.
    
    Parameters:
    -----------
    columns_config : dict
        Dictionary where keys are column names and values are data types.
        Supported types: 'int', 'float', 'string', 'bool', 'date', 'nullable_int', 'nullable_string'
    num_rows : int
        Number of rows to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame : Generated test DataFrame
    
    Example:
    --------
    config = {
        'id': 'int',
        'name': 'string', 
        'salary': 'float',
        'active': 'bool',
        'hire_date': 'date',
        'department': 'nullable_string'
    }
    df = generate_test_dataframe(config, 500)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    data = {}
    
    for col_name, data_type in columns_config.items():
        if data_type == 'int':
            data[col_name] = np.random.randint(1, 10000, num_rows)
            
        elif data_type == 'float':
            data[col_name] = np.round(np.random.uniform(100.0, 100000.0, num_rows), 2)
            
        elif data_type == 'string':
            # Generate random strings of varying lengths
            strings = []
            for _ in range(num_rows):
                length = random.randint(3, 15)
                strings.append(''.join(random.choices(string.ascii_letters + string.digits, k=length)))
            data[col_name] = strings
            
        elif data_type == 'bool':
            data[col_name] = np.random.choice([True, False], num_rows)
            
        elif data_type == 'date':
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2024, 12, 31)
            dates = []
            for _ in range(num_rows):
                random_days = random.randint(0, (end_date - start_date).days)
                dates.append(start_date + timedelta(days=random_days))
            data[col_name] = dates
            
        elif data_type == 'nullable_int':
            # Create integers with some null values
            values = np.random.randint(1, 1000, num_rows)
            null_indices = np.random.choice(num_rows, size=num_rows//10, replace=False)
            values = values.astype(float)  # Convert to float to allow NaN
            values[null_indices] = np.nan
            data[col_name] = values
            
        elif data_type == 'nullable_string':
            # Create strings with some null/empty values
            strings = []
            for i in range(num_rows):
                rand = random.random()
                if rand < 0.1:  # 10% null
                    strings.append(np.nan)
                elif rand < 0.15:  # 5% empty string
                    strings.append("")
                elif rand < 0.18:  # 3% "null" string
                    strings.append("null")
                else:
                    length = random.randint(3, 12)
                    strings.append(''.join(random.choices(string.ascii_letters, k=length)))
            data[col_name] = strings
            
        elif data_type == 'category':
            # Generate categorical data
            categories = ['A', 'B', 'C', 'D', 'E']
            data[col_name] = np.random.choice(categories, num_rows)
            
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    return pd.DataFrame(data)


def benchmark_formula(formula_text: str, columns_config: dict, num_rows: int = 1000, 
                     output_col: str = "result", iterations: int = 5) -> dict:
    """
    Benchmark a formula against generated test data.
    
    Parameters:
    -----------
    formula_text : str
        The formula to test
    columns_config : dict
        Configuration for generating test DataFrame
    num_rows : int
        Number of rows in test DataFrame
    output_col : str
        Name of the output column
    iterations : int
        Number of iterations to run for timing
        
    Returns:
    --------
    dict : Benchmark results including timing, success/failure, and sample output
    """
    import time
    
    results = {
        'formula': formula_text,
        'config': columns_config,
        'num_rows': num_rows,
        'success': False,
        'error': None,
        'avg_time': None,
        'sample_output': None,
        'output_type': None
    }
    
    try:
        # Generate test data
        df = generate_test_dataframe(columns_config, num_rows)
        formula_engine = Formula(df.copy())
        
        # Time the formula execution
        times = []
        for _ in range(iterations):
            test_df = df.copy()
            test_engine = Formula(test_df)
            
            start_time = time.time()
            result_df = test_engine.evaluate(formula_text, output_col)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Calculate average time
        results['avg_time'] = sum(times) / len(times)
        results['success'] = True
        
        # Get sample output
        final_result = formula_engine.evaluate(formula_text, output_col)
        results['sample_output'] = final_result[output_col].head(10).tolist()
        results['output_type'] = str(final_result[output_col].dtype)
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
    
    return results


def run_formula_tests():
    """
    Run a comprehensive test suite to identify bugs in the Formula class.
    """
    print("=" * 60)
    print("FORMULA ENGINE TEST SUITE")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Basic String Operations',
            'config': {'name': 'string', 'description': 'nullable_string'},
            'formulas': [
                'name.upper()',
                'name[:3]',
                'name.__contains__("a")',
                'IsNull(description)',
                'IsNotNull(name)'
            ]
        },
        {
            'name': 'Numeric Operations',
            'config': {'amount': 'float', 'count': 'int', 'optional_num': 'nullable_int'},
            'formulas': [
                'amount * 1.1',
                'count + 100',
                'amount / count',
                'IsNull(optional_num)'
            ]
        },
        {
            'name': 'Conditional Logic',
            'config': {'score': 'int', 'category': 'category', 'active': 'bool'},
            'formulas': [
                '"High" if score > 500 else "Low"',
                '"Premium" if score > 800 else "Standard" if score > 400 else "Basic"',
                'category == "A"',
                'active & (score > 300)'
            ]
        },
        {
            'name': 'Mixed Data Types',
            'config': {
                'id': 'int',
                'name': 'string', 
                'salary': 'float',
                'dept': 'nullable_string',
                'active': 'bool'
            },
            'formulas': [
                'name + "_" + str(id)',
                'salary * 1.05 if active else salary',
                'IsNotNull(dept) & active',
                'name[:2] + dept[:1] if IsNotNull(dept) else name[:3]'
            ]
        },
        {
            'name': 'Edge Cases & Bug Tests',
            'config': {
                'text': 'string',
                'num': 'int',
                'nullable_text': 'nullable_string',
                'decimal': 'float'
            },
            'formulas': [
                # Test str() conversion
                'text + str(num)',
                '"Value: " + str(decimal)',
                # Test multiple conditions
                '"A" if num > 5000 else "B" if num > 1000 else "C"',
                # Test null handling in conditions  
                'text if IsNotNull(nullable_text) else "DEFAULT"',
                # Test string operations on nullable columns
                'nullable_text.upper() if IsNotNull(nullable_text) else "EMPTY"',
                # Test slicing with nullable strings
                'nullable_text[:2] if IsNotNull(nullable_text) else "NA"',
                # Test complex combinations
                'text[:1] + str(num)[:2] + ("_" + nullable_text[:1] if IsNotNull(nullable_text) else "_X")'
            ]
        },
        {
            'name': 'Mathematical Operations',
            'config': {'x': 'float', 'y': 'int', 'z': 'nullable_int'},
            'formulas': [
                'x + y',
                'x * y / 100',
                'x ** 2',
                'x + y if IsNotNull(z) else x - y',
                # Test division by zero protection
                'x / y if y != 0 else 0'
            ]
        },
        {
            'name': 'String Method Tests',
            'config': {'text1': 'string', 'text2': 'nullable_string'},
            'formulas': [
                'text1.lower()',
                'text1.replace("a", "X")',
                'text1.startswith("A")',
                'text1.endswith("z")',
                'text2.strip() if IsNotNull(text2) else ""',
                # Test chained operations
                'text1.upper().replace("A", "1")'
            ]
        }
    ]
    
    all_results = []
    
    for test_group in test_configs:
        print(f"\n{test_group['name']}")
        print("-" * 40)
        
        for formula in test_group['formulas']:
            print(f"\nTesting: {formula}")
            result = benchmark_formula(
                formula, 
                test_group['config'], 
                num_rows=100,  # Smaller for testing
                iterations=3
            )
            
            if result['success']:
                print(f"✓ SUCCESS - Time: {result['avg_time']:.4f}s")
                print(f"  Output type: {result['output_type']}")
                print(f"  Sample: {result['sample_output'][:3]}")
            else:
                print(f"✗ FAILED - Error: {result['error']}")
            
            all_results.append(result)
    
    # Summary
    successful = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successful}/{total} tests passed")
    print(f"{'='*60}")
    
    return all_results


# Example usage and quick test
if __name__ == "__main__":
    # Quick example
    config = {
        'name': 'string',
        'age': 'int', 
        'salary': 'float',
        'department': 'nullable_string',
        'active': 'bool'
    }
    
    print("Generating sample DataFrame...")
    df = generate_test_dataframe(config, 10)
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    print("\nRunning comprehensive tests...")
    run_formula_tests()