import os
import shutil

# Define structure
dirs = [
    'core',
    'app',
    'research',
    'research/notebooks',
    'data',
    'models'
]

# Create directories
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)
        print(f"Created directory: {d}")

# Define moves (source -> dest)
moves = {
    # Core
    'src/database_builder.py': 'core/database_builder.py',
    'src/fetch_external_data.py': 'core/fetch_external_data.py',
    'src/Feature_Engineering_V2.py': 'core/Feature_Engineering_V2.py',
    'src/Train_Universal_Model.py': 'core/Train_Universal_Model.py',
    
    # App
    'src/dashboard.py': 'app/main.py',
    'src/auth_system.py': 'app/auth_system.py', # Might not exist yet
    
    # Research
    'src/Backtest_Universal.py': 'research/Backtest_Universal.py',
    'src/Analyze_Prob_vs_Return.py': 'research/Analyze_Prob_vs_Return.py',
    'src/Final_Performance_Report.py': 'research/Final_Performance_Report.py',
    
    # Notebooks
    'Feature_Analytic.ipynb': 'research/notebooks/Feature_Analytic.ipynb',
    'Stage1_Direction_Optimization.ipynb': 'research/notebooks/Stage1_Direction_Optimization.ipynb',
    'Stage 1 Performance.ipynb': 'research/notebooks/Stage 1 Performance.ipynb',
    'Model_Performance.ipynb': 'research/notebooks/Model_Performance.ipynb',
    'EDA1.ipynb': 'research/notebooks/EDA1.ipynb',
    'Feature_Engineering_Demo.ipynb': 'research/notebooks/Feature_Engineering_Demo.ipynb'
}

for src, dst in moves.items():
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {src} -> {dst}")
    else:
        print(f"Skipped (not found): {src}")

# Create __init__.py for packages
open('core/__init__.py', 'w').close()
open('app/__init__.py', 'w').close()
print("Created __init__.py files.")

# Clean up src if empty
if os.path.exists('src') and not os.listdir('src'):
    os.rmdir('src')
    print("Removed empty src directory.")
