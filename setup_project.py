import os

def create_project_structure():
    base_dir = "strategy_engine"
    
    directories =[
        "data",
        "results",
        "config",
        "src"
    ]
    
    files =[
        "src/data_loader.py",
        "src/data_alignment.py",
        "src/indicators.py",
        "src/preconditions.py",
        "src/signals.py",
        "src/strategy_engine.py",
        "src/metrics.py",
        "src/range_tester.py",
        "main.py"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
        
    # Create empty files
    for file in files:
        filepath = os.path.join(base_dir, file)
        with open(filepath, 'a') as f:
            pass # Just touch the file
            
    # Test / Verification Output
    print(f"Project structure created successfully in: {os.path.abspath(base_dir)}")
    print("Base directories:", os.listdir(base_dir))
    print("Src files:", os.listdir(os.path.join(base_dir, "src")))
    print("PASS")

if __name__ == "__main__":
    create_project_structure()