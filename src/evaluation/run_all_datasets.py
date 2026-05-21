import subprocess
import time
import sys

# List of all dataset wavelength ranges defined in your mamba_training.py
DATASETS = [
    "loop_1mm",
    "loop_5mm",
    "loop_10mm",
    "loop_14mm",
]


extra_args = ["--epochs", "50"]


def main():
    print(f"Starting automated Mamba training sweep across {len(DATASETS)} datasets...")
    print("Results will be saved to the results/ directory.\n")

    for dataset in DATASETS:
        print(f"Starting Training for Dataset: {dataset}")
        
        # Create a unique run_id using a timestamp to keep results organized
        run_id = f"batch_sweep_{int(time.time())}"
        
        # Construct the command list
        command = [
            sys.executable, # Uses the exact same python interpreter running this script
            "src/training/mamba_training.py", 
            "--wavelength-range", dataset,
            "--run-id", run_id
        ] + extra_args
        
        try:
            # Run the command and wait for it to finish
            subprocess.run(command, check=True)
            print(f"\nFinished training on {dataset}.\n")
            
        except subprocess.CalledProcessError as e:
            # If the training script crashes, this catches the error
            print(f"\nError occurred while training on {dataset}.")
            print(f"Command failed with exit status {e.returncode}.")
            print("Halting the remaining batch runs.")
            break
            
        except KeyboardInterrupt:
            # Allows you to Ctrl+C out of the entire sweep cleanly
            print("\nSweep manually interrupted by user. Exiting.")
            break

    print("Sweep process finished!")

if __name__ == "__main__":
    main()