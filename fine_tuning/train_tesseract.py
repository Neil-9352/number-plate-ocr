import os
import subprocess

# ----------------------------
# CONFIG
# ----------------------------
FILTERED_DIR = os.path.join(os.path.dirname(__file__), "boxed_data")            # use boxed data with .box files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.getcwd()                                                      # current working directory = project root
MODELS_DIR = os.path.join(PROJECT_ROOT, "anpr", "models", "tessdata")           # final model
os.makedirs(MODELS_DIR, exist_ok=True)

TRAINEDDATA = os.path.join(os.path.dirname(__file__), "eng_best.traineddata")   # base model
MAX_ITER = 6000

# ----------------------------
# STEP 1: Extract eng.lstm
# ----------------------------
def extract_lstm():
    lstm_path = os.path.join(OUTPUT_DIR, "eng.lstm")
    if not os.path.exists(lstm_path):
        print("Extracting LSTM network from eng.traineddata...")
        subprocess.run([
            "combine_tessdata", "-e", TRAINEDDATA, lstm_path
        ], check=True)
    else:
        print("eng.lstm already exists.")
    return lstm_path

# ----------------------------
# STEP 2: Generate .lstmf files
# ----------------------------
def generate_lstmf():
    valid_exts = [".png", ".jpg", ".jpeg"]
    for img_name in os.listdir(FILTERED_DIR):
        if not any(img_name.lower().endswith(ext) for ext in valid_exts):
            continue
        img_path = os.path.join(FILTERED_DIR, img_name)
        base = os.path.splitext(img_name)[0]
        lstmf_file = os.path.join(FILTERED_DIR, f"{base}.lstmf")
        if os.path.exists(lstmf_file):
            continue
        print(f"Generating LSTM features for {img_name}...")
        subprocess.run([
            "tesseract", img_path, os.path.join(FILTERED_DIR, base),
            "--psm", "7", "lstm.train"
        ], check=True)

# ----------------------------
# STEP 3: Create listfile.txt
# ----------------------------
def create_listfile():
    listfile = os.path.join(OUTPUT_DIR, "listfile.txt")
    with open(listfile, "w") as f:
        for fname in os.listdir(FILTERED_DIR):
            if fname.endswith(".lstmf"):
                f.write(os.path.join(FILTERED_DIR, fname) + "\n")
    count = sum(1 for fname in os.listdir(FILTERED_DIR) if fname.endswith(".lstmf"))
    print(f"listfile.txt created with {count} entries.")
    if count == 0:
        raise RuntimeError("No .lstmf files found — check your data.")
    return listfile

# ----------------------------
# STEP 4: Run fine-tuning
# ----------------------------
def run_training(lstm_path, listfile):
    model_output = os.path.join(OUTPUT_DIR, "plates")
    cmd = [
        "lstmtraining",
        "--model_output", model_output,
        "--continue_from", lstm_path,
        "--traineddata", TRAINEDDATA,
        "--train_listfile", listfile,
        "--max_iterations", str(MAX_ITER)
    ]
    print("Running training...")
    subprocess.run(cmd, check=True)

# ----------------------------
# STEP 5: Stop training + package
# ----------------------------
def finalize_model():
    checkpoints = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.startswith("plates_checkpoint")]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in output/")
    checkpoint = sorted(checkpoints)[-1]

    final_model = os.path.join(MODELS_DIR, "plates.traineddata")
    cmd = [
        "lstmtraining", "--stop_training",
        "--continue_from", checkpoint,
        "--traineddata", TRAINEDDATA,
        "--model_output", final_model
    ]
    print("Finalizing model...")
    subprocess.run(cmd, check=True)
    print(f"Model ready at {final_model}")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    lstm_path = extract_lstm()
    generate_lstmf()
    listfile = create_listfile()
    run_training(lstm_path, listfile)
    finalize_model()
