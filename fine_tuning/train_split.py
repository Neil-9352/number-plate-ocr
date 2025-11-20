import os
import subprocess
import random
import re

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
FILTERED_DIR = os.path.join(BASE_DIR, "boxed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "anpr", "models", "tessdata")
os.makedirs(MODELS_DIR, exist_ok=True)

TRAINEDDATA = os.path.join(BASE_DIR, "eng_best.traineddata")
MAX_ITER = 6000
RANDOM_SEED = 42
TRAIN_RATIO = 0.8

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
    os.makedirs(FILTERED_DIR, exist_ok=True)

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
# STEP 3: Create train/eval listfiles (80/20 split)
# ----------------------------
def collect_lstmf_files():
    files = sorted([
        os.path.join(FILTERED_DIR, f)
        for f in os.listdir(FILTERED_DIR)
        if f.endswith(".lstmf")
    ])
    print(f"Found {len(files)} .lstmf files.")
    if len(files) == 0:
        raise RuntimeError("No .lstmf files found — check your data.")
    return files

def split_and_write_listfiles(files, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    random.seed(seed)
    shuffled = files[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)

    train_files = shuffled[:split_idx]
    eval_files = shuffled[split_idx:]

    train_listfile = os.path.join(OUTPUT_DIR, "train_listfile.txt")
    eval_listfile = os.path.join(OUTPUT_DIR, "eval_listfile.txt")

    with open(train_listfile, "w") as f:
        for p in train_files:
            f.write(p + "\n")

    with open(eval_listfile, "w") as f:
        for p in eval_files:
            f.write(p + "\n")

    print(f"Train: {len(train_files)} files -> {train_listfile}")
    print(f"Eval : {len(eval_files)} files -> {eval_listfile}")

    if len(train_files) == 0:
        raise RuntimeError("Train split is empty — need more data or adjust TRAIN_RATIO.")
    if len(eval_files) == 0:
        print("Warning: Eval split is empty — evaluation will be skipped (consider lowering TRAIN_RATIO).")

    return train_listfile, eval_listfile

# ----------------------------
# STEP 4: Run fine-tuning (train on train_listfile)
# ----------------------------
def run_training(lstm_path, train_listfile):
    model_output = os.path.join(OUTPUT_DIR, "plates")
    cmd = [
        "lstmtraining",
        "--model_output", model_output,
        "--continue_from", lstm_path,
        "--traineddata", TRAINEDDATA,
        "--train_listfile", train_listfile,
        "--max_iterations", str(MAX_ITER)
    ]
    print("Running training...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return model_output

# ----------------------------
# STEP 5: Stop training + package
# ----------------------------
def finalize_model():
    checkpoints = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("plates_checkpoint")
    ]
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
    return final_model

# ----------------------------
# STEP 6: Evaluate with lstmeval and print accuracy
# ----------------------------
def evaluate_model(model_path, eval_listfile):
    """
    Run lstmeval, capture output, parse many possible metrics, and return accuracy (0..1) or None.
    Writes output/accuracy.txt with percentage if parsed.
    """
    if not eval_listfile or not os.path.exists(eval_listfile):
        print("No eval listfile found; skipping evaluation.")
        return None

    if model_path.endswith(".traineddata") and os.path.exists(model_path):
        cmd = ["lstmeval", "--model", model_path, "--eval_listfile", eval_listfile]
    else:
        cmd = ["lstmeval", "--model", model_path, "--traineddata", TRAINEDDATA, "--eval_listfile", eval_listfile]

    print("Running evaluation...")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out = out.strip()

    if out:
        print("--- lstmeval output ---")
        print(out)
        print("--- end output ---")
    else:
        print("lstmeval produced no output on stdout/stderr.")
        return None

    # 1) Prefer BCER eval if present (character error rate in percent)
    m_bcer = re.search(r"BCER\s*eval\s*[=:\s]\s*([0-9]*\.?[0-9]+)", out, flags=re.IGNORECASE)
    if m_bcer:
        try:
            bcer = float(m_bcer.group(1))
            # assume it's percent (e.g. 6.642 means 6.642%)
            accuracy = max(0.0, min(1.0, (100.0 - bcer) / 100.0))
            print(f"\nParsed BCER eval = {bcer} -> Accuracy = {accuracy * 100:.2f}%")
            # write to file
            try:
                acc_file = os.path.join(OUTPUT_DIR, "accuracy.txt")
                with open(acc_file, "w") as f:
                    f.write(f"{accuracy * 100:.4f}%\n")
                print(f"Wrote accuracy to {acc_file}")
            except Exception:
                pass
            return accuracy
        except Exception:
            pass

    # 2) Fallback: BWER eval (word error rate)
    m_bwer = re.search(r"BWER\s*eval\s*[=:\s]\s*([0-9]*\.?[0-9]+)", out, flags=re.IGNORECASE)
    if m_bwer:
        try:
            bwer = float(m_bwer.group(1))
            accuracy = max(0.0, min(1.0, (100.0 - bwer) / 100.0))
            print(f"\nParsed BWER eval = {bwer} -> Word-level Accuracy ≈ {accuracy * 100:.2f}%")
            try:
                acc_file = os.path.join(OUTPUT_DIR, "accuracy.txt")
                with open(acc_file, "w") as f:
                    f.write(f"word_accuracy:{accuracy * 100:.4f}%\n")
                print(f"Wrote accuracy to {acc_file}")
            except Exception:
                pass
            return accuracy
        except Exception:
            pass

    # 3) Other common patterns (Accuracy, Error rate, CER, etc.)
    patterns = [
        (r"Accuracy\s*[=:]\s*([0-9]*\.?[0-9]+)\s*%?", lambda v: float(v) / 100.0 if ("%" in v or float(v) > 1.0) else float(v)),
        (r"Percent\s+correct\s*[=:]\s*([0-9]*\.?[0-9]+)\s*%?", lambda v: float(v) / 100.0 if ("%" in v or float(v) > 1.0) else float(v)),
        (r"Error\s+rate\s*[=:]\s*([0-9]*\.?[0-9]+)\s*%?", lambda v: 1.0 - (float(v) / 100.0 if "%" in v or float(v) > 1.0 else float(v))),
        (r"Character\s+error\s+rate\s*[=:]\s*([0-9]*\.?[0-9]+)\s*%?", lambda v: 1.0 - (float(v) / 100.0 if "%" in v or float(v) > 1.0 else float(v))),
        (r"\bCER\b\s*[=:]\s*([0-9]*\.?[0-9]+)\s*%?", lambda v: 1.0 - (float(v) / 100.0 if "%" in v or float(v) > 1.0 else float(v))),
    ]

    for pat, conv in patterns:
        m = re.search(pat, out, flags=re.IGNORECASE)
        if m:
            try:
                val = m.group(1)
                acc = conv(val)
                if 0.0 <= acc <= 1.0:
                    print(f"\nParsed metric via pattern '{pat}' -> Accuracy = {acc*100:.2f}%")
                    try:
                        acc_file = os.path.join(OUTPUT_DIR, "accuracy.txt")
                        with open(acc_file, "w") as f:
                            f.write(f"{acc * 100:.4f}%\n")
                        print(f"Wrote accuracy to {acc_file}")
                    except Exception:
                        pass
                    return acc
            except Exception:
                continue

    # 4) Try to find a general "error rate" float and convert to accuracy
    m_err = re.search(r"error\s*rate\s*[=:]\s*([0-9]*\.?[0-9]+)", out, flags=re.IGNORECASE)
    if m_err:
        try:
            err = float(m_err.group(1))
            if err > 1.0:
                err = err / 100.0
            accuracy = max(0.0, 1.0 - err)
            print(f"\nParsed error rate = {err} -> Accuracy = {accuracy*100:.2f}%")
            try:
                acc_file = os.path.join(OUTPUT_DIR, "accuracy.txt")
                with open(acc_file, "w") as f:
                    f.write(f"{accuracy * 100:.4f}%\n")
                print(f"Wrote accuracy to {acc_file}")
            except Exception:
                pass
            return accuracy
        except Exception:
            pass

    print("\nCould not parse accuracy from lstmeval output. See raw output above for debugging.")
    return None

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    lstm_path = extract_lstm()
    generate_lstmf()

    all_files = collect_lstmf_files()
    train_listfile, eval_listfile = split_and_write_listfiles(all_files)

    run_training(lstm_path, train_listfile)
    final_traineddata_path = finalize_model()

    accuracy = evaluate_model(final_traineddata_path, eval_listfile)

    if accuracy is not None:
        print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")
    else:
        print("\nEvaluation did not produce an accuracy metric.")

    print("Done.")
