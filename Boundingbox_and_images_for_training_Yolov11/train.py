# ---- set env first (Windows + Intel OMP) ----
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# os.environ["MPLBACKEND"] = "Agg"   # disable interactive plots
# os.environ["OMP_NUM_THREADS"] = "8"  # optional: limit BLAS threads

from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    model = YOLO("yolo11n-pose.pt")
    model.train(
        data="./myproj/multi_user_review.yaml",
        epochs=100,
        imgsz=960,        # try 640 if you want faster
        batch=16,         # tune as needed
        device=0,         # GPU
        workers=0,        # set 0 to avoid Windows spawn issues
        cache="ram",
        amp=True,
        plots=False       # avoid plotting (where OMP often triggers)
    )

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
