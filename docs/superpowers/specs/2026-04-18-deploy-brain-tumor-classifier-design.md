# Deploy Brain Tumor Classifier — Design

**Date:** 2026-04-18
**Status:** Approved — ready for implementation planning

## Goal

Deploy the existing Streamlit-based Brain Tumor Classifier to a free, publicly accessible URL so anyone can upload an MRI image and get a tumor/no-tumor prediction.

## Context

- **Codebase:** `C:\Users\amitc\Documents\Projects\amit-test-project` (public GitHub repo `amitco2707/Brain-Tumor-Classifier`, branch `master`).
- **App:** `streamlit_app.py` — Streamlit UI that loads a fine-tuned ResNet-18 model from `outputs/brain_tumor_model.pth` (~45 MB) and classifies uploaded MRI images.
- **Model state:** Trained weights already exist locally. File is currently excluded by `.gitignore` (`outputs/*.pth`).
- **Dependencies:** `requirements.txt` pins versions that appear likely-invalid (e.g. `torch==2.11.0`, `streamlit==1.56.0`, `Pillow==12.2.0`). Must be verified against PyPI before deployment.

## Decisions

### Hosting: Streamlit Community Cloud
- Free for public repos.
- Auto-deploys on `git push`.
- Public URL in the form `<user>-<repo>.streamlit.app`.
- No custom infrastructure to manage.

Rejected alternatives: Hugging Face Spaces (comparable, but Streamlit Cloud is the native host for a Streamlit app); Render/Railway (unnecessary complexity for this app); Docker/VM (not free, more setup).

### Model distribution: commit `.pth` directly to the repo
- File is 45 MB — well under GitHub's 100 MB single-file limit, so Git LFS is not needed.
- Simplest path: nothing changes in `streamlit_app.py` (it already loads from the local path).
- One-time repo size increase of ~45 MB is acceptable for this project.

Rejected alternatives: Git LFS (flaky with Streamlit Cloud, added setup); external storage + download-at-startup (more code, first-run latency, another service to manage).

### Repo visibility: public (unchanged)
- Streamlit Community Cloud is free for public repos.

## Implementation outline

Work happens in this order. Each step is small enough to plan/test individually.

1. **Verify and fix `requirements.txt`.**
   - Check every pinned version against PyPI.
   - Replace invalid versions with the latest stable release compatible with Streamlit Cloud (Python 3.11 default).
   - Use CPU-only PyTorch wheel to stay well under Streamlit Cloud's image size limit (~1 GB).
   - Smoke-test install locally (`pip install -r requirements.txt` in a clean venv) before pushing.

2. **Include the trained model in git.**
   - Edit `.gitignore`: keep `outputs/*.pth` (so future retraining runs aren't accidentally committed) but add an explicit un-ignore line `!outputs/brain_tumor_model.pth` so that specific file is tracked.
   - `git add outputs/brain_tumor_model.pth` and commit.
   - Verify with `git ls-files outputs/` before pushing.

3. **Pin the Python version.**
   - Streamlit Community Cloud reads Python version from the app's Advanced Settings UI (selectable from a dropdown, no file needed). Set it to 3.11 when creating the app in step 5.
   - No file change is required for this step; it's a configuration choice made in the Streamlit Cloud web UI.

4. **Push to GitHub.**
   - After steps 1–3 are committed on `master`, push the branch.

5. **Create the Streamlit Cloud app.**
   - Sign in to `share.streamlit.io` with GitHub.
   - New app → pick `amitco2707/Brain-Tumor-Classifier`, branch `master`, main file `streamlit_app.py`.
   - Deploy. First build will install dependencies and launch.

6. **Smoke test the live app.**
   - Open the generated public URL.
   - Upload a sample MRI image from the local `data/` folder.
   - Verify: page loads, model prediction returns, result banner + confidence bars render correctly for both tumor and no-tumor cases.

## Risks and mitigations

- **Dependency resolution fails during Streamlit Cloud build.**
  Mitigation: verify every version against PyPI and smoke-test `pip install -r requirements.txt` in a clean local venv before pushing.

- **PyTorch wheel bloats the image.**
  Mitigation: install CPU-only torch (no CUDA). Typical size ~200 MB; total image well under the 1 GB limit.

- **Model file committed to the wrong branch or missed by `.gitignore` edit.**
  Mitigation: after commit, verify with `git ls-files | grep brain_tumor_model.pth` before pushing.

- **App crashes on first load because the model path differs on the server.**
  Mitigation: `streamlit_app.py` uses `config.OUTPUTS_DIR`, which is relative to the file via `os.path.dirname(__file__)`. This works identically on Streamlit Cloud. No code change needed, but confirm during smoke test.

## Out of scope

- Custom domain.
- Authentication / access control.
- Analytics / logging.
- Auto-retraining pipelines.
- Any changes to the model itself or the training code.
- Paid hosting tiers.

## Success criteria

- Public URL is reachable.
- Uploading an MRI image returns a prediction with confidence bars — no runtime errors in the Streamlit Cloud logs.
- Both "Tumor" and "No Tumor" visual states render correctly.
