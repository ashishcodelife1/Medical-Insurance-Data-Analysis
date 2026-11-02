# MEDINSIGHT AI - Medical Insurance Cost Prediction System

A Streamlit web app that estimates annual medical insurance charges (USD) with a scikit-learn **Linear Regression** pipeline.
**Educational estimate only. Not an actual insurance quote.**

> Verification note: `train_model.py` and `src/model_utils.py` were run on temporary synthetic data during development
> (then deleted). The Streamlit interface (`app.py`) was only syntax-checked because Streamlit was not installable in that
> environment. Run it locally and report any error you see.

## Files
| File | You must... |
|---|---|
| `app.py`, `train_model.py`, `src/`, `assets/custom.css`, `.streamlit/config.toml`, `requirements.txt`, `.gitignore`, docs | Already provided - just place them |
| `data/insurance.csv` | **Download yourself** (see below) |
| `models/insurance_model.joblib`, `models/metrics.json` | **Generated automatically** by `python train_model.py` |
| `screenshots/*.png` | Take them yourself from the running app |

## Dataset
- Name: **Medical Cost Personal Datasets** (`insurance.csv`), originally from Brett Lantz, *Machine Learning with R*; hosted on Kaggle (search "Medical Cost Personal Datasets", author Miri Choi).
- Alternative copy: `https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv` (open in browser, Ctrl+S, save as `insurance.csv`). Availability of links is not guaranteed; if one fails, use the other.
- Save to `data/insurance.csv`. Expected columns:
```
age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,southwest,16884.924
18,male,33.77,1,no,southeast,1725.5523
```
- If the file is missing or invalid, the app shows a clear message; nothing is invented.

## Windows + VS Code setup (run each command in the VS Code terminal)
1. **Create folder**: e.g. `C:\Users\<you>\Documents\Medical-Insurance-Prediction`. Put all provided files inside, keeping subfolders.
2. **Open in VS Code**: File > Open Folder > select that folder. Open terminal: Terminal > New Terminal. Expected: prompt shows the project path.
3. **Check Python**: `python --version` (expect `Python 3.10`-`3.12` recommended). If not found, try `py --version`; if still missing, install from python.org and tick "Add python.exe to PATH".
4. **Create virtual environment**: `python -m venv .venv` (use `py -m venv .venv` if you used `py`). A `.venv` folder appears.
5. **Activate (PowerShell)**: `.\.venv\Scripts\Activate.ps1`. Expected: `(.venv)` appears at the prompt start.
   - If you see "running scripts is disabled": run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` (affects only this terminal window), then activate again.
   - Safe alternative: open **Command Prompt** (Terminal > dropdown > Command Prompt) and run `.venv\Scripts\activate.bat`.
6. **Install packages**: `python -m pip install -r requirements.txt` (needs internet once). Expected: "Successfully installed ...".
7. **Add dataset** at `data/insurance.csv` (see above).
8. **Create files**: copy each provided file to the path shown in the structure below (VS Code: right-click Explorer > New File / New Folder).
9. **Train**: `python train_model.py`. Expected: row counts, MAE/RMSE/R2, and "Saved model -> ...". Creates `models/`.
10. **Launch**: `python -m streamlit run app.py`
11. **Open**: the terminal prints `Local URL: http://localhost:8501`; Ctrl+click it or paste it in your browser.
12. **Predict**: go to **Prediction**, set age, sex, BMI, children, smoker, region, press **Predict Insurance Cost**. A green result card appears on the right with the USD estimate, input summary and disclaimer. **Reset** restores defaults.
13. **Metrics/charts**: **Model Performance** shows MAE, RMSE, R2 and the actual-vs-predicted chart; **Data Explorer** shows table, types, missing values, statistics and four charts.
14. **Stop**: click the terminal and press `Ctrl+C`. If you retrain, restart the app.

```
Medical-Insurance-Prediction/
  app.py  train_model.py  requirements.txt  README.md  .gitignore
  .streamlit/config.toml
  data/insurance.csv            (you download)
  models/                       (generated)
  src/__init__.py  src/model_utils.py
  assets/custom.css
  reports/project_report.md
  presentation/presentation_outline.md
  screenshots/README.md
```

## What each screen looks like
- **Home**: navy header with MEDINSIGHT AI, hero panel, three cards (dataset size, model type, evaluation status - all from real data), "Start a prediction" button.
- **Prediction**: two columns - form on the left (6 fields, Predict + Reset), result on the right (empty state before the first prediction).
- **Model Performance**: three metric cards, actual-vs-predicted scatter with a dashed ideal line, plain-language explanation.
- **Data Explorer**: row/column cards, preview, dtypes, missing values, statistics, histogram, BMI scatter, age scatter, smoker box plot.
- **About / How It Works**: project text, limitations, ethics, and the 10-step workflow diagram.
After clicking **Predict**: inputs are validated, a spinner shows, the saved pipeline predicts (no retraining), and the result card appears. Real screenshots are not included; capture your own.

## Testing checklist
| Test | How | Expected |
|---|---|---|
| Missing dataset | Rename `data/insurance.csv`, run train / open app | Clear "file not found" message |
| Invalid columns | Rename a header (e.g. `bmi` -> `bmx`) | Message listing missing column |
| Missing values | Blank a cell | "contains missing values" message |
| Invalid inputs | Try age 5 or BMI 80 | Field limits / red error, no prediction |
| Valid categories | Try each dropdown value | Prediction works |
| Training | `python train_model.py` | Metrics printed, files created |
| Model/metrics missing | Delete `models/` contents | Yellow warning with train command |
| Prediction | Default inputs | Finite USD number |
| After prediction | Switch pages and back | App works, result still shown |
| Reset | Press Reset | Inputs default, result cleared |
| Navigation | Click each of 6 pages | Each page renders |
| Launch | Documented commands | App opens at localhost:8501 |
Restore any file you broke afterwards.

## Troubleshooting
- `python` not recognised: use `py`, or reinstall Python with PATH ticked.
- `No module named streamlit/sklearn`: venv not active (no `(.venv)`) - activate, then reinstall.
- `ModuleNotFoundError: src`: run commands from the project root (the folder containing `app.py`).
- Port busy: `python -m streamlit run app.py --server.port 8502`.
- Debugging traces: PowerShell `$env:MEDINSIGHT_DEBUG="1"`, then run the app.
- **Asking Claude for a fix**: copy the *full* terminal error text, say which command you ran and your Python version (`python --version`), and paste the related file if asked.

## Submission checklist
- [ ] Dataset downloaded, model trained, app runs
- [ ] Screenshots saved in `screenshots/` and inserted in the report
- [ ] Actual MAE/RMSE/R2 and dataset size copied from the app into the report and slides
- [ ] Report converted to Word/PDF; slides built from the outline
- [ ] Viva answers practised in your own words
- [ ] Dataset credited (Lantz / Kaggle) in references
