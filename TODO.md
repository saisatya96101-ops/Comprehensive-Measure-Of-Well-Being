# TODO - HDI Predictor (Flask + ML)

## Plan (approved)
1. Create project structure (app code, templates, static, ML training scripts).
2. Add a data training pipeline script:
   - load dataset (CSV path configurable)
   - preprocessing (missing values)
   - model training (e.g., Linear Regression + evaluation)
   - save trained model as pickle.
3. Add Flask web app:
   - home page
   - prediction form
   - prediction result page
   - load the saved pickle model at startup.
4. Add HTML templates + CSS styling.
5. Add optional EDA notebook/script to generate plots (if dataset available).
6. Add README with setup/run instructions.

## Progress
- [x] Create project scaffolding and initial files
- [x] Implement training script (train_model.py)
- [x] Implement Flask app (app.py) + templates/static
- [x] Add README + run instructions
- [ ] Add optional EDA script


