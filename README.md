# Aura Gold AI

Aura Gold AI is a research system for forecasting the next completed hourly candle of gold futures using Yahoo Finance GC=F. It records forecasts before the target and joins outcomes only when the matching candle is available. The project supports a historical defense demonstration and out-of-sample evaluation.

يدرس المشروع توقع إغلاق شمعة الساعة التالية لعقود الذهب باستخدام سلسلة GC=F. يسجل كل توقع ومرجعه وهدفه قبل الاستحقاق، ثم يحسب خطأه بعد توفر الشمعة المطابقة.

- [Current site](https://vmi3769572.contaboserver.net/) · [Historical defense dashboard](https://vmi3769572.contaboserver.net/demo/)
- [Architecture](docs/ARCHITECTURE.md) · [Install](docs/INSTALLATION.md) · [Deployment](docs/DEPLOYMENT.md)
- [Models](docs/AI_MODELS.md) · [Data](docs/DATA_SOURCES.md) · [Tests](docs/TESTING.md)
- [Scientific status](docs/PROJECT_STATUS.md) · [Engineering report](docs/REPORT.md)

Python, FastAPI, SQLite, pandas, scikit-learn, CatBoost and PyTorch support the application and model evaluation. The production champion is original. Foundation-model adapters are research comparators. Targets spanning market breaks follow the candle calendar. Short published observations permit descriptive comparisons, including lower drift20 MAE in the snapshot, while an independent holdout is required for a claim of superiority. Future work includes additional authorized contract data and independent evaluation per forecast horizon.
