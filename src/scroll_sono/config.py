"""Caminhos, constantes e metadados das colunas do projeto."""

from pathlib import Path

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bedtime_screentime_sleep_debt.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RANDOM_STATE = 42

# Colunas especiais
ID_COL = "user_id"
TARGET = "sleep_debt_category"
TARGET_CLASSES = ["Optimal Recovery", "Mild Deficit", "Moderate Debt", "Severe Sleep Debt"]

# Tipos de atributo
NUMERIC_COLS = [
    "age",
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
    "sleep_latency_min",
    "total_sleep_hours",
    "deep_sleep_pct",
    "rem_sleep_pct",
    "morning_alarm_snoozes",
    "next_day_fatigue_score",
]
NOMINAL_COLS = ["gender", "occupation_type", "primary_bedtime_app"]
ORDINAL_COLS = {"chronotype": ["Morning Lark", "Intermediate", "Night Owl"]}
BINARY_COLS = ["blue_light_filter_active"]
CATEGORICAL_COLS = NOMINAL_COLS + list(ORDINAL_COLS)

# Momento em que cada informação é obtida em relação à noite de sono
MOMENT = {
    "perfil": ["age", "gender", "occupation_type", "chronotype"],
    "antes de dormir": [
        "bedtime_phone_minutes",
        "primary_bedtime_app",
        "screen_brightness_pct",
        "blue_light_filter_active",
        "caffeine_post_5pm_mg",
        "physical_activity_min",
    ],
    "durante o sono": ["sleep_latency_min", "total_sleep_hours", "deep_sleep_pct", "rem_sleep_pct"],
    "dia seguinte": ["morning_alarm_snoozes", "next_day_fatigue_score"],
}

# Cenários de atributos usados como preditores
FEATURES_BEHAVIORAL = MOMENT["perfil"] + MOMENT["antes de dormir"]
FEATURES_FULL = FEATURES_BEHAVIORAL + MOMENT["durante o sono"] + MOMENT["dia seguinte"]
FEATURE_SETS = {"comportamental": FEATURES_BEHAVIORAL, "completo": FEATURES_FULL}

# Domínio esperado de cada coluna numérica: (mínimo, máximo, deve ser inteiro).
# Vem da documentação do dataset ou, quando ela não define, de limites físicos plausíveis.
EXPECTED_DOMAINS = {
    "age": (0, 120, True),
    "bedtime_phone_minutes": (0, 24 * 60, True),
    "screen_brightness_pct": (10, 100, True),
    "blue_light_filter_active": (0, 1, True),
    "caffeine_post_5pm_mg": (0, 1000, False),
    "physical_activity_min": (0, 24 * 60, False),
    "sleep_latency_min": (0, 24 * 60, False),
    "total_sleep_hours": (0, 24, False),
    "deep_sleep_pct": (0, 100, False),
    "rem_sleep_pct": (0, 100, False),
    "morning_alarm_snoozes": (0, None, True),
    "next_day_fatigue_score": (1, 10, False),
}
