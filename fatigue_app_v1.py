import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import math
import os


# =============================
# KONFIGURATION
# =============================

HISTORY_FILE = "fatigue_measurements_history.csv"
GOOGLE_SHEET_NAME = "fatigue_measurements"
GOOGLE_WORKSHEET_NAME = "measurements"


# =============================
# GOOGLE SHEETS OPTIONAL
# =============================

def google_sheets_configured() -> bool:
    """True, wenn Streamlit Secrets für Google Sheets vorhanden sind."""
    try:
        return "gcp_service_account" in st.secrets
    except Exception:
        return False


def get_google_worksheet():
    """Öffnet das konfigurierte Google Sheet. Fallback erfolgt ausserhalb dieser Funktion."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes,
        )
        client = gspread.authorize(credentials)

        sheet_name = st.secrets.get("google_sheet", {}).get("sheet_name", GOOGLE_SHEET_NAME)
        worksheet_name = st.secrets.get("google_sheet", {}).get("worksheet_name", GOOGLE_WORKSHEET_NAME)

        spreadsheet = client.open(sheet_name)
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except Exception:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=2000, cols=80)
        return worksheet
    except Exception as e:
        st.warning(f"Google-Sheet-Speicherung nicht verfügbar. Lokale CSV wird verwendet. Grund: {e}")
        return None


def load_google_history() -> pd.DataFrame:
    worksheet = get_google_worksheet()
    if worksheet is None:
        return pd.DataFrame()
    rows = worksheet.get_all_records()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def append_google_measurement(row: dict) -> bool:
    worksheet = get_google_worksheet()
    if worksheet is None:
        return False

    existing_values = worksheet.get_all_values()
    row_headers = list(row.keys())

    if not existing_values:
        headers = row_headers
        worksheet.append_row(headers)
    else:
        headers = existing_values[0]
        missing_headers = [h for h in row_headers if h not in headers]
        if missing_headers:
            headers = headers + missing_headers
            worksheet.update("1:1", [headers])

    worksheet.append_row([row.get(h, "") for h in headers])
    return True


# =============================
# HILFSFUNKTIONEN FÜR VERLAUF UND TRENDS
# =============================

def load_history() -> pd.DataFrame:
    """Lädt Verlauf: bevorzugt Google Sheet, sonst lokale CSV."""
    if google_sheets_configured():
        df_google = load_google_history()
        if not df_google.empty:
            return df_google

    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False)


def save_single_measurement(row: dict) -> str:
    """Speichert eine Messung: zuerst Google Sheet, sonst lokale CSV."""
    if google_sheets_configured():
        if append_google_measurement(row):
            return "Google Sheet"

    df_existing = load_history()
    df_new = pd.DataFrame([row])
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    save_history(df_all)
    return "lokale CSV"


def ln_rmssd(value: float) -> float:
    return math.log(max(float(value), 1.0))


def compute_hrv_trend(history_df: pd.DataFrame, current_rmssd: float, user_id: str) -> dict:
    """
    HRV-Trendlogik:
    - interne Auswertung mit LnRMSSD
    - mindestens 3 gültige Messungen für einen ersten Trend
    - 7-Messpunkt-Rolling-Average
    - individuelle Baseline als bis zu 14 letzte Messungen
    - SWC als 0.5 * SD der LnRMSSD-Baseline, Minimum 0.03
    """
    current_row = pd.DataFrame([{
        "Zeitpunkt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "User-ID": user_id,
        "RMSSD": current_rmssd,
        "LnRMSSD": ln_rmssd(current_rmssd),
    }])

    if history_df is None or history_df.empty:
        temp_df = current_row.copy()
    else:
        temp_df = history_df.copy()
        if "User-ID" in temp_df.columns:
            temp_df = temp_df[temp_df["User-ID"].astype(str) == str(user_id)]

        if "LnRMSSD" not in temp_df.columns and "RMSSD" in temp_df.columns:
            temp_df["LnRMSSD"] = temp_df["RMSSD"].apply(ln_rmssd)

        keep_cols = [c for c in ["Zeitpunkt", "User-ID", "RMSSD", "LnRMSSD"] if c in temp_df.columns]
        temp_df = temp_df[keep_cols]
        temp_df = pd.concat([temp_df, current_row], ignore_index=True)

    temp_df = temp_df.dropna(subset=["LnRMSSD"])
    n = len(temp_df)

    if n < 3:
        return {
            "valid_measurements": n,
            "ln_rmssd_current": ln_rmssd(current_rmssd),
            "ln_rmssd_rolling": None,
            "ln_rmssd_baseline": None,
            "swc": None,
            "trend_delta": None,
            "status": "nicht genügend Messungen",
            "badness": 0.0,
            "explanation": "Für eine erste HRV-Trendbewertung werden mindestens 3 gültige Messungen pro User benötigt.",
        }

    rolling_window = min(7, n)
    baseline_window = min(14, n)

    ln_rolling = temp_df["LnRMSSD"].tail(rolling_window).mean()
    baseline_values = temp_df["LnRMSSD"].tail(baseline_window)
    baseline_mean = baseline_values.mean()
    baseline_sd = baseline_values.std(ddof=1)

    swc = 0.03 if pd.isna(baseline_sd) or baseline_sd == 0 else max(0.03, 0.5 * baseline_sd)
    trend_delta = ln_rolling - baseline_mean
    lower = baseline_mean - swc
    upper = baseline_mean + swc

    if ln_rolling < lower:
        status = "unter Baseline"
        badness = min(85, 35 + ((lower - ln_rolling) / swc) * 35)
        explanation = "Der 7-Messpunkt-LnRMSSD-Trend liegt unter der individuellen Veränderungsschwelle. Intensive Belastung sollte reduziert werden."
    elif ln_rolling > upper:
        status = "über Baseline - Kontext prüfen"
        badness = 10
        explanation = "Der 7-Messpunkt-LnRMSSD-Trend liegt über der individuellen Veränderungsschwelle. Das ist nicht automatisch positiv und sollte mit Load, Ruhepuls und Müdigkeit geprüft werden."
    else:
        status = "stabil"
        badness = 0
        explanation = "Der 7-Messpunkt-LnRMSSD-Trend liegt innerhalb der individuellen Veränderungsschwelle."

    return {
        "valid_measurements": n,
        "ln_rmssd_current": ln_rmssd(current_rmssd),
        "ln_rmssd_rolling": ln_rolling,
        "ln_rmssd_baseline": baseline_mean,
        "swc": swc,
        "trend_delta": trend_delta,
        "status": status,
        "badness": float(badness),
        "explanation": explanation,
    }


# =============================
# DATA STRUCTURE
# =============================

@dataclass
class UserInputV3:
    acute_load: float
    chronic_load: float
    resting_hr: float
    rmssd: float
    baseline_resting_hr: float
    baseline_rmssd: float
    measurement_context: str
    respiratory_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    baseline_systolic_bp: Optional[float] = None
    baseline_diastolic_bp: Optional[float] = None
    general_fatigue: int = 1
    muscle_soreness: int = 1
    mental_stress: int = 1
    illness: int = 1
    sleep_quality: int = 10
    mood: int = 10
    hrv_trend_status: str = "nicht genügend Messungen"
    hrv_trend_badness: float = 0.0
    hrv_valid_measurements: int = 0


# =============================
# MODEL
# =============================

class FatigueProfilerV3:
    def __init__(self, data: UserInputV3):
        self.d = data

    def clamp(self, v, lo, hi):
        return max(lo, min(v, hi))

    def good_from_1_to_10(self, x):
        return ((x - 1) / 9) * 100

    def bad_from_1_to_10(self, x):
        return ((x - 1) / 9) * 100

    def hrv_ratio(self):
        return self.d.rmssd / self.d.baseline_rmssd

    def hr_delta(self):
        return self.d.resting_hr - self.d.baseline_resting_hr

    def acwr(self):
        return self.d.acute_load / self.d.chronic_load

    def hrv_badness(self):
        single_day_badness = self.clamp((1 - self.hrv_ratio()) * 140, 0, 100)
        if self.d.hrv_valid_measurements >= 3:
            return self.clamp(single_day_badness * 0.70 + self.d.hrv_trend_badness * 0.30, 0, 100)
        return single_day_badness

    def hr_badness(self):
        return self.clamp(self.hr_delta() * 10, 0, 100)

    def load_badness(self):
        ratio = self.acwr()
        if ratio <= 1.0:
            return 0
        if ratio >= 1.5:
            return 100
        return self.clamp((ratio - 1.0) / 0.5 * 100, 0, 100)

    def respiratory_badness(self):
        rr = self.d.respiratory_rate
        if rr is None:
            return 0
        if 10 <= rr <= 16:
            return 0
        if 8 <= rr < 10:
            return 15
        if 16 < rr <= 18:
            return 30
        if 18 < rr <= 22:
            return 60
        if rr > 22:
            return 85
        return 40

    def bp_badness(self):
        if None in [self.d.systolic_bp, self.d.diastolic_bp, self.d.baseline_systolic_bp, self.d.baseline_diastolic_bp]:
            return 0
        sys_delta = abs(self.d.systolic_bp - self.d.baseline_systolic_bp)
        dia_delta = abs(self.d.diastolic_bp - self.d.baseline_diastolic_bp)
        return self.clamp(sys_delta * 3 + dia_delta * 4, 0, 100)

    def hrv_score(self):
        return self.clamp(100 - self.hrv_badness(), 0, 100)

    def hr_score(self):
        return self.clamp(100 - self.hr_badness(), 0, 100)

    def load_score(self):
        return self.clamp(100 - self.load_badness(), 0, 100)

    def subjective_score(self):
        fatigue_good = 100 - self.bad_from_1_to_10(self.d.general_fatigue)
        soreness_good = 100 - self.bad_from_1_to_10(self.d.muscle_soreness)
        stress_good = 100 - self.bad_from_1_to_10(self.d.mental_stress)
        illness_good = 100 - self.bad_from_1_to_10(self.d.illness)
        sleep_good = self.good_from_1_to_10(self.d.sleep_quality)
        mood_good = self.good_from_1_to_10(self.d.mood)
        return fatigue_good * 0.20 + soreness_good * 0.12 + stress_good * 0.17 + illness_good * 0.28 + sleep_good * 0.15 + mood_good * 0.08

    def profile_scores(self):
        fatigue_bad = self.bad_from_1_to_10(self.d.general_fatigue)
        soreness_bad = self.bad_from_1_to_10(self.d.muscle_soreness)
        stress_bad = self.bad_from_1_to_10(self.d.mental_stress)
        illness_bad = self.bad_from_1_to_10(self.d.illness)
        sleep_bad = 100 - self.good_from_1_to_10(self.d.sleep_quality)
        mood_bad = 100 - self.good_from_1_to_10(self.d.mood)
        hrv_bad = self.hrv_badness()
        hr_bad = self.hr_badness()
        load_bad = self.load_badness()
        resp_bad = self.respiratory_badness()
        bp_bad = self.bp_badness()
        hrv_trend_bad = self.d.hrv_trend_badness if self.d.hrv_valid_measurements >= 3 else 0

        central = hrv_bad * 0.24 + hr_bad * 0.17 + stress_bad * 0.21 + fatigue_bad * 0.18 + sleep_bad * 0.10 + resp_bad * 0.07 + hrv_trend_bad * 0.03
        muscular = soreness_bad * 0.45 + load_bad * 0.35 + fatigue_bad * 0.20
        illness = illness_bad * 0.65 + hr_bad * 0.15 + hrv_bad * 0.08 + sleep_bad * 0.08 + mood_bad * 0.04
        circulatory = bp_bad
        global_load = central * 0.35 + muscular * 0.25 + illness * 0.25 + circulatory * 0.15
        recovery = 100 - (central * 0.28 + muscular * 0.23 + illness * 0.27 + circulatory * 0.08 + load_bad * 0.14)

        return {
            "Erholungsindex": self.clamp(recovery, 0, 100),
            "Zentrale Erschöpfung": self.clamp(central, 0, 100),
            "Muskuläre Ermüdung": self.clamp(muscular, 0, 100),
            "Globale Belastung": self.clamp(global_load, 0, 100),
            "Kreislaufregulation": self.clamp(circulatory, 0, 100),
            "Krankheitssymptome": self.clamp(illness, 0, 100),
        }

    def dominant_profile(self):
        scores = self.profile_scores()
        sorted_profiles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_profiles[0]
        secondary = sorted_profiles[1]
        confidence = self.clamp(50 + (primary[1] - secondary[1]) * 2, 50, 95)
        return primary[0], confidence, secondary[0], scores

    def drivers(self):
        drivers = []
        if self.hrv_badness() > 40:
            drivers.append("RMSSD deutlich unter Baseline oder HRV-Trend auffällig")
        if self.d.hrv_trend_status == "unter Baseline":
            drivers.append("7-Messpunkt-LnRMSSD-Trend unter individueller Schwelle")
        if self.d.hrv_trend_status == "über Baseline - Kontext prüfen":
            drivers.append("LnRMSSD-Trend über Schwelle: Kontext mit Load, Puls und Müdigkeit prüfen")
        if self.hr_badness() > 40:
            drivers.append("Ruhepuls deutlich über Baseline")
        if self.load_badness() > 40:
            drivers.append("Akuter Load deutlich höher als chronischer Load")
        if self.respiratory_badness() > 40:
            drivers.append("Atemfrequenz auffällig erhöht")
        if self.d.general_fatigue >= 7:
            drivers.append("Allgemeine Müdigkeit hoch")
        if self.d.muscle_soreness >= 7:
            drivers.append("Muskuläre Schmerzen hoch")
        if self.d.mental_stress >= 7:
            drivers.append("Mentaler Stress hoch")
        if self.d.illness >= 3:
            drivers.append("Krankheitssymptome vorhanden")
        if self.d.sleep_quality <= 4:
            drivers.append("Schlafqualität tief")
        if self.d.mood <= 4:
            drivers.append("Stimmung tief")
        if self.bp_badness() > 40:
            drivers.append("Blutdruck deutlich ausserhalb Baseline")
        if not drivers:
            drivers.append("Keine klaren Belastungstreiber erkannt")
        return drivers

    def training_readiness(self):
        score = self.hrv_score() * 0.24 + self.hr_score() * 0.14 + self.load_score() * 0.18 + self.subjective_score() * 0.34 + (100 - self.respiratory_badness()) * 0.08 + (100 - self.d.hrv_trend_badness) * 0.06
        if self.d.illness > 6:
            score = min(score, 25)
        elif 3 <= self.d.illness <= 6:
            score = min(score, 55)
        if self.d.hrv_trend_status == "unter Baseline":
            score = min(score, 65)
        return self.clamp(score, 0, 100)

    def work_readiness(self):
        fatigue_good = 100 - self.bad_from_1_to_10(self.d.general_fatigue)
        stress_good = 100 - self.bad_from_1_to_10(self.d.mental_stress)
        illness_good = 100 - self.bad_from_1_to_10(self.d.illness)
        sleep_good = self.good_from_1_to_10(self.d.sleep_quality)
        mood_good = self.good_from_1_to_10(self.d.mood)
        score = fatigue_good * 0.20 + stress_good * 0.24 + illness_good * 0.25 + sleep_good * 0.19 + mood_good * 0.12
        if self.d.illness > 6:
            score = min(score, 35)
        elif 3 <= self.d.illness <= 6:
            score = min(score, 65)
        return self.clamp(score, 0, 100)

    def recommendation(self, profile):
        if self.d.illness > 6:
            return "Kein Training empfohlen. Krankheitssymptome sind deutlich erhöht. Erst wieder trainieren, wenn die Symptome stark gesunken sind."
        if 3 <= self.d.illness <= 6:
            return "Nur Training mit tiefer Intensität empfohlen. Keine Intervalle, keine harte Kraftbelastung und keine hohe muskuläre Belastung."
        if self.d.hrv_trend_status == "unter Baseline":
            return "Der LnRMSSD-Trend liegt unter der individuellen Schwelle. Heute keine intensive Einheit; besser lockeres Training oder Erholung."
        if self.d.hrv_trend_status == "über Baseline - Kontext prüfen" and (self.d.general_fatigue >= 7 or self.d.acute_load / self.d.chronic_load > 1.3):
            return "Der LnRMSSD-Trend ist erhöht, gleichzeitig gibt es Belastungszeichen. Nicht automatisch als top erholt interpretieren; heute eher kontrolliert trainieren."
        if profile == "Erholungsindex":
            return "Gute bis solide Erholung. Training möglich, Intensität abhängig von Ziel und Load."
        if profile == "Zentrale Erschöpfung":
            return "Heute eher lockeres Training, Stress reduzieren, Schlaf und Regeneration priorisieren."
        if profile == "Muskuläre Ermüdung":
            return "Heute keine harte muskuläre Belastung. Lockeres Training, Mobility oder aktive Erholung."
        if profile == "Globale Belastung":
            return "Mehrere Systeme wirken belastet. Trainingsintensität deutlich reduzieren."
        if profile == "Kreislaufregulation":
            return "Kreislaufregulation auffällig. Belastung vorsichtig wählen und Verlauf beobachten."
        if profile == "Krankheitssymptome":
            return "Krankheitssymptome beachten. Nur sehr lockere Belastung, wenn Symptome gering sind."
        return "Moderate Belastung und Selbstbeobachtung empfohlen."

    def run(self):
        primary, confidence, secondary, scores = self.dominant_profile()
        return {
            "training_readiness": round(self.training_readiness(), 1),
            "work_readiness": round(self.work_readiness(), 1),
            "primary_profile": primary,
            "secondary_profile": secondary,
            "confidence": round(confidence, 1),
            "profile_scores": {k: round(v, 1) for k, v in scores.items()},
            "drivers": self.drivers(),
            "recommendation": self.recommendation(primary),
            "subscores": {
                "HRV Score": round(self.hrv_score(), 1),
                "Ruhepuls Score": round(self.hr_score(), 1),
                "Load Score": round(self.load_score(), 1),
                "Subjective Score": round(self.subjective_score(), 1),
                "Ungewöhnliche Atemfrequenz": round(self.respiratory_badness(), 1),
                "Kreislaufregulation": round(self.bp_badness(), 1),
                "LnRMSSD Trendbelastung": round(self.d.hrv_trend_badness, 1),
            }
        }


# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="Fatigue App V3", page_icon="🏃", layout="wide")
st.title("🏃 Fatigue App V3")
st.caption("Training Readiness, Work Readiness, Fatigue-Profile, HRV-Trends und Google-Sheet-Speicherung")

if google_sheets_configured():
    st.info("Datenspeicherung: Google Sheet ist konfiguriert.")
else:
    st.info("Datenspeicherung: lokale CSV. Für Streamlit Cloud bitte Google-Sheet-Secrets konfigurieren.")

st.sidebar.header("Eingabe")
st.sidebar.subheader("User")
user_id = st.sidebar.text_input("User-ID", value="test_user_01", help="Bitte eindeutige ID verwenden, z.B. athlete_01 oder beni_01.")
st.sidebar.caption("Jede gespeicherte Messung wird mit dieser User-ID abgelegt.")

with st.sidebar.expander("Training Load", expanded=True):
    acute_load = st.number_input("Akuter Load", min_value=1, value=80, step=1)
    chronic_load = st.number_input("Chronischer Load", min_value=1, value=65, step=1)

with st.sidebar.expander("HRV & Herzfrequenz", expanded=True):
    rmssd = st.number_input("Aktuelle RMSSD", min_value=1, value=45, step=1)
    baseline_rmssd = st.number_input("Baseline RMSSD", min_value=1, value=50, step=1)
    resting_hr = st.number_input("Aktueller Ruhepuls", min_value=1, value=52, step=1)
    baseline_resting_hr = st.number_input("Baseline Ruhepuls", min_value=1, value=50, step=1)

with st.sidebar.expander("Messung & Atmung", expanded=True):
    st.caption("Der Messkontext beeinflusst HRV, Herzfrequenz und Atemfrequenz. Für diese Version werden nur Ruhe- oder Schlafmessungen verwendet.")
    measurement_context = st.selectbox("Messkontext", ["rest", "sleep"])
    respiratory_rate = st.number_input("Atemfrequenz", min_value=1, value=14, step=1)

with st.sidebar.expander("Optional: Blutdruck", expanded=False):
    use_bp = st.checkbox("Blutdruck einbeziehen")
    if use_bp:
        systolic_bp = st.number_input("Systolischer Blutdruck", min_value=1, value=120, step=1)
        diastolic_bp = st.number_input("Diastolischer Blutdruck", min_value=1, value=76, step=1)
        baseline_systolic_bp = st.number_input("Baseline systolisch", min_value=1, value=120, step=1)
        baseline_diastolic_bp = st.number_input("Baseline diastolisch", min_value=1, value=76, step=1)
    else:
        systolic_bp = None
        diastolic_bp = None
        baseline_systolic_bp = None
        baseline_diastolic_bp = None

with st.sidebar.expander("Subjektive Faktoren", expanded=True):
    st.caption("Bei Belastungsfaktoren gilt: 1 = kein Problem, 10 = stark ausgeprägt.")
    general_fatigue = st.slider("Allgemeine Müdigkeit", 1, 10, 5)
    muscle_soreness = st.slider("Muskuläre Schmerzen", 1, 10, 5)
    mental_stress = st.slider("Mentaler Stress", 1, 10, 5)
    illness = st.slider("Krankheitssymptome", 1, 10, 1)
    sleep_quality = st.slider("Schlafqualität", 1, 10, 7)
    mood = st.slider("Stimmung", 1, 10, 7)


# Verlauf laden und HRV-Trend berechnen
if "measurements" not in st.session_state:
    st.session_state.measurements = load_history().to_dict("records")

history_df = pd.DataFrame(st.session_state.measurements)
hrv_trend = compute_hrv_trend(history_df, rmssd, user_id.strip())

data = UserInputV3(
    acute_load=acute_load,
    chronic_load=chronic_load,
    resting_hr=resting_hr,
    rmssd=rmssd,
    baseline_resting_hr=baseline_resting_hr,
    baseline_rmssd=baseline_rmssd,
    measurement_context=measurement_context,
    respiratory_rate=respiratory_rate,
    systolic_bp=systolic_bp,
    diastolic_bp=diastolic_bp,
    baseline_systolic_bp=baseline_systolic_bp,
    baseline_diastolic_bp=baseline_diastolic_bp,
    general_fatigue=general_fatigue,
    muscle_soreness=muscle_soreness,
    mental_stress=mental_stress,
    illness=illness,
    sleep_quality=sleep_quality,
    mood=mood,
    hrv_trend_status=hrv_trend["status"],
    hrv_trend_badness=hrv_trend["badness"],
    hrv_valid_measurements=hrv_trend["valid_measurements"],
)

profiler = FatigueProfilerV3(data)
result = profiler.run()


# Messung speichern
st.sidebar.divider()
st.sidebar.subheader("Messung speichern")

if st.sidebar.button("Aktuelle Messung speichern"):
    if user_id.strip() == "":
        st.sidebar.error("Bitte zuerst eine User-ID eingeben.")
    else:
        new_measurement = {
            "Zeitpunkt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "User-ID": user_id.strip(),
            "Messkontext": measurement_context,
            "Akuter Load": acute_load,
            "Chronischer Load": chronic_load,
            "ACWR": round(acute_load / chronic_load, 2),
            "RMSSD": rmssd,
            "LnRMSSD": round(ln_rmssd(rmssd), 4),
            "Baseline RMSSD": baseline_rmssd,
            "Ruhepuls": resting_hr,
            "Baseline Ruhepuls": baseline_resting_hr,
            "Atemfrequenz": respiratory_rate,
            "Allgemeine Müdigkeit": general_fatigue,
            "Muskuläre Schmerzen": muscle_soreness,
            "Mentaler Stress": mental_stress,
            "Krankheitssymptome": illness,
            "Schlafqualität": sleep_quality,
            "Stimmung": mood,
            "Training Readiness": result["training_readiness"],
            "Work Readiness": result["work_readiness"],
            "Primäres Profil": result["primary_profile"],
            "Sekundäres Profil": result["secondary_profile"],
            "Konfidenz": result["confidence"],
            "Empfehlung": result["recommendation"],
            "HRV Trendstatus": hrv_trend["status"],
            "LnRMSSD Rolling": None if hrv_trend["ln_rmssd_rolling"] is None else round(hrv_trend["ln_rmssd_rolling"], 4),
            "LnRMSSD Baseline": None if hrv_trend["ln_rmssd_baseline"] is None else round(hrv_trend["ln_rmssd_baseline"], 4),
            "SWC": None if hrv_trend["swc"] is None else round(hrv_trend["swc"], 4),
        }
        for profile_name, score in result["profile_scores"].items():
            new_measurement[profile_name] = score

        storage_target = save_single_measurement(new_measurement)
        st.session_state.measurements = load_history().to_dict("records")
        st.sidebar.success(f"Messung gespeichert in: {storage_target}.")

if st.sidebar.button("Lokalen Verlauf löschen"):
    st.session_state.measurements = []
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.sidebar.warning("Lokaler Verlauf gelöscht. Google-Sheet-Daten werden nicht gelöscht.")


# =============================
# OUTPUT
# =============================

col1, col2, col3 = st.columns(3)
col1.metric("Training Readiness", f"{result['training_readiness']} / 100")
col2.metric("Work Readiness", f"{result['work_readiness']} / 100")
col3.metric("Konfidenz", f"{result['confidence']} %")

st.divider()
st.subheader("Dominantes Profil")
st.success(f"Primäres Profil: {result['primary_profile']}")
st.info(f"Sekundäres Profil: {result['secondary_profile']}")
st.write("**Empfehlung:**")
st.write(result["recommendation"])
st.write("**Treiber:**")
for driver in result["drivers"]:
    st.write(f"- {driver}")

st.divider()
st.subheader("HRV-Trend nach LnRMSSD")
trend_col1, trend_col2, trend_col3, trend_col4 = st.columns(4)
trend_col1.metric("Gültige Messungen", hrv_trend["valid_measurements"])
trend_col2.metric("Trendstatus", hrv_trend["status"])
trend_col3.metric("7-Messpunkt-LnRMSSD", "-" if hrv_trend["ln_rmssd_rolling"] is None else round(hrv_trend["ln_rmssd_rolling"], 3))
trend_col4.metric("Individuelle Schwelle (SWC)", "-" if hrv_trend["swc"] is None else round(hrv_trend["swc"], 3))
st.caption(hrv_trend["explanation"])

st.divider()
st.subheader("Fatigue Profile Scores")
profile_df = pd.DataFrame(list(result["profile_scores"].items()), columns=["Profil", "Score"])
st.dataframe(profile_df, use_container_width=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(profile_df["Profil"], profile_df["Score"])
ax.set_ylim(0, 100)
ax.set_ylabel("Score")
ax.set_title("Fatigue Profile Scores")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)

st.divider()
st.subheader("Subscores")
sub_df = pd.DataFrame(list(result["subscores"].items()), columns=["Subscore", "Wert"])
st.dataframe(sub_df, use_container_width=True)

st.divider()
st.subheader("Verlauf der Messungen")
history_df = pd.DataFrame(st.session_state.measurements)

if history_df.empty:
    st.info("Noch keine Messungen gespeichert.")
else:
    if "User-ID" in history_df.columns:
        available_users = sorted(history_df["User-ID"].dropna().astype(str).unique())
        selected_user = st.selectbox(
            "User für Verlauf auswählen",
            available_users,
            index=available_users.index(user_id.strip()) if user_id.strip() in available_users else 0,
        )
        display_history_df = history_df[history_df["User-ID"].astype(str) == selected_user].copy()
    else:
        selected_user = "alle"
        display_history_df = history_df.copy()

    st.dataframe(display_history_df, use_container_width=True)
    csv_data = display_history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Verlauf als CSV herunterladen",
        data=csv_data,
        file_name=f"{selected_user}_fatigue_measurements_history.csv",
        mime="text/csv",
    )

    chart_cols = ["Training Readiness", "Work Readiness", "RMSSD", "Ruhepuls", "Krankheitssymptome", "Schlafqualität", "Mentaler Stress"]
    existing_chart_cols = [c for c in chart_cols if c in display_history_df.columns]

    if existing_chart_cols and len(display_history_df) >= 2:
        chart_df = display_history_df.copy()
        chart_df["Zeitpunkt"] = pd.to_datetime(chart_df["Zeitpunkt"])
        chart_df = chart_df.set_index("Zeitpunkt")
        st.line_chart(chart_df[existing_chart_cols])
    else:
        st.info("Für Trenddiagramme braucht es mindestens 2 gespeicherte Messungen pro User.")

    if "LnRMSSD" in display_history_df.columns and len(display_history_df) >= 2:
        st.subheader("LnRMSSD-Verlauf")
        ln_df = display_history_df.copy()
        ln_df["Zeitpunkt"] = pd.to_datetime(ln_df["Zeitpunkt"])
        ln_df = ln_df.set_index("Zeitpunkt")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(ln_df.index, ln_df["LnRMSSD"], marker="o", label="LnRMSSD")
        if len(ln_df) >= 3:
            rolling = ln_df["LnRMSSD"].rolling(window=min(7, len(ln_df)), min_periods=3).mean()
            ax2.plot(ln_df.index, rolling, marker="o", label="Rolling Average")
        ax2.set_ylabel("LnRMSSD")
        ax2.set_title("LnRMSSD-Verlauf")
        ax2.legend()
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig2)

st.divider()
st.caption(
    "Hinweis: Dieses Modell ist ein heuristisches Monitoring-Tool und ersetzt keine medizinische Diagnostik. "
    "HRV wird trendbasiert über LnRMSSD und wiederholte Messungen interpretiert."
)
