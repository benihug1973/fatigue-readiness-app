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


def training_type_factor(training_type: str) -> float:
    factors = {
        "Ausdauer": 1.00,
        "Recovery Training": 0.70,
        "HIIT": 1.35,
        "Wettkampf": 1.60,
        "Kraft": 1.15,
    }
    return factors.get(training_type, 1.0)


def strength_type_factor(strength_type: str) -> float:
    factors = {
        "Hypertrophie": 1.10,
        "Maximalkraft": 1.25,
        "Kraftausdauer": 1.00,
        "keine Kraftangabe": 1.00,
    }
    return factors.get(strength_type, 1.0)


def calculate_session_rpe_load(duration_min: int, intensity_rpe: int, training_type: str, strength_type: str) -> float:
    """Session-RPE Load: Dauer in Minuten * RPE * Belastungsfaktor."""
    load = duration_min * intensity_rpe * training_type_factor(training_type)
    if training_type == "Kraft":
        load *= strength_type_factor(strength_type)
    return round(load, 1)


def calculate_chronic_training_load(
    weekly_hours: float,
    intensive_sessions_per_week: int,
    strength_sessions_per_week: int,
    dominant_strength_type: str,
) -> dict:
    """Schätzt den chronischen Load aus den letzten 30 Tagen.

    Für die Testversion wird aus Wochenstunden, intensiven Einheiten und Krafttraining
    ein durchschnittlicher Wochenload berechnet. Für das Modell wird daraus ein
    durchschnittlicher Tagesload abgeleitet, damit er mit dem heutigen Session-RPE-Load
    verglichen werden kann.
    """
    weekly_minutes = weekly_hours * 60
    estimated_average_rpe = 4.2
    estimated_average_rpe += min(intensive_sessions_per_week, 6) * 0.45
    estimated_average_rpe += min(strength_sessions_per_week, 6) * 0.15

    if dominant_strength_type == "Maximalkraft":
        estimated_average_rpe += 0.35
    elif dominant_strength_type == "Hypertrophie":
        estimated_average_rpe += 0.25
    elif dominant_strength_type == "Kraftausdauer":
        estimated_average_rpe += 0.15

    estimated_average_rpe = max(2.0, min(8.5, estimated_average_rpe))
    weekly_load = weekly_minutes * estimated_average_rpe
    daily_equivalent_load = weekly_load / 7 if weekly_load > 0 else 1

    return {
        "weekly_load": round(weekly_load, 1),
        "daily_equivalent_load": round(daily_equivalent_load, 1),
        "estimated_average_rpe": round(estimated_average_rpe, 1),
    }


def rr_interval_ms(heart_rate_bpm: float) -> float:
    """R-R-Intervall in Millisekunden aus der Herzfrequenz."""
    return 60000 / max(float(heart_rate_bpm), 1.0)


def compute_hrr_badness(hrr_1min: Optional[float]) -> float:
    """Pragmatische Basisbewertung von HRR60.

    HRR60 = Peak-HF am Ende eines standardisierten Tests minus HF nach 60 Sekunden.
    Wichtig: Diese Funktion liefert nur einen Basiswert. Die App interpretiert HRR
    zusätzlich im Kontext von HRV, Ruhepuls, Blutdruck, Atemfrequenz, Training Load
    und subjektiver Müdigkeit. Sehr tiefe Werte (<= 12 bpm) gelten als deutlich
    auffällig und sollen standardisiert wiederholt werden.
    """
    if hrr_1min is None:
        return 0.0
    if hrr_1min >= 40:
        return 0.0
    if hrr_1min >= 35:
        return 8.0
    if hrr_1min >= 25:
        return 25.0
    if hrr_1min >= 18:
        return 50.0
    if hrr_1min >= 12:
        return 75.0
    return 95.0


def hrr_interpretation_text(hrr_1min: Optional[float]) -> str:
    if hrr_1min is None:
        return "Kein HRR-Test erfasst."
    if hrr_1min >= 40:
        return "HRR60 ist sehr schnell. Meist ein gutes Zeichen, aber immer im Kontext von Load, Müdigkeit und Trainingsphase interpretieren."
    if hrr_1min >= 35:
        return "HRR60 ist gut. Verlauf und Standardisierung beachten."
    if hrr_1min >= 25:
        return "HRR60 ist moderat bis gut. Der Wert ist brauchbar, wenn der Test standardisiert durchgeführt wurde."
    if hrr_1min >= 18:
        return "HRR60 ist verlangsamt. Heute keine maximale Belastung; Test unter gleichen Bedingungen wiederholen."
    if hrr_1min >= 12:
        return "HRR60 ist deutlich verlangsamt. Nur lockere Belastung und den Test standardisiert wiederholen."
    return "HRR60 ist sehr tief und auffällig. Das kann ein Test-/Messfehler sein, bei gleichzeitig hohem Blutdruck oder erhöhter Atemfrequenz aber auch ein echtes Kreislauf-Warnsignal. Heute keine intensive Belastung und Test wiederholen."


def compute_hrv_trend(history_df: pd.DataFrame, current_rmssd: float, current_resting_hr: float, user_id: str) -> dict:
    """
    HRV-Trendlogik nach den aktuellen Studienüberlegungen:
    - interne Auswertung mit LnRMSSD
    - mindestens 3 gültige Messungen für einen ersten Trend
    - 7-Messpunkt-Rolling-Average für Tagessteuerung
    - bis zu 14 letzte Messungen als individuelle Baseline
    - SWC als 0.5 * SD der LnRMSSD-Baseline, Minimum 0.03
    - RMSSD/LnRMSSD-Stabilität über CV/SD als Zusatzmarker
    - R-R-Intervall und LnRMSSD:R-R-Ratio als Plews-orientierter Saturation-Hinweis
    - Wochenmittel der letzten zwei Wochen, sobald genügend Daten vorhanden sind
    """
    current_row = pd.DataFrame([{
        "Zeitpunkt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "User-ID": user_id,
        "RMSSD": current_rmssd,
        "LnRMSSD": ln_rmssd(current_rmssd),
        "Ruhepuls": current_resting_hr,
        "RR_Intervall_ms": rr_interval_ms(current_resting_hr),
    }])

    if history_df is None or history_df.empty:
        temp_df = current_row.copy()
    else:
        temp_df = history_df.copy()
        if "User-ID" in temp_df.columns:
            temp_df = temp_df[temp_df["User-ID"].astype(str) == str(user_id)]

        if "LnRMSSD" not in temp_df.columns and "RMSSD" in temp_df.columns:
            temp_df["LnRMSSD"] = temp_df["RMSSD"].apply(ln_rmssd)
        if "RR_Intervall_ms" not in temp_df.columns and "Ruhepuls" in temp_df.columns:
            temp_df["RR_Intervall_ms"] = temp_df["Ruhepuls"].apply(rr_interval_ms)

        keep_cols = [c for c in ["Zeitpunkt", "User-ID", "RMSSD", "LnRMSSD", "Ruhepuls", "RR_Intervall_ms"] if c in temp_df.columns]
        temp_df = temp_df[keep_cols]
        temp_df = pd.concat([temp_df, current_row], ignore_index=True)

    temp_df = temp_df.dropna(subset=["LnRMSSD"])
    n = len(temp_df)

    # Standardwerte
    result = {
        "valid_measurements": n,
        "ln_rmssd_current": ln_rmssd(current_rmssd),
        "ln_rmssd_rolling": None,
        "ln_rmssd_baseline": None,
        "swc": None,
        "trend_delta": None,
        "status": "nicht genügend Messungen",
        "badness": 0.0,
        "weekly_mean_current": None,
        "weekly_mean_previous": None,
        "weekly_mean_delta": None,
        "lnrmssd_sd_7": None,
        "lnrmssd_cv_7": None,
        "cv_badness": 0.0,
        "rr_interval_current": rr_interval_ms(current_resting_hr),
        "lnrmssd_rr_ratio_current": ln_rmssd(current_rmssd) / rr_interval_ms(current_resting_hr),
        "lnrmssd_rr_ratio_rolling": None,
        "saturation_score": 0.0,
        "freshness_score": 0.0,
        "hrv_outlier_score": 0.0,
        "measurement_warning": "",
        "explanation": "Für eine erste HRV-Trendbewertung werden mindestens 3 gültige Messungen pro User benötigt.",
    }

    if n < 3:
        return result

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

    # Plausibilitätscheck: Ein einzelner sehr starker RMSSD/LnRMSSD-Sprung kann
    # physiologisch echt sein, aber auch durch Messfehler, Sensor-Kontakt, andere
    # Messzeit, Bewegung, Atmung oder Alkohol entstehen. Deshalb wird er als
    # Warnhinweis ausgegeben und nicht blind als definitive Ermüdung gewertet.
    ln_current = ln_rmssd(current_rmssd)
    current_delta = ln_current - baseline_mean
    outlier_limit = max(0.35, 3.0 * swc)
    hrv_outlier_score = 0.0
    measurement_warning = ""
    if abs(current_delta) >= outlier_limit:
        hrv_outlier_score = min(100.0, 55.0 + ((abs(current_delta) - outlier_limit) / max(outlier_limit, 0.01)) * 45.0)
        direction = "höher" if current_delta > 0 else "tiefer"
        measurement_warning = (
            f"Die heutige HRV/RMSSD-Messung liegt ungewöhnlich stark {direction} als dein individueller Verlauf. "
            "Das kann echte Belastung/Erholung anzeigen, aber auch ein Messfehler sein. Bitte Messung möglichst "
            "standardisiert wiederholen oder spätestens morgen erneut messen."
        )

    # HRV-Stabilität / CV: Plews/Esco-Logik, aber pragmatisch mit LnRMSSD-SD/CV.
    rolling_values = temp_df["LnRMSSD"].tail(rolling_window)
    ln_sd_7 = rolling_values.std(ddof=1)
    ln_cv_7 = None
    cv_badness = 0.0
    if not pd.isna(ln_sd_7) and ln_rolling != 0:
        ln_cv_7 = abs(ln_sd_7 / ln_rolling) * 100
        # Pragmatik: nicht absolute medizinische Grenzwerte, sondern Stabilitätsmarker.
        if ln_cv_7 < 2.0:
            cv_badness = 0.0
        elif ln_cv_7 < 4.0:
            cv_badness = 20.0
        elif ln_cv_7 < 6.0:
            cv_badness = 45.0
        else:
            cv_badness = 70.0

    # Plews: R-R/LnRMSSD-Ratio zur Einordnung möglicher HRV-Saturation.
    rr_current = rr_interval_ms(current_resting_hr)
    ratio_current = ln_rmssd(current_rmssd) / rr_current
    ratio_rolling = None
    saturation_score = 0.0
    freshness_score = 0.0

    if "RR_Intervall_ms" in temp_df.columns:
        temp_df["LnRMSSD_RR_Ratio"] = temp_df["LnRMSSD"] / temp_df["RR_Intervall_ms"]
        ratio_rolling = temp_df["LnRMSSD_RR_Ratio"].tail(rolling_window).mean()
        rr_baseline = temp_df["RR_Intervall_ms"].tail(baseline_window).mean()
        rr_delta = rr_current - rr_baseline

        # Saturation: sehr tiefer Puls/lange RR-Intervalle, LnRMSSD leicht tiefer, Ratio nicht erhöht.
        if rr_delta > 60 and ln_rolling < baseline_mean and ratio_rolling <= temp_df["LnRMSSD_RR_Ratio"].tail(baseline_window).mean():
            saturation_score = min(85.0, 35.0 + (rr_delta / 120.0) * 30.0)
            freshness_score = min(70.0, 20.0 + saturation_score * 0.5)

    if ln_rolling < lower:
        status = "unter Baseline"
        badness = min(85, 35 + ((lower - ln_rolling) / swc) * 35)
        explanation = "Der 7-Messpunkt-LnRMSSD-Trend liegt unter der individuellen Veränderungsschwelle. Kontext prüfen: Bei tiefem Ruhepuls kann auch HRV-Saturation vorliegen."
        if saturation_score >= 40:
            status = "unter Baseline - mögliche Saturation"
            badness = max(5.0, badness * 0.45)
            explanation = "LnRMSSD ist tiefer, aber der Ruhepuls/R-R-Kontext spricht für mögliche HRV-Saturation. Nicht automatisch als schlechte Erholung interpretieren."
    elif ln_rolling > upper:
        status = "über Baseline - Kontext prüfen"
        badness = 10
        explanation = "Der 7-Messpunkt-LnRMSSD-Trend liegt über der individuellen Veränderungsschwelle. Das ist nicht automatisch positiv und sollte mit Load, Ruhepuls und Müdigkeit geprüft werden."
    else:
        status = "stabil"
        badness = 0
        explanation = "Der 7-Messpunkt-LnRMSSD-Trend liegt innerhalb der individuellen Veränderungsschwelle."

    # Wochenmittel: letzte 7 vs vorherige 7 Messungen, wenn vorhanden.
    weekly_mean_current = None
    weekly_mean_previous = None
    weekly_mean_delta = None
    if n >= 14:
        weekly_mean_current = temp_df["LnRMSSD"].tail(7).mean()
        weekly_mean_previous = temp_df["LnRMSSD"].iloc[-14:-7].mean()
        weekly_mean_delta = weekly_mean_current - weekly_mean_previous

    result.update({
        "ln_rmssd_rolling": ln_rolling,
        "ln_rmssd_baseline": baseline_mean,
        "swc": swc,
        "trend_delta": trend_delta,
        "status": status,
        "badness": float(badness),
        "weekly_mean_current": weekly_mean_current,
        "weekly_mean_previous": weekly_mean_previous,
        "weekly_mean_delta": weekly_mean_delta,
        "lnrmssd_sd_7": None if pd.isna(ln_sd_7) else float(ln_sd_7),
        "lnrmssd_cv_7": None if ln_cv_7 is None else float(ln_cv_7),
        "cv_badness": float(cv_badness),
        "lnrmssd_rr_ratio_rolling": ratio_rolling,
        "saturation_score": float(saturation_score),
        "freshness_score": float(freshness_score),
        "hrv_outlier_score": float(hrv_outlier_score),
        "measurement_warning": measurement_warning,
        "explanation": explanation,
    })
    return result


# =============================
# DATA STRUCTURE
# =============================

@dataclass
class UserInputV4:
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
    hrv_cv_badness: float = 0.0
    hrv_saturation_score: float = 0.0
    hrv_freshness_score: float = 0.0
    hrv_outlier_score: float = 0.0
    hrv_measurement_warning: str = ""
    hrr_1min: Optional[float] = None
    hrr_badness_value: float = 0.0


# =============================
# MODEL
# =============================

class FatigueProfilerV4:
    def __init__(self, data: UserInputV4):
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
            combined = (
                single_day_badness * 0.60 +
                self.d.hrv_trend_badness * 0.25 +
                self.d.hrv_cv_badness * 0.15
            )
            # Wenn der aktuelle Wert ein starker Ausreisser ist, wird ein Messfehler-Hinweis
            # ausgegeben. Der Wert wird nicht komplett ignoriert, aber weniger hart bestraft,
            # solange nicht weitere klare Warnzeichen vorliegen.
            if self.d.hrv_outlier_score >= 70:
                combined = min(combined, 55 + single_day_badness * 0.20)
            # Plews-Logik: bei möglicher HRV-Saturation tiefe HRV nicht zu hart bestrafen.
            if self.d.hrv_saturation_score >= 40:
                combined *= 0.55
            return self.clamp(combined, 0, 100)
        return single_day_badness

    def hrv_instability_badness(self):
        return self.clamp(self.d.hrv_cv_badness, 0, 100)

    def hrv_saturation_score(self):
        return self.clamp(self.d.hrv_saturation_score, 0, 100)

    def hrv_freshness_score(self):
        return self.clamp(self.d.hrv_freshness_score, 0, 100)

    def hrr_badness(self):
        return self.clamp(self.d.hrr_badness_value, 0, 100)

    def hr_badness(self):
        return self.clamp(self.hr_delta() * 10, 0, 100)

    def load_badness(self):
        """Progressive ACWR-Bewertung statt harter Deckelung ab 1.5.

        Ziel: Auch bei sehr hohen ACWR-Werten soll die App noch auf
        Veränderungen der Session-RPE-Intensität reagieren. Früher wurde
        bereits ab ACWR >= 1.5 immer 100 vergeben; dadurch blieben
        Szenarien mit RPE 9, 7 oder 4 praktisch gleich bewertet.
        """
        ratio = self.acwr()
        if ratio <= 1.0:
            return 0
        if ratio <= 1.3:
            return self.clamp((ratio - 1.0) / 0.3 * 40, 0, 40)
        if ratio <= 1.8:
            return self.clamp(40 + (ratio - 1.3) / 0.5 * 35, 40, 75)
        if ratio <= 2.5:
            return self.clamp(75 + (ratio - 1.8) / 0.7 * 20, 75, 95)
        # Ab hier bleibt die Belastung sehr hoch, aber nicht vollständig
        # gesättigt. So bleibt noch etwas Differenzierung möglich.
        return self.clamp(95 + min((ratio - 2.5) * 1.0, 5), 95, 100)

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

    def compensation_badness(self):
        """Erkennt mögliche parasympathische Kompensation.

        Hintergrund:
        Sehr hohe RMSSD + tiefer Ruhepuls sind oft gut. In Kombination mit
        hohem Load, Müdigkeit, Muskelschmerzen oder schlechtem Schlaf können sie
        aber auch eine kompensatorische parasympathische Dominanz anzeigen.
        """
        hrv_ratio = self.hrv_ratio()
        hr_delta = self.hr_delta()
        load_bad = self.load_badness()
        fatigue_bad = self.bad_from_1_to_10(self.d.general_fatigue)
        soreness_bad = self.bad_from_1_to_10(self.d.muscle_soreness)
        sleep_bad = 100 - self.good_from_1_to_10(self.d.sleep_quality)
        mood_bad = 100 - self.good_from_1_to_10(self.d.mood)

        strong_high_hrv = hrv_ratio >= 1.30
        clearly_low_hr = hr_delta <= -4
        relevant_context = (
            self.acwr() >= 1.30 or
            self.d.general_fatigue >= 7 or
            self.d.muscle_soreness >= 7 or
            self.d.sleep_quality <= 4
        )

        if not (strong_high_hrv and clearly_low_hr and relevant_context):
            return 0

        raw = (
            25 +
            load_bad * 0.30 +
            fatigue_bad * 0.25 +
            soreness_bad * 0.20 +
            sleep_bad * 0.15 +
            mood_bad * 0.10
        )
        return self.clamp(raw, 0, 95)

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
        hrv_cv_bad = self.hrv_instability_badness()
        hrr_bad = self.hrr_badness()

        compensation = self.compensation_badness()

        # V4: mentale Belastung und Schlaf wirken etwas stärker auf zentrale Erschöpfung.
        central = (
            hrv_bad * 0.21 +
            hr_bad * 0.16 +
            stress_bad * 0.24 +
            fatigue_bad * 0.20 +
            sleep_bad * 0.13 +
            resp_bad * 0.04 +
            hrv_trend_bad * 0.02 +
            hrv_cv_bad * 0.04 +
            hrr_bad * 0.03
        )

        muscular = (
            soreness_bad * 0.55 +
            load_bad * 0.25 +
            fatigue_bad * 0.20
        )

        # V4: Krankheitssymptome bleiben sehr dominant.
        illness = (
            illness_bad * 0.70 +
            hr_bad * 0.12 +
            hrv_bad * 0.06 +
            sleep_bad * 0.08 +
            mood_bad * 0.04
        )

        circulatory = (
            bp_bad * 0.55 +
            hrr_bad * 0.45
        )

        # Kombinationen aus sehr tiefer HRR, hohem Blutdruck und erhöhter Atemfrequenz
        # werden stärker gewichtet als isolierte Einzelwerte.
        if hrr_bad >= 90 and bp_bad >= 70:
            circulatory = max(circulatory, 88)
        if hrr_bad >= 90 and bp_bad >= 70 and resp_bad >= 30:
            circulatory = max(circulatory, 95)

        # Gesamtstatus: Der Erholungsindex übernimmt die globale Einordnung.
        # Das frühere Profil "Globale Belastung" wurde bewusst entfernt,
        # damit die dominante Profilanzeige nur konkrete Ursachen zeigt.
        recovery = 100 - (
            central * 0.30 +
            muscular * 0.23 +
            illness * 0.30 +
            circulatory * 0.07 +
            load_bad * 0.08 +
            compensation * 0.03 +
            hrv_cv_bad * 0.04
        )

        # V4: harte Plausibilitäts-Caps aus den Szenariotests.
        if self.d.illness > 6:
            recovery = min(recovery, 40)
        elif 3 <= self.d.illness <= 6:
            recovery = min(recovery, 65)

        if compensation >= 50:
            recovery = min(recovery, 55)

        if stress_bad >= 85 and sleep_bad >= 70 and fatigue_bad >= 70:
            recovery = min(recovery, 60)

        if load_bad >= 90 and fatigue_bad >= 80 and sleep_bad >= 70:
            recovery = min(recovery, 45)

        if hrv_cv_bad >= 45:
            recovery = min(recovery, 70)

        if hrr_bad >= 70:
            recovery = min(recovery, 55)
        if hrr_bad >= 90 and bp_bad >= 70:
            recovery = min(recovery, 45)
        if hrr_bad >= 90 and bp_bad >= 70 and resp_bad >= 30:
            recovery = min(recovery, 35)

        # Freshness/Saturation: bei tieferem LnRMSSD, aber sehr tiefem Ruhepuls und stabiler subjektiver Lage
        # wird Recovery nicht automatisch abgestraft.
        if self.hrv_freshness_score() >= 45 and fatigue_bad < 55 and illness_bad < 30:
            recovery = max(recovery, 70)

        return {
            "Erholungsindex": self.clamp(recovery, 0, 95),
            "Zentrale Erschöpfung": self.clamp(central, 0, 100),
            "Muskuläre Ermüdung": self.clamp(muscular, 0, 100),
            "Kreislaufregulation": self.clamp(circulatory, 0, 100),
            "Krankheitssymptome": self.clamp(illness, 0, 100),
            "Kompensationsbelastung": self.clamp(compensation, 0, 100),
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
        if self.d.hrv_measurement_warning:
            drivers.append("HRV-Messung ungewöhnlich: Messung wiederholen oder morgen erneut standardisiert messen")
        if self.d.hrv_cv_badness >= 45:
            drivers.append("HRV-Stabilität auffällig: stärkere Tag-zu-Tag-Schwankung im LnRMSSD")
        if self.d.hrv_saturation_score >= 40:
            drivers.append("Mögliche HRV-Saturation: tiefer Ruhepuls verändert die HRV-Interpretation")
        if self.d.hrv_freshness_score >= 45:
            drivers.append("Mögliches Freshness-Muster: tiefer Puls und HRV-Kontext sprechen nicht zwingend für Ermüdung")
        if self.hrr_badness() >= 45:
            drivers.append("Heart Rate Recovery langsam oder auffällig")
        if self.compensation_badness() >= 50:
            drivers.append("Mögliche parasympathische Kompensation: hohe RMSSD, tiefer Ruhepuls und Belastungszeichen")
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
        score = (
            self.hrv_score() * 0.22 +
            self.hr_score() * 0.12 +
            self.load_score() * 0.17 +
            self.subjective_score() * 0.36 +
            (100 - self.respiratory_badness()) * 0.07 +
            (100 - self.d.hrv_trend_badness) * 0.035 +
            (100 - self.d.hrv_cv_badness) * 0.035 +
            (100 - self.compensation_badness()) * 0.02 +
            (100 - self.hrr_badness()) * 0.01 +
            self.hrv_freshness_score() * 0.015
        )

        # V4 Safety Override: Krankheitssymptome werden härter begrenzt.
        if self.d.illness > 6:
            score = min(score, 15)
        elif 3 <= self.d.illness <= 6:
            score = min(score, 55)

        if self.d.hrv_trend_status == "unter Baseline":
            score = min(score, 65)

        if self.compensation_badness() >= 50:
            score = min(score, 60)

        if self.d.hrv_cv_badness >= 60:
            # HRV-Instabilität bleibt ein Warnsignal, soll aber andere
            # Eingaben wie Session-RPE nicht vollständig überdecken.
            score = min(score, 70)

        if self.hrr_badness() >= 70:
            score = min(score, 55)
        if self.hrr_badness() >= 90 and self.bp_badness() >= 70:
            score = min(score, 45)
        if self.hrr_badness() >= 90 and self.bp_badness() >= 70 and self.respiratory_badness() >= 30:
            score = min(score, 40)

        if self.load_badness() >= 90 and self.d.general_fatigue >= 8 and self.d.sleep_quality <= 3:
            score = min(score, 35)

        return self.clamp(score, 0, 95)

    def work_readiness(self):
        fatigue_good = 100 - self.bad_from_1_to_10(self.d.general_fatigue)
        stress_good = 100 - self.bad_from_1_to_10(self.d.mental_stress)
        illness_good = 100 - self.bad_from_1_to_10(self.d.illness)
        sleep_good = self.good_from_1_to_10(self.d.sleep_quality)
        mood_good = self.good_from_1_to_10(self.d.mood)
        score = fatigue_good * 0.20 + stress_good * 0.26 + illness_good * 0.26 + sleep_good * 0.18 + mood_good * 0.10
        if self.d.illness > 6:
            score = min(score, 30)
        elif 3 <= self.d.illness <= 6:
            score = min(score, 65)
        return self.clamp(score, 0, 95)

    def recommendation(self, profile):
        """Konkrete, profil- und situationsabhängige Trainingsempfehlung.

        Ziel: Nicht nur "kontrolliert trainieren", sondern eine konkrete
        Handlungsempfehlung: Pause, lockere Grundlagenausdauer, Technik,
        kurze Steigerungen oder intensive Einheit.
        """
        readiness = self.training_readiness()
        fatigue_bad = self.bad_from_1_to_10(self.d.general_fatigue)
        stress_bad = self.bad_from_1_to_10(self.d.mental_stress)
        soreness_bad = self.bad_from_1_to_10(self.d.muscle_soreness)
        sleep_bad = 100 - self.good_from_1_to_10(self.d.sleep_quality)
        hrr_bad = self.hrr_badness()
        bp_bad = self.bp_badness()
        resp_bad = self.respiratory_badness()
        load_bad = self.load_badness()

        # Sicherheitsregeln zuerst.
        if self.d.illness > 6:
            return (
                "Heute kein Training empfohlen. Die Krankheitssymptome sind deutlich erhöht. "
                "Priorität haben Schlaf, Flüssigkeit und Erholung. Erst wieder trainieren, wenn die Symptome klar gesunken sind."
            )
        if 3 <= self.d.illness <= 6:
            return (
                "Heute höchstens sehr lockere Bewegung: Spaziergang, Mobility oder 20-30 Minuten sehr lockere Grundlagenausdauer. "
                "Keine Intervalle, keine harten Kraftbelastungen und keine Wettkampfintensität."
            )

        if hrr_bad >= 90 and bp_bad >= 70:
            return (
                "Die Kreislaufwerte sind heute deutlich auffällig: sehr tiefe HRR kombiniert mit erhöhtem Blutdruck. "
                "Heute keine intensive Belastung. Wenn du dich bewegen möchtest: 15-30 Minuten sehr locker in Zone 1, Mobility oder Spaziergang. "
                "HRR-Test und Blutdruck unter standardisierten Bedingungen wiederholen; bei Beschwerden medizinisch abklären."
            )
        if hrr_bad >= 90:
            return (
                "Die Heart Rate Recovery ist sehr tief. Das kann ein Test-/Messfehler sein, aber auch auf unvollständige Erholung hinweisen. "
                "Heute keine harten Intervalle und keine maximale Kraft. Wenn Training: 20-40 Minuten lockere Grundlagenausdauer in Zone 1-2. "
                "Den HRR-Test bitte unter gleichen Bedingungen wiederholen."
            )

        if self.d.hrv_measurement_warning:
            return (
                "Die heutige HRV-Messung ist ungewöhnlich und könnte auch ein Messfehler sein. "
                "Bitte Messung möglichst standardisiert wiederholen oder morgen erneut messen. "
                "Bis zur Bestätigung: keine maximale Einheit; geeignet sind lockere Grundlagenausdauer, Techniktraining oder Mobility."
            )

        if self.d.hrv_trend_status == "unter Baseline - mögliche Saturation":
            return (
                "LnRMSSD ist tiefer, aber der tiefe Ruhepuls spricht für mögliche HRV-Saturation. "
                "Nicht automatisch als schlechte Erholung werten. Wenn du dich gut fühlst: moderate Einheit möglich; "
                "keine unnötige Maximalbelastung."
            )
        if self.d.hrv_trend_status == "unter Baseline":
            return (
                "Der LnRMSSD-Trend liegt unter deiner individuellen Schwelle. Heute eher aktivierend statt ermüdend trainieren: "
                "20-45 Minuten Grundlagenausdauer in Zone 1-2, Techniktraining oder Mobility. Keine langen intensiven Intervalle."
            )
        if self.d.hrv_cv_badness >= 60:
            return (
                "Deine HRV schwankt stärker als üblich. Das kann Belastung, Erholung oder einen Messfehler widerspiegeln. "
                "Heute geeignet: lockere Grundlagenausdauer, Techniktraining, Bewegungsqualität oder Mobility. "
                "Wenn du dich sehr gut fühlst, sind kurze Steigerungen möglich; keine erschöpfende Einheit."
            )
        if self.compensation_badness() >= 50:
            return (
                "Mögliche Kompensationsbelastung: hohe RMSSD und tiefer Ruhepuls wirken gut, passen aber nicht zu Load/Müdigkeit/Schlaf. "
                "Heute keine Hero-Session. Besser: lockere bis moderate Grundlagenausdauer, Technik oder kurze Aktivierung. "
                "Intensität nur erhöhen, wenn dies bewusst Teil eines geplanten Belastungsblocks ist."
            )
        if self.d.hrv_trend_status == "über Baseline - Kontext prüfen" and (self.d.general_fatigue >= 7 or self.d.acute_load / self.d.chronic_load > 1.3):
            return (
                "Der LnRMSSD-Trend ist erhöht, gleichzeitig gibt es Belastungszeichen. Hohe HRV nicht automatisch als top erholt interpretieren. "
                "Heute: Grundlagenausdauer Zone 1-2, Techniktraining oder eine moderate Einheit ohne Erschöpfung."
            )

        # Profil-spezifische Empfehlungen.
        if profile == "Zentrale Erschöpfung":
            if readiness < 45:
                return (
                    "Dein Nervensystem wirkt deutlich belastet. Heute keine HIIT-Einheit, keine Wettkampfintensität und keine Maximalkraft. "
                    "Empfehlung: 20-40 Minuten lockere Grundlagenausdauer in Zone 1-2, Mobility oder Techniktraining mit niedriger bis moderater Intensität."
                )
            return (
                "Moderate zentrale Ermüdung. Ziel: Organismus aktivieren, ohne zusätzlich stark zu ermüden. "
                "Geeignet sind Grundlagenausdauer, Techniktraining, Koordination oder kurze Steigerungen, falls du dich subjektiv gut fühlst. "
                "Keine langen harten Intervalle."
            )

        if profile == "Muskuläre Ermüdung":
            if soreness_bad >= 70 or readiness < 45:
                return (
                    "Die Muskulatur ist deutlich belastet. Heute keine schwere Kraft, keine hohen Volumenblöcke und keine harten Berg-/Sprintbelastungen. "
                    "Empfehlung: lockere Grundlagenausdauer, Mobility, Technik oder alternative Muskelgruppen."
                )
            return (
                "Muskuläre Ermüdung ist vorhanden, aber nicht zwingend ein Pausengrund. "
                "Geeignet sind lockere Grundlagenausdauer, Techniktraining und Bewegungsqualität. "
                "Wenn du dich gut fühlst: kurze Steigerungen oder kurze Schnelligkeitsreize sind möglich, aber keine erschöpfende Einheit."
            )

        if profile == "Kreislaufregulation":
            return (
                "Herz-Kreislauf-Werte sind auffällig. Heute keine maximale Belastung, keine langen Intervalle und kein Wettkampftraining. "
                "Wenn Training: 15-40 Minuten sehr locker bis locker in Zone 1-2. Blutdruck/HRR bei auffälligen Werten erneut standardisiert prüfen."
            )

        if profile == "Krankheitssymptome":
            return (
                "Krankheitssymptome beachten. Wenn überhaupt, nur sehr lockere Bewegung ohne Atemnot oder Druckgefühl: Spaziergang, Mobility oder kurze lockere Zone-1-Einheit. "
                "Keine Intensität, solange Symptome bestehen."
            )

        if profile == "Kompensationsbelastung":
            return (
                "Hohe HRV bedeutet heute möglicherweise nicht automatisch gute Erholung. "
                "Besser aktivierend statt ermüdend: Grundlagenausdauer Zone 1-2, Techniktraining oder kurze Aktivierung. "
                "Keine maximale Einheit, ausser ein geplanter Intensitäts-/Belastungsblock wird bewusst durchgeführt und der Verlauf wird eng beobachtet."
            )

        if profile == "Erholungsindex":
            if readiness >= 75:
                return (
                    "Gute Voraussetzungen für Training. Je nach Trainingsplan sind intensive Intervalle, Krafttraining oder eine längere Einheit möglich. "
                    "Trotzdem primäres/sekundäres Profil und Körpergefühl beachten."
                )
            if readiness >= 55:
                return (
                    "Solide, aber nicht perfekte Readiness. Geeignet sind Grundlagenausdauer, Techniktraining oder moderate Kraft. "
                    "Wenn du dich gut fühlst, kannst du kurze Steigerungen einbauen. Vermeide unnötig lange erschöpfende Einheiten."
                )
            return (
                "Readiness ist reduziert. Heute eher aktivierend trainieren: lockere Grundlagenausdauer, Mobility oder Technik. "
                "Keine maximale Intensität."
            )

        # Zusätzlicher Kontext für hohe Last, falls kein anderes Profil dominiert.
        if load_bad >= 80 and fatigue_bad < 55 and sleep_bad < 55:
            return (
                "Der akute Load ist hoch, aber die subjektiven Signale sind nicht stark auffällig. "
                "Wenn dies ein geplanter Belastungsblock ist, kann eine höhere Belastung bewusst akzeptiert werden. "
                "Verlauf, Schlaf, Symptome und HRV in den nächsten Tagen eng beobachten."
            )

        return (
            "Moderate Einheit möglich. Am sinnvollsten: Grundlagenausdauer, Techniktraining, Koordination oder moderate Kraft. "
            "Kurze Steigerungen sind möglich, wenn du dich gut fühlst; keine Einheit bis zur Erschöpfung."
        )

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
                "HRV Instabilität": round(self.d.hrv_cv_badness, 1),
                "HRV Saturation": round(self.hrv_saturation_score(), 1),
                "Freshness Score": round(self.hrv_freshness_score(), 1),
                "HRV Ausreisser / Messwarnung": round(self.d.hrv_outlier_score, 1),
                "Heart Rate Recovery Badness": round(self.hrr_badness(), 1),
                "Kompensationsbelastung": round(self.compensation_badness(), 1),
            }
        }


# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="Readiness-App V9", page_icon="🏃", layout="wide")
st.title("🏃 Readiness-App V9")
st.caption("Training Readiness, Work Readiness, Fatigue-Profile, HRV-Trends, Session-RPE-Load und Google-Sheet-Speicherung")

if google_sheets_configured():
    st.info("Datenspeicherung: Google Sheet ist konfiguriert.")
else:
    st.info("Datenspeicherung: lokale CSV. Für Streamlit Cloud bitte Google-Sheet-Secrets konfigurieren.")

st.sidebar.header("Eingabe")
st.sidebar.subheader("User")
user_id = st.sidebar.text_input("User-ID", value="test_user_01", help="Bitte eindeutige ID verwenden, z.B. athlete_01 oder beni_01.")
st.sidebar.caption("Jede gespeicherte Messung wird mit dieser User-ID abgelegt.")

# =============================
# NUTZERLEITFADEN PDF
# =============================

st.sidebar.divider()
st.sidebar.subheader("📘 Nutzerleitfaden")

try:
    with open("Readyness_App_Nutzerleitfaden.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.sidebar.download_button(
        label="📥 Nutzerleitfaden öffnen / herunterladen",
        data=PDFbyte,
        file_name="Readyness_App_Nutzerleitfaden.pdf",
        mime="application/pdf"
    )

except FileNotFoundError:
    st.sidebar.warning(
        "Nutzerleitfaden nicht gefunden. "
        "Bitte prüfen ob die PDF-Datei in GitHub hochgeladen wurde."
    )
with st.sidebar.expander("Trainingsbelastung - Session RPE", expanded=True):
    st.caption("Akuter Load = Dauer in Minuten × Intensität 1-10 × Trainingsart-Faktor.")

    training_type = st.selectbox(
        "Trainingsart",
        ["Ausdauer", "Kraft", "HIIT", "Wettkampf", "Recovery Training"],
    )

    duration_min = st.number_input(
        "Dauer der Einheit (Minuten)",
        min_value=0,
        value=60,
        step=5,
    )

    intensity_rpe = st.slider(
        "Intensität / Session RPE",
        min_value=1,
        max_value=10,
        value=5,
        help="1 = sehr leicht, 10 = maximal anstrengend",
    )

    if training_type == "Kraft":
        strength_type = st.selectbox(
            "Krafttraining-Typ",
            ["Hypertrophie", "Maximalkraft", "Kraftausdauer"],
        )
    else:
        strength_type = "keine Kraftangabe"

    acute_load = calculate_session_rpe_load(
        duration_min=duration_min,
        intensity_rpe=intensity_rpe,
        training_type=training_type,
        strength_type=strength_type,
    )

    st.metric("Berechneter akuter Trainingload", acute_load)

with st.sidebar.expander("Chronische Trainingload über die letzten 30 Tage", expanded=True):
    st.caption("Für die Testversion wird der chronische Load aus deinem typischen Training der letzten 30 Tage geschätzt.")

    weekly_training_hours = st.number_input(
        "Trainingsstunden pro Woche im Mittel",
        min_value=0.0,
        value=5.0,
        step=0.5,
    )

    intensive_sessions_per_week = st.number_input(
        "Intensive Einheiten pro Woche (HIIT/Wettkampf)",
        min_value=0,
        value=1,
        step=1,
    )

    strength_sessions_per_week = st.number_input(
        "Krafttrainings pro Woche",
        min_value=0,
        value=1,
        step=1,
    )

    chronic_strength_type = st.selectbox(
        "Dominanter Krafttraining-Typ",
        ["Hypertrophie", "Maximalkraft", "Kraftausdauer", "keine Kraftangabe"],
    )

    chronic_estimate = calculate_chronic_training_load(
        weekly_hours=weekly_training_hours,
        intensive_sessions_per_week=intensive_sessions_per_week,
        strength_sessions_per_week=strength_sessions_per_week,
        dominant_strength_type=chronic_strength_type,
    )

    chronic_load = chronic_estimate["daily_equivalent_load"]
    st.metric("Berechneter chronischer Tages-Load", chronic_load)
    st.caption(
        f"Geschätzter Wochenload: {chronic_estimate['weekly_load']} | "
        f"geschätzte Durchschnittsintensität: {chronic_estimate['estimated_average_rpe']}"
    )

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

with st.sidebar.expander("Optional: Heart Rate Recovery - 3-Minuten-Test", expanded=False):
    st.caption(
        "Standardvorschlag: 3 Minuten kontrolliert harte Belastung (ca. 85-90 % HFmax oder RPE 8/10), "
        "danach sofort absitzen, nicht sprechen und die Herzfrequenz nach 60 Sekunden erfassen. "
        "HRR60 = Peak-HF minus HF nach 60 Sekunden."
    )
    st.info(
        "HRR bitte nur vergleichen, wenn Belastungsart, Dauer, Intensität, Tageszeit und Erholungssituation "
        "möglichst gleich bleiben. Schneller ist nicht immer besser; die App interpretiert HRR im Kontext."
    )
    use_hrr = st.checkbox("Standardisierten HRR-Test einbeziehen")
    if use_hrr:
        hrr_test_type = st.selectbox(
            "Testart",
            ["3-Minuten-Stufentest / kontrolliert hart", "anderer standardisierter Test"],
        )
        hr_peak_exercise = st.number_input("Peak-Herzfrequenz am Ende des Tests", min_value=1, value=170, step=1)
        hr_1min_recovery = st.number_input("Herzfrequenz nach 1 Minute sitzender Erholung", min_value=1, value=140, step=1)
        hrr_1min = max(0, hr_peak_exercise - hr_1min_recovery)
        hrr_badness_value = compute_hrr_badness(hrr_1min)
        st.metric("HRR60", hrr_1min)
        st.caption(hrr_interpretation_text(hrr_1min))
    else:
        hrr_test_type = "kein HRR-Test"
        hr_peak_exercise = None
        hr_1min_recovery = None
        hrr_1min = None
        hrr_badness_value = 0.0

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
hrv_trend = compute_hrv_trend(history_df, rmssd, resting_hr, user_id.strip())

data = UserInputV4(
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
    hrv_cv_badness=hrv_trend["cv_badness"],
    hrv_saturation_score=hrv_trend["saturation_score"],
    hrv_freshness_score=hrv_trend["freshness_score"],
    hrv_outlier_score=hrv_trend["hrv_outlier_score"],
    hrv_measurement_warning=hrv_trend["measurement_warning"],
    hrr_1min=hrr_1min,
    hrr_badness_value=hrr_badness_value,
)

profiler = FatigueProfilerV4(data)
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
            "Trainingsart": training_type,
            "Dauer Minuten": duration_min,
            "Session RPE": intensity_rpe,
            "Krafttraining Typ": strength_type,
            "Akuter Load": acute_load,
            "Chronischer Load": chronic_load,
            "ACWR": round(acute_load / chronic_load, 2),
            "30 Tage Wochenstunden": weekly_training_hours,
            "30 Tage intensive Einheiten/Woche": intensive_sessions_per_week,
            "30 Tage Krafttrainings/Woche": strength_sessions_per_week,
            "30 Tage Krafttyp": chronic_strength_type,
            "Geschätzter Wochenload": chronic_estimate["weekly_load"],
            "RMSSD": rmssd,
            "LnRMSSD": round(ln_rmssd(rmssd), 4),
            "Baseline RMSSD": baseline_rmssd,
            "Ruhepuls": resting_hr,
            "Baseline Ruhepuls": baseline_resting_hr,
            "Atemfrequenz": respiratory_rate,
            "HRR Testart": hrr_test_type,
            "HRR 1 Minute": hrr_1min,
            "HRR Badness": hrr_badness_value,
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
            "Kompensationsbelastung": result["subscores"].get("Kompensationsbelastung", 0),
            "LnRMSSD Rolling": None if hrv_trend["ln_rmssd_rolling"] is None else round(hrv_trend["ln_rmssd_rolling"], 4),
            "LnRMSSD Baseline": None if hrv_trend["ln_rmssd_baseline"] is None else round(hrv_trend["ln_rmssd_baseline"], 4),
            "SWC": None if hrv_trend["swc"] is None else round(hrv_trend["swc"], 4),
            "LnRMSSD CV 7": None if hrv_trend["lnrmssd_cv_7"] is None else round(hrv_trend["lnrmssd_cv_7"], 3),
            "HRV CV Badness": round(hrv_trend["cv_badness"], 1),
            "RR Intervall ms": round(hrv_trend["rr_interval_current"], 1),
            "LnRMSSD RR Ratio": round(hrv_trend["lnrmssd_rr_ratio_current"], 6),
            "HRV Saturation Score": round(hrv_trend["saturation_score"], 1),
            "Freshness Score": round(hrv_trend["freshness_score"], 1),
            "HRV Outlier Score": round(hrv_trend["hrv_outlier_score"], 1),
            "HRV Messwarnung": hrv_trend["measurement_warning"],
            "Wochenmittel LnRMSSD aktuell": None if hrv_trend["weekly_mean_current"] is None else round(hrv_trend["weekly_mean_current"], 4),
            "Wochenmittel LnRMSSD vorher": None if hrv_trend["weekly_mean_previous"] is None else round(hrv_trend["weekly_mean_previous"], 4),
            "Wochenmittel Delta": None if hrv_trend["weekly_mean_delta"] is None else round(hrv_trend["weekly_mean_delta"], 4),
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

trend_col5, trend_col6, trend_col7, trend_col8 = st.columns(4)
trend_col5.metric("LnRMSSD CV 7", "-" if hrv_trend["lnrmssd_cv_7"] is None else f"{round(hrv_trend['lnrmssd_cv_7'], 2)} %")
trend_col6.metric("HRV Instabilität", round(hrv_trend["cv_badness"], 1))
trend_col7.metric("HRV Saturation", round(hrv_trend["saturation_score"], 1))
trend_col8.metric("Freshness Score", round(hrv_trend["freshness_score"], 1))

if hrv_trend["weekly_mean_current"] is not None:
    st.caption(
        f"Wochenmittel aktuell: {round(hrv_trend['weekly_mean_current'], 3)} | "
        f"Vorwoche: {round(hrv_trend['weekly_mean_previous'], 3)} | "
        f"Delta: {round(hrv_trend['weekly_mean_delta'], 3)}"
    )

compensation_value = result["subscores"].get("Kompensationsbelastung", 0)
if compensation_value >= 50:
    st.warning(
        "Mögliche parasympathische Kompensation erkannt: hohe RMSSD und tiefer Ruhepuls "
        "treten zusammen mit Load/Müdigkeit/Muskelschmerzen oder schlechtem Schlaf auf."
    )

if hrv_trend["measurement_warning"]:
    st.warning(hrv_trend["measurement_warning"])

if hrv_trend["cv_badness"] >= 45:
    st.warning("HRV-Stabilität auffällig: Die LnRMSSD-Werte schwanken über die letzten Messungen stärker als üblich. Das kann echte Belastung oder auch Messvariabilität bedeuten.")

if hrv_trend["saturation_score"] >= 40:
    st.info("Mögliche HRV-Saturation nach Plews: Tiefer Ruhepuls/R-R-Kontext verändert die Interpretation einer tieferen HRV.")

if hrr_badness_value >= 45:
    st.warning(
        "Heart Rate Recovery verlangsamt oder auffällig. Bitte nur interpretieren, wenn der 3-Minuten-Test "
        "standardisiert durchgeführt wurde. Bei sehr tiefer HRR (z.B. <=12 bpm) Test wiederholen; zusammen mit "
        "hohem Blutdruck oder erhöhter Atemfrequenz heute keine intensive Belastung."
    )

st.divider()
st.subheader("Readiness-Profile")
profile_df = pd.DataFrame(list(result["profile_scores"].items()), columns=["Profil", "Score"])
st.dataframe(profile_df, use_container_width=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(profile_df["Profil"], profile_df["Score"])
ax.set_ylim(0, 100)
ax.set_ylabel("Score")
ax.set_title("Readiness-Profile")
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

    chart_cols = ["Training Readiness", "Work Readiness", "RMSSD", "Ruhepuls", "Krankheitssymptome", "Kompensationsbelastung", "HRV CV Badness", "HRV Outlier Score", "HRV Saturation Score", "Freshness Score", "Schlafqualität", "Mentaler Stress"]
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
    "Die Readiness-App interpretiert HRV trendbasiert über LnRMSSD, berücksichtigt HRV-Stabilität/CV, "
    "mögliche HRV-Saturation nach Plews, optionale Heart Rate Recovery als Kontextmarker, HRV-Ausreisser/Messwarnungen "
    "und konkrete Trainingsempfehlungen für Grundlagenausdauer, Technik, Aktivierung oder Erholung."
)
