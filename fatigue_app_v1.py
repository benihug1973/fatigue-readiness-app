import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


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

    supine_hr: Optional[float] = None
    standing_hr: Optional[float] = None

    hr_peak_exercise: Optional[float] = None
    hr_1min_recovery: Optional[float] = None

    general_fatigue: int = 1
    muscle_soreness: int = 1
    mental_stress: int = 1
    illness: int = 1
    sleep_quality: int = 10
    mood: int = 10


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
        return self.clamp((1 - self.hrv_ratio()) * 160, 0, 100)

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
        if None in [
            self.d.systolic_bp,
            self.d.diastolic_bp,
            self.d.baseline_systolic_bp,
            self.d.baseline_diastolic_bp,
        ]:
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

        return (
            fatigue_good * 0.22 +
            soreness_good * 0.13 +
            stress_good * 0.18 +
            illness_good * 0.20 +
            sleep_good * 0.17 +
            mood_good * 0.10
        )

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

        central = (
            hrv_bad * 0.25 +
            hr_bad * 0.18 +
            stress_bad * 0.22 +
            fatigue_bad * 0.18 +
            sleep_bad * 0.10 +
            resp_bad * 0.07
        )

        muscular = (
            soreness_bad * 0.45 +
            load_bad * 0.35 +
            fatigue_bad * 0.20
        )

        illness = (
            illness_bad * 0.50 +
            hr_bad * 0.20 +
            hrv_bad * 0.12 +
            sleep_bad * 0.10 +
            mood_bad * 0.08
        )

        circulatory = bp_bad

        global_load = (
            central * 0.35 +
            muscular * 0.25 +
            illness * 0.20 +
            circulatory * 0.20
        )

        recovery = 100 - (
            central * 0.30 +
            muscular * 0.25 +
            illness * 0.20 +
            circulatory * 0.10 +
            load_bad * 0.15
        )

        return {
            "Erholungsindex": self.clamp(recovery, 0, 100),
            "Zentrale Erschöpfung": self.clamp(central, 0, 100),
            "Muskuläre Ermüdung": self.clamp(muscular, 0, 100),
            "Globale Belastung": self.clamp(global_load, 0, 100),
            "Kreislauf / BR auffällig": self.clamp(circulatory, 0, 100),
            "Infekt-Risiko": self.clamp(illness, 0, 100),
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
            drivers.append("RMSSD deutlich unter Baseline")
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
        if self.d.illness >= 6:
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
            self.hrv_score() * 0.25 +
            self.hr_score() * 0.15 +
            self.load_score() * 0.20 +
            self.subjective_score() * 0.30 +
            (100 - self.respiratory_badness()) * 0.10
        )
        return self.clamp(score, 0, 100)

    def work_readiness(self):
        fatigue_good = 100 - self.bad_from_1_to_10(self.d.general_fatigue)
        stress_good = 100 - self.bad_from_1_to_10(self.d.mental_stress)
        illness_good = 100 - self.bad_from_1_to_10(self.d.illness)
        sleep_good = self.good_from_1_to_10(self.d.sleep_quality)
        mood_good = self.good_from_1_to_10(self.d.mood)

        score = (
            fatigue_good * 0.22 +
            stress_good * 0.25 +
            illness_good * 0.20 +
            sleep_good * 0.20 +
            mood_good * 0.13
        )

        return self.clamp(score, 0, 100)

    def recommendation(self, profile):
        if profile == "Erholungsindex":
            return "Gute bis solide Erholung. Training möglich, Intensität abhängig von Ziel und Load."
        if profile == "Zentrale Erschöpfung":
            return "Heute eher lockeres Training, Stress reduzieren, Schlaf und Regeneration priorisieren."
        if profile == "Muskuläre Ermüdung":
            return "Heute keine harte muskuläre Belastung. Lockeres Training, Mobility oder aktive Erholung."
        if profile == "Globale Belastung":
            return "Mehrere Systeme wirken belastet. Trainingsintensität deutlich reduzieren."
        if profile == "Kreislauf / BR auffällig":
            return "Kreislaufregulation auffällig. Belastung vorsichtig wählen und Verlauf beobachten."
        if profile == "Infekt-Risiko":
            return "Kein intensives Training. Symptome beobachten, Erholung und Schlaf priorisieren."

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
                "ungewöhnliche Atemfrequenz": round(self.respiratory_badness(), 1),
                "BP Badness": round(self.bp_badness(), 1),
            }
        }


# =============================
# STREAMLIT UI
# =============================

st.set_page_config(
    page_title="Fatigue App V1",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Fatigue App V1")
st.caption("Training Readiness, Work Readiness und Fatigue-Profile")

st.sidebar.header("Eingabe")

with st.sidebar.expander("Training Load", expanded=True):
    acute_load = st.number_input("Akuter Load", min_value=1.0, value=80.0)
    chronic_load = st.number_input("Chronischer Load", min_value=1.0, value=65.0)

with st.sidebar.expander("HRV & Herzfrequenz", expanded=True):
    rmssd = st.number_input("Aktuelle RMSSD", min_value=1.0, value=45.0)
    baseline_rmssd = st.number_input("Baseline RMSSD", min_value=1.0, value=50.0)
    resting_hr = st.number_input("Aktueller Ruhepuls", min_value=1.0, value=52.0)
    baseline_resting_hr = st.number_input("Baseline Ruhepuls", min_value=1.0, value=50.0)

with st.sidebar.expander("Messung & Atmung", expanded=True):
    st.caption(
        "Der Messkontext beeinflusst HRV, Herzfrequenz und Atemfrequenz. "
        "Für diese V1 werden nur Ruhe- oder Schlafmessungen verwendet."
    )

    measurement_context = st.selectbox(
        "Messkontext",
        ["rest", "sleep"]
    )

    respiratory_rate = st.number_input("Atemfrequenz", min_value=1.0, value=14.0)

with st.sidebar.expander("Optional: Blutdruck", expanded=False):
    use_bp = st.checkbox("Blutdruck einbeziehen")
    if use_bp:
        systolic_bp = st.number_input("Systolischer Blutdruck", min_value=1.0, value=120.0)
        diastolic_bp = st.number_input("Diastolischer Blutdruck", min_value=1.0, value=76.0)
        baseline_systolic_bp = st.number_input("Baseline systolisch", min_value=1.0, value=120.0)
        baseline_diastolic_bp = st.number_input("Baseline diastolisch", min_value=1.0, value=76.0)
    else:
        systolic_bp = None
        diastolic_bp = None
        baseline_systolic_bp = None
        baseline_diastolic_bp = None

with st.sidebar.expander("Subjektive Faktoren", expanded=True):
    general_fatigue = st.slider("Allgemeine Müdigkeit", 1, 10, 5)
    muscle_soreness = st.slider("Muskuläre Schmerzen", 1, 10, 5)
    mental_stress = st.slider("Mentaler Stress", 1, 10, 5)
    illness = st.slider("Krankheit / Infekt", 1, 10, 1)
    sleep_quality = st.slider("Schlafqualität", 1, 10, 7)
    mood = st.slider("Stimmung", 1, 10, 7)


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
)

profiler = FatigueProfilerV3(data)
result = profiler.run()


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

st.subheader("Fatigue Profile Scores")

profile_df = pd.DataFrame(
    list(result["profile_scores"].items()),
    columns=["Profil", "Score"]
)

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

sub_df = pd.DataFrame(
    list(result["subscores"].items()),
    columns=["Subscore", "Wert"]
)

st.dataframe(sub_df, use_container_width=True)

st.divider()

st.caption(
    "Hinweis: Dieses Modell ist ein heuristisches Monitoring-Tool und ersetzt keine medizinische Diagnostik."
)