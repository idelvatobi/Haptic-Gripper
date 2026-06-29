# Haptic Gripper 1.0

**Master-slave robotic gripper with haptic force feedback — Arduino · FSR · Servo · ML · AI · Streamlit**

Developed as part of the Human Modeling, Processing and Simulation course at the University of Deusto (Biomedical Engineering, 2025–26). Awarded **Matrícula de Honor** (Distinction).

> This is an educational prototype designed for training and simulation environments. It is not a medical device and has not undergone clinical validation.

---

## Overview

Haptic Gripper 1.0 is a master-slave robotic system that addresses one of the key limitations of modern surgical robotics: the loss of haptic feedback. Systems like the da Vinci Surgical System provide high precision but eliminate the tactile sensation that surgeons rely on when manipulating tissue.

This prototype demonstrates a functional closed-loop haptic feedback pipeline using low-cost embedded hardware: the surgeon's pinching motion on the master gripper is mirrored by the slave gripper, while a force sensor on the slave transmits contact force back to the master as physical resistance — allowing the user to feel what the robot grips.

The system extends beyond hardware, integrating a Random Forest classifier and an AI-assisted session analysis dashboard built in Streamlit.

---

## System Architecture

```
MASTER SIDE                    ARDUINO (Central Processing)           SLAVE SIDE
──────────────                 ──────────────────────────────         ──────────────
Pinching motion   ──▶  Potentiometer (A1) → map → PWM (D10)  ──▶   Slave servo closes gripper
                                                                            │
Physical resistance ◀── Feedback servo (D9) ◀── FSR signal (A0) ◀──── FSR detects grip force
                                    │
                              Serial output (9600 baud)
                                    │
                         Streamlit Dashboard (Python)
                         ├── Live FSR signal plot
                         ├── Random Forest classification
                         │   (rest / light / medium / full force)
                         └── AI-generated session report (Anthropic Claude)
```

---

## Hardware Components

| Component | Pin | Role |
|---|---|---|
| FSR Force Sensor (Seeed Studio) | A0 | Grip force measurement at slave tip |
| Potentiometer (Seeed Studio) | A1 | Master gripper angle reading |
| Feedback Servo (MG996R) | D9 | Haptic resistance on master side |
| Slave Servo (MG996R) | D10 | Slave gripper actuation |
| Arduino UNO WiFi Rev2 | USB | Central processing and serial output |
| 3D-printed parts (PLA, FDM) | — | Master and slave gripper structures |

**Total prototype cost: €64.68**

---

## Key Features

**Logarithmic force mapping** — The FSR signal is mapped to feedback servo angle using a logarithmic function, providing fine resolution at low forces and progressive resistance at higher forces. This mimics the natural response of tissue resistance more accurately than a linear mapping.

**Moving average filter** — A 5-sample rolling average reduces FSR noise and prevents erratic servo behaviour during stable grip states.

**Smooth servo control** — The feedback servo advances in configurable steps rather than jumping to the target angle, eliminating sudden mechanical jolts during force transitions.

**Random Forest classifier** — The dashboard classifies each session into four grip states (rest, light force, medium force, full force) based on FSR signal features extracted from the CSV session data.

**AI-assisted session report** — The dashboard connects to the Anthropic Claude API to generate a structured, human-readable session summary including grip pattern analysis, stability assessment, and recommendations.

---

## Repository Structure

```
HapticGripper/
├── arduino/
│   └── firmware/
│       └── haptic_gripper_firmware.ino   # Main firmware: FSR + potentiometer + dual servo
│
├── dashboard/
│   ├── app.py                            # Streamlit dashboard (live capture + ML + AI report)
│   ├── haptic_classifier.joblib          # Trained Random Forest model
│   ├── requirements.txt                  # Python dependencies
│   ├── launch.sh                         # Launch script (Linux/Mac)
│   ├── launch.bat                        # Launch script (Windows)
│   └── sessions/                         # Example session CSV files
│
├── images/                               # Prototype photos
└── docs/
    └── Final_Report.pdf                  # Full project report
```

---

## Getting Started

### Arduino

1. Open `arduino/firmware/haptic_gripper_firmware.ino` in Arduino IDE
2. Wire components according to the pin table above
3. Upload to Arduino UNO WiFi Rev2
4. Open Serial Monitor at **9600 baud** to verify CSV output:

```
timestamp_ms,fsr_raw,fsr_smoothed,feedback_target,feedback_current,pot_value,slave_angle
```

### Dashboard

```bash
# Install dependencies
pip install -r dashboard/requirements.txt

# Set your Anthropic API key (required for AI reports)
export ANTHROPIC_API_KEY=your_key_here

# Launch
cd dashboard
streamlit run app.py
```

Or use the launch scripts:
- **Linux/Mac**: `bash launch.sh`
- **Windows**: double-click `launch.bat`

---

## Force Mapping

The logarithmic mapping used for haptic feedback:

```
normalized = (FSR - FSR_MIN) / (FSR_MAX - FSR_MIN)
log_mapped  = log(1 + GAIN × normalized) / log(1 + GAIN)
angle       = FEEDBACK_MIN + log_mapped × (FEEDBACK_MAX - FEEDBACK_MIN)
```

With `GAIN = 10`, `FSR range = [0, 680]`, `feedback angle range = [30°, 150°]`.

This provides high sensitivity at low grip forces — relevant for delicate tissue manipulation — while still covering the full resistance range.

---

## Session Output

Each session generates a timestamped CSV file:

```
timestamp_ms, fsr_raw, fsr_smoothed, feedback_target, feedback_current, pot_value, slave_angle
```

Session files are stored in `dashboard/sessions/` and can be analysed independently through the dashboard's session analysis tab.

---

## Classification Results (Example Session)

| Metric | Value |
|---|---|
| Session duration | 26.1 s |
| Total samples | 330 |
| Grip events detected | 4 |
| Signal stability | 35.0% |
| Signal spikes | 12 |
| Mean FSR (active) | 569.9 / 1023 |
| Peak FSR | 718.0 / 1023 |
| Min FSR | 18.0 / 1023 |

---

## Authors

**Iñigo Del Valle** — Hardware design, Arduino firmware (logarithmic mapping, filtering, smooth servo control), AI integration, system integration
**Luis Aja** — Project concept and mechanical design
**Aimar García** — Design Thinking, ideation, prototyping
**Nicolás Ruiz de Azúa** — State of the art, documentation, validation

University of Deusto · Biomedical Engineering · May 2026

---

## Disclaimer

This prototype is an educational device developed for academic purposes in a controlled laboratory environment. It is not intended for clinical use, patient contact, or sterile environments. All outputs should be interpreted as demonstrative and non-clinical.
