import sys
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt6.QtGui import QIcon
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout, QMessageBox, QLineEdit,
    QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# QSS
QSS = r"""
QMainWindow { background: #081025; color: #e6eef8; }
QLabel { color: #e6eef8; }
QPushButton { background: #0f1724; color: #e6eef8; padding: 8px 12px; border-radius:8px; }
QPushButton:hover { background: #111827; }
QTableWidget { background: #071029; color: #e6eef8; gridline-color: #102030; }
QHeaderView::section { background: #0b1624; color: #e6eef8; padding: 6px; }
QProgressBar { background: #0b1220; color: #e6eef8; border-radius: 6px; }
QGroupBox { border: 1px solid #102030; margin-top: 6px; padding: 6px; }
QLineEdit, QSpinBox, QDoubleSpinBox { background: #071029; color: #e6eef8; }
"""

# Worker Threads
class TrainerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def run(self):
        try:
            self.progress.emit(5)
            # simple pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
            ])
            self.progress.emit(20)
            pipeline.fit(self.X, self.y)
            self.progress.emit(90)
            self.finished.emit(pipeline)
            self.progress.emit(100)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)

class PredictorThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pipeline, X):
        super().__init__()
        self.pipeline = pipeline
        self.X = X

    def run(self):
        try:
            preds = self.pipeline.predict(self.X)
            self.finished.emit(preds)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)

# Feature engineering
def prepare_features(df: pd.DataFrame) -> (np.ndarray, pd.DataFrame):
    df2 = df.copy()
    # Ensure necessary columns
    df2['cvss'] = pd.to_numeric(df2.get('cvss', 0), errors='coerce').fillna(0)
    # exploit_available: 1/0
    df2['exploit_available'] = df2.get('exploit_available', 0).apply(lambda x: 1 if str(x).strip() in ['1','True','true','yes','y'] else 0) if 'exploit_available' in df2.columns else 0
    # patch_available
    df2['patch_available'] = df2.get('patch_available', 0).apply(lambda x: 1 if str(x).strip() in ['1','True','true','yes','y'] else 0) if 'patch_available' in df2.columns else 0
    # assets_affected
    df2['assets_affected'] = pd.to_numeric(df2.get('assets_affected', 1), errors='coerce').fillna(1)
    # asset_criticality
    df2['asset_criticality'] = pd.to_numeric(df2.get('asset_criticality', 3), errors='coerce').fillna(3)
    # days since published
    def days_since(publ):
        try:
            d = pd.to_datetime(publ)
            return (pd.Timestamp.now() - d).days
        except Exception:
            return 365
    df2['days_since_published'] = df2.get('published_date', None).apply(days_since) if 'published_date' in df2.columns else 365
    # derived
    df2['exposure_score'] = (df2['assets_affected'] * df2['asset_criticality'])
    df2['time_decay'] = 1 / (1 + np.log1p(df2['days_since_published']))
    # features list
    features = ['cvss','exploit_available','patch_available','assets_affected','asset_criticality','exposure_score','time_decay']
    X = df2[features].fillna(0).values.astype(float)
    return X, df2

# Synthetic training label generator
def synthesize_priority_labels(df: pd.DataFrame) -> np.ndarray:
    # rule-based synthetic label: combine CVSS, exploit, asset criticality, patch availability, and exposure
    cvss = df['cvss'].values
    exploit = df['exploit_available'].values
    patch = df['patch_available'].values
    exposure = df['exposure_score'].values
    time_decay = df['time_decay'].values
    # base score from cvss (scaled 0-1)
    base = cvss / 10.0
    score = base * (0.5 + 0.5*exploit)  # exploit doubles weight
    score += 0.15 * (exposure / (1 + exposure))
    score += 0.1 * (1 - patch)  # no patch increases priority
    score *= (1 + 0.2 * time_decay)
    # map to 0-100
    score = np.clip(score, 0, 2)  # keep reasonable
    return (score / 2.0 * 100.0)

# GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('VulnPrioritizer - AI Vulnerability Prioritization')
        self.setWindowIcon(QIcon("AI4.ico"))
        self.resize(1200, 760)
        self.pipeline = None
        self.df = None

        central = QWidget(); self.setCentralWidget(central)
        main = QHBoxLayout(); central.setLayout(main)

        # Left panel: controls
        left = QVBoxLayout(); main.addLayout(left, 1)

        btn_load = QPushButton('Load Vulnerabilities CSV')
        btn_load.clicked.connect(self.load_csv)
        btn_generate = QPushButton('Generate Example Dataset')
        btn_generate.clicked.connect(self.generate_example)
        btn_train = QPushButton('Train Prioritization Model')
        btn_train.clicked.connect(self.train_model)
        btn_predict = QPushButton('Run Prioritization')
        btn_predict.clicked.connect(self.run_prioritization)
        btn_export = QPushButton('Export Results (CSV)')
        btn_export.clicked.connect(self.export_results)
        btn_save = QPushButton('Save Model (.joblib)')
        btn_save.clicked.connect(self.save_model)
        btn_load_model = QPushButton('Load Model (.joblib)')
        btn_load_model.clicked.connect(self.load_model)

        left.addWidget(btn_load)
        left.addWidget(btn_generate)
        left.addWidget(btn_train)
        left.addWidget(btn_predict)
        left.addWidget(btn_export)
        left.addWidget(btn_save)
        left.addWidget(btn_load_model)

        # Model config
        grp = QGroupBox('Model Config (quick)')
        form = QFormLayout()
        self.spin_estimators = QSpinBox(); self.spin_estimators.setRange(10,1000); self.spin_estimators.setValue(200)
        self.spin_test_size = QDoubleSpinBox(); self.spin_test_size.setRange(0.05,0.5); self.spin_test_size.setSingleStep(0.05); self.spin_test_size.setValue(0.2)
        form.addRow('n_estimators', self.spin_estimators)
        form.addRow('test_size', self.spin_test_size)
        grp.setLayout(form)
        left.addWidget(grp)

        # Progress and log
        self.progress = QProgressBar(); self.progress.setValue(0)
        left.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(160)
        left.addWidget(self.log)

        # Right: table and visualization
        right = QVBoxLayout(); main.addLayout(right, 3)

        self.table = QTableWidget();
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(['cve_id','cvss','published_date','exploit_available','patch_available','assets_affected','asset_criticality','priority','description'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self.on_table_item_changed)
        right.addWidget(self.table)

        self.fig = Figure(figsize=(6,2.5))
        self.canvas = FigureCanvas(self.fig)
        right.addWidget(self.canvas)

        # Bottom quick filter
        bottom = QHBoxLayout(); right.addLayout(bottom)
        bottom.addWidget(QLabel('Show top N priorities:'))
        self.top_n = QSpinBox(); self.top_n.setRange(1,1000); self.top_n.setValue(10)
        bottom.addWidget(self.top_n)
        btn_filter = QPushButton('Filter Top N'); btn_filter.clicked.connect(self.filter_top_n)
        bottom.addWidget(btn_filter)
        btn_reset = QPushButton('Reset View'); btn_reset.clicked.connect(self.reset_view)
        bottom.addWidget(btn_reset)

        self.setStyleSheet(QSS)

    # Helpers
    def log_msg(self, s):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log.append(f'[{ts}] {s}')

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open vulnerabilities CSV', os.getcwd(), 'CSV Files (*.csv)')
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.df = df
            self.log_msg(f'Loaded CSV: {path} ({len(df)} rows)')
            self.populate_table()
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))
            self.log_msg('Load CSV error: ' + str(e))

    def generate_example(self):
        rows = []
        for i in range(120):
            cvss = round(np.clip(np.random.normal(7,1.5), 0, 10),1)
            exploit = 1 if np.random.rand() < 0.25 else 0
            patch = 1 if np.random.rand() < 0.6 else 0
            assets = np.random.randint(1,50)
            criticality = np.random.randint(1,6)
            pub = (pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1,1000))).strftime('%Y-%m-%d')
            rows.append({'cve_id': f'CVE-2025-{1000+i}','cvss':cvss,'published_date':pub,'exploit_available':exploit,'patch_available':patch,'assets_affected':assets,'asset_criticality':criticality,'description':'Example vuln'})
        self.df = pd.DataFrame(rows)
        self.log_msg('Generated synthetic example dataset')
        self.populate_table()

    def populate_table(self):
        if self.df is None:
            return
        self.table.blockSignals(True)
        display = self.df.copy()
        # ensure priority column exists
        if 'priority' not in display.columns:
            display['priority'] = 0.0
        self.table.setRowCount(len(display))
        for r, (_, row) in enumerate(display.iterrows()):
            self.table.setItem(r, 0, QTableWidgetItem(str(row.get('cve_id',''))))
            self.table.setItem(r, 1, QTableWidgetItem(str(row.get('cvss',''))))
            self.table.setItem(r, 2, QTableWidgetItem(str(row.get('published_date',''))))
            self.table.setItem(r, 3, QTableWidgetItem(str(row.get('exploit_available',''))))
            self.table.setItem(r, 4, QTableWidgetItem(str(row.get('patch_available',''))))
            self.table.setItem(r, 5, QTableWidgetItem(str(row.get('assets_affected',''))))
            self.table.setItem(r, 6, QTableWidgetItem(str(row.get('asset_criticality',''))))
            self.table.setItem(r, 7, QTableWidgetItem(str(round(float(row.get('priority',0)),2))))
            self.table.setItem(r, 8, QTableWidgetItem(str(row.get('description',''))))
        self.table.blockSignals(False)
        self.plot_priorities()
        self.log_msg('Table populated')

    def train_model(self):
        if self.df is None:
            QMessageBox.warning(self, 'No data', 'Load or generate a dataset first')
            return
        try:
            X, df2 = prepare_features(self.df)
            y = synthesize_priority_labels(df2)
            # threaded training
            self.trainer = TrainerThread(X, y)
            self.trainer.progress.connect(self.progress.setValue)
            self.trainer.finished.connect(self.on_trained)
            self.trainer.error.connect(self.on_worker_error)
            self.trainer.start()
            self.log_msg('Training started in background...')
        except Exception as e:
            QMessageBox.critical(self, 'Train error', str(e))
            self.log_msg('Train error: ' + str(e))

    def on_trained(self, pipeline):
        self.pipeline = pipeline
        self.progress.setValue(0)
        self.log_msg('Training finished - model ready')

    def run_prioritization(self):
        if self.df is None:
            QMessageBox.warning(self, 'No data', 'Load or generate dataset first')
            return
        if self.pipeline is None:
            QMessageBox.warning(self, 'No model', 'Train or load a model first')
            return
        try:
            X, df2 = prepare_features(self.df)
            self.predictor = PredictorThread(self.pipeline, X)
            self.predictor.finished.connect(lambda preds: self.on_predicted(preds, df2))
            self.predictor.error.connect(self.on_worker_error)
            self.predictor.start()
            self.log_msg('Running prioritization...')
        except Exception as e:
            QMessageBox.critical(self, 'Predict error', str(e))
            self.log_msg('Predict error: ' + str(e))

    def on_predicted(self, preds, df2):
        try:
            df2['priority'] = np.clip(preds, 0, 100)
            self.df = df2
            self.populate_table()
            self.log_msg('Prioritization finished')
        except Exception as e:
            self.log_msg('Post-predict error: ' + str(e))

    def save_model(self):
        if self.pipeline is None:
            QMessageBox.warning(self, 'No model', 'Train a model first')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save model', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        joblib.dump(self.pipeline, path)
        self.log_msg(f'Model saved to {path}')

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load model', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        try:
            self.pipeline = joblib.load(path)
            self.log_msg(f'Model loaded from {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Load model error', str(e))
            self.log_msg('Load model error: ' + str(e))

    def export_results(self):
        if self.df is None or 'priority' not in self.df.columns:
            QMessageBox.warning(self, 'No results', 'Run prioritization first')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export results', os.getcwd(), 'CSV Files (*.csv)')
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            self.log_msg(f'Results exported to {path}')
            QMessageBox.information(self, 'Exported', f'Results exported to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Export error', str(e))
            self.log_msg('Export failed: ' + str(e))

    def filter_top_n(self):
        if self.df is None or 'priority' not in self.df.columns:
            QMessageBox.warning(self, 'No data', 'Run prioritization first')
            return
        n = int(self.top_n.value())
        df_sorted = self.df.sort_values('priority', ascending=False).head(n)
        self.table.blockSignals(True)
        self.table.setRowCount(len(df_sorted))
        for r, (_, row) in enumerate(df_sorted.iterrows()):
            self.table.setItem(r, 0, QTableWidgetItem(str(row.get('cve_id',''))))
            self.table.setItem(r, 1, QTableWidgetItem(str(row.get('cvss',''))))
            self.table.setItem(r, 2, QTableWidgetItem(str(row.get('published_date',''))))
            self.table.setItem(r, 3, QTableWidgetItem(str(row.get('exploit_available',''))))
            self.table.setItem(r, 4, QTableWidgetItem(str(row.get('patch_available',''))))
            self.table.setItem(r, 5, QTableWidgetItem(str(row.get('assets_affected',''))))
            self.table.setItem(r, 6, QTableWidgetItem(str(row.get('asset_criticality',''))))
            self.table.setItem(r, 7, QTableWidgetItem(str(round(float(row.get('priority',0)),2))))
            self.table.setItem(r, 8, QTableWidgetItem(str(row.get('description',''))))
        self.table.blockSignals(False)
        self.plot_priorities()
        self.log_msg(f'Showing top {n} vulnerabilities')

    def reset_view(self):
        self.populate_table()

    def plot_priorities(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if self.df is None or 'priority' not in self.df.columns:
            ax.text(0.5,0.5,'No data', ha='center', va='center')
        else:
            pr = self.df['priority'].fillna(0).values
            ax.hist(pr, bins=20)
            ax.set_title('Priority distribution')
            ax.set_xlabel('Priority score')
        self.canvas.draw()

    # Table edits -> real-time recalculation
    def on_table_item_changed(self, item: QTableWidgetItem):
        try:
            row = item.row()
            col = item.column()
            # if user edited asset_criticality, exploit flag, assets_affected or cvss, update df and rerun prediction for that row
            key_map = {1: 'cvss', 3: 'exploit_available', 4: 'patch_available', 5: 'assets_affected', 6: 'asset_criticality', 7: 'priority'}
            if col in key_map and self.df is not None:
                # update df
                val = item.text()
                colname = key_map[col]
                # coerce
                if colname in ['cvss']:
                    self.df.at[row, colname] = float(val)
                elif colname in ['exploit_available','patch_available']:
                    self.df.at[row, colname] = 1 if val.strip().lower() in ['1','true','yes','y'] else 0
                else:
                    self.df.at[row, colname] = pd.to_numeric(val, errors='coerce')
                # recalc single prediction if pipeline exists
                if self.pipeline is not None:
                    X, df2 = prepare_features(self.df)
                    single_X = X[[row], :]
                    try:
                        pred = float(self.pipeline.predict(single_X)[0])
                        self.df.at[row, 'priority'] = np.clip(pred, 0, 100)
                        # update table silently
                        self.table.blockSignals(True)
                        self.table.setItem(row, 7, QTableWidgetItem(str(round(float(self.df.at[row,'priority']),2))))
                        self.table.blockSignals(False)
                        self.plot_priorities()
                        self.log_msg(f'Recomputed priority for row {row}')
                    except Exception as e:
                        self.log_msg('Recompute error: ' + str(e))
        except Exception as e:
            self.log_msg('Table change handler error: ' + str(e))

    def on_worker_error(self, tb):
        QMessageBox.critical(self, 'Worker error', 'A background thread failed - see log')
        self.log_msg('Worker error:\n' + str(tb))

# Main entry
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())