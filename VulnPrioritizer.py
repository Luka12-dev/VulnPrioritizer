import sys
import os
import re
import math
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt6.QtGui import QIcon
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QCheckBox, QGroupBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

# QSS
QSS = r"""
QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #071029, stop:1 #071a2f); color: #e6eef8; }
QLabel { color: #e6eef8; }
QLineEdit { background: #071828; color: #e6eef8; padding:8px; border-radius:6px; }
QPushButton { background: #0f1724; color: #e6eef8; padding:8px 12px; border-radius:6px; }
QPushButton:hover { background: #111827; }
QProgressBar { background: #03101a; color: #e6eef8; border-radius:8px; height:18px; }
QGroupBox { border: 1px solid #082233; margin-top: 6px; padding: 8px; }
"""

# Feature extraction

COMMON_PASSWORDS = set([
        'password','123456','qwerty','letmein','admin','welcome','iloveyou','000000',
        '123456789','12345678','12345','123123','111111','1234','abc123','1234567','dragon',
        'monkey','football','baseball','shadow','master','superman','696969','jordan','harley',
        'computer','hunter','buster','soccer','batman','thomas','jessica','michael','charlie',
        'pepper','secret','andrew','tigger','sunshine','iloveu','princess','flower','hottie',
        'loveme','zaq12wsx','starwars','hello','freedom','whatever','qazwsx','trustno1',
        'password1','qwerty123','letmein1','welcome1','123qwe','654321','donald','pokemon',
        'naruto','superstar','killer','mustang','jordan23','summer','love','angel','cheese',
        'daniel','family','william','george','maggie','cookie','ginger','amanda','justin',
        'music','happy','peppermint','blink182','matrix','carlos','samsung','google','ninja',
        '123abc','asdfgh','zxcvbn','qwertyuiop','1q2w3e4r','q1w2e3r4','admin123','welcome123',
        'iloveyou2','princess1','love123','football1','baseball1','soccer1','basketball',
        'hockey','tennis','guitar','bandit','ranger','cowboy','indigo','orange','yellow',
        'purple','green','blue','black','white','red','silver','gold','diamond','crystal',
        'mercedes','corvette','porsche','ferrari','honda','toyota','chevy','ford','bmw',
        'windows','mypass','pass123','passw0rd','pa55word','letmein123','123456a','abc12345',
        '1qaz2wsx','qwerty1','asdf1234','ilovegod','jesus','god123','heaven','faith','church',
        'america','canada','mexico','brazil','england','france','germany','italy','spain',
        'japan','china','india','serbia','russia','ukraine','poland','turkey','sweden',
        'denmark','norway','finland','ireland','scotland','wales','australia','newzealand',
        'thailand','vietnam','philippines','indonesia','malaysia','singapore','korea','egypt',
        'africa','nigeria','kenya','southafrica','morocco','algeria','tunisia','worldcup',
        'arsenal','chelsea','liverpool','manutd','realmadrid','barcelona','juventus','milan',
        'bayern','psg','olympics','champion','winner','loser','gameover','playstation',
        'nintendo','xbox','minecraft','fortnite','roblox','amongus','steam','counterstrike',
        'halflife','portal','doom','quake','diablo','warcraft','starcraft','overwatch',
        'apple','banana','orange1','lemon','grape','mango','kiwi','watermelon','strawberry',
        'chocolate','candy','icecream','coffee','pizza','burger','taco','sushi','ramen',
        'donut','cookie1','cake123','cupcake','beer','vodka','whiskey','tequila','rum',
        'marijuana','cocaine','heroin','weed420','stoner','drugs','smoking','alcohol','party',
        'dance','music123','rapstar','eminem','drdre','snoopdogg','2pac','biggie','metallica',
        'nirvana','queen','beatles','elvis','madonna','beyonce','rihanna','shakira','ladygaga',
        'taylor','selena','miley','justin1','bieber','harry','zayn','niall','liam','louis',
        'onepiece','naruto123','sasuke','goku','vegeta','dragonball','pokemon123','pikachu',
        'charmander','bulbasaur','squirtle','ashketchum','yugioh','digimon','marvel','dccomics',
        'spiderman','ironman','batman123','superman1','hulk','thor','loki','captain','flash',
        'joker','harleyquinn','deadpool','wolverine','avengers','justiceleague','starwars1',
        'jedi','sith','vader','yoda','skywalker','obiwan','han','leia','chewbacca','r2d2',
        'c3po','darth','empire','galaxy','universe','planet','earth','moon','sun','mars',
        'venus','mercury','saturn','uranus','neptune','pluto','cosmos','space','stars',
        'blackhole','milkyway','bigbang','eclipse','meteor','asteroid','comet','alien',
        'ufo','area51','extraterrestrial','robot','android','cyborg','ai','machine','hacker',
        'root','toor','nmap','kali','linux','ubuntu','windows1','macos','iphone','samsung1',
        'galaxy1','pixel','nokia','sony','xiaomi','huawei','oppo','vivo','realme','lenovo',
        'dell','hp','asus','acer','msi','thinkpad','alienware','razor','logitech','corsair'
])

def entropy_estimate(pw: str) -> float:
    # Shannon entropy estimate on characters
    if not pw:
        return 0.0
    from collections import Counter
    counts = Counter(pw)
    probs = [v/len(pw) for v in counts.values()]
    H = -sum(p*math.log2(p) for p in probs)
    return H

def char_class_counts(pw: str):
    lower = sum(1 for c in pw if c.islower())
    upper = sum(1 for c in pw if c.isupper())
    digits = sum(1 for c in pw if c.isdigit())
    special = sum(1 for c in pw if not c.isalnum())
    return lower, upper, digits, special

def has_common_pattern(pw: str):
    s = pw.lower()
    if s in COMMON_PASSWORDS:
        return 1
    # sequences like 1234, abcd
    seqs = ['1234','abcd','qwerty','4321','pass']
    return int(any(ss in s for ss in seqs))

def extract_features(pw: str) -> dict:
    lower, upper, digits, special = char_class_counts(pw)
    entropy = entropy_estimate(pw)
    length = len(pw)
    pattern = has_common_pattern(pw)
    diversity = sum(1 for v in [lower>0, upper>0, digits>0, special>0] if v)
    features = {
        'length': length,
        'entropy': entropy,
        'lower': lower,
        'upper': upper,
        'digits': digits,
        'special': special,
        'pattern_common': pattern,
        'diversity': diversity
    }
    return features

# Worker threads
class PredictorThread(QThread):
    finished = pyqtSignal(float, dict)
    error = pyqtSignal(str)

    def __init__(self, model, features):
        super().__init__()
        self.model = model
        self.features = features

    def run(self):
        try:
            X = [self.features]
            dv = self.model['dv']
            Xv = dv.transform(X)
            score = float(self.model['clf'].predict(Xv)[0])
            self.finished.emit(score, self.features)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(tb)

# Simple trainer
def build_model():
    # creates a small synthetic dataset mapping features to 0-100 score
    rows = []
    def score_from_features(f):
        s = f['entropy'] * 10 + f['diversity']*5 + (f['length']*2)
        s -= f['pattern_common']*30
        s = max(0, min(100, s))
        return s

    samples = set([
        'password', '123456', 'qwerty', 'letmein', 'admin', 'welcome', 'iloveyou', '000000',
        '123456789', '12345678', '12345', '123123', '111111', '1234', 'abc123', '1234567', 'dragon',
        'monkey', 'football', 'baseball', 'shadow', 'master', 'superman', '696969', 'jordan', 'harley',
        'computer', 'hunter', 'buster', 'soccer', 'batman', 'thomas', 'jessica', 'michael', 'charlie',
        'pepper', 'secret', 'andrew', 'tigger', 'sunshine', 'iloveu', 'princess', 'flower', 'hottie',
        'loveme', 'zaq12wsx', 'starwars', 'hello', 'freedom', 'whatever', 'qazwsx', 'trustno1',
        'password1', 'qwerty123', 'letmein1', 'welcome1', '123qwe', '654321', 'donald', 'pokemon',
        'naruto', 'superstar', 'killer', 'mustang', 'jordan23', 'summer', 'love', 'angel', 'cheese',
        'daniel', 'family', 'william', 'george', 'maggie', 'cookie', 'ginger', 'amanda', 'justin',
        'music', 'happy', 'peppermint', 'blink182', 'matrix', 'carlos', 'samsung', 'google', 'ninja',
        '123abc', 'asdfgh', 'zxcvbn', 'qwertyuiop', '1q2w3e4r', 'q1w2e3r4', 'admin123', 'welcome123',
        'iloveyou2', 'princess1', 'love123', 'football1', 'baseball1', 'soccer1', 'basketball',
        'hockey', 'tennis', 'guitar', 'bandit', 'ranger', 'cowboy', 'indigo', 'orange', 'yellow',
        'purple', 'green', 'blue', 'black', 'white', 'red', 'silver', 'gold', 'diamond', 'crystal',
        'mercedes', 'corvette', 'porsche', 'ferrari', 'honda', 'toyota', 'chevy', 'ford', 'bmw',
        'windows', 'mypass', 'pass123', 'passw0rd', 'pa55word', 'letmein123', '123456a', 'abc12345',
        '1qaz2wsx', 'qwerty1', 'asdf1234', 'ilovegod', 'jesus', 'god123', 'heaven', 'faith', 'church',
        'america', 'canada', 'mexico', 'brazil', 'england', 'france', 'germany', 'italy', 'spain',
        'japan', 'china', 'india', 'serbia', 'russia', 'ukraine', 'poland', 'turkey', 'sweden',
        'denmark', 'norway', 'finland', 'ireland', 'scotland', 'wales', 'australia', 'newzealand',
        'thailand', 'vietnam', 'philippines', 'indonesia', 'malaysia', 'singapore', 'korea', 'egypt',
        'africa', 'nigeria', 'kenya', 'southafrica', 'morocco', 'algeria', 'tunisia', 'worldcup',
        'arsenal', 'chelsea', 'liverpool', 'manutd', 'realmadrid', 'barcelona', 'juventus', 'milan',
        'bayern', 'psg', 'olympics', 'champion', 'winner', 'loser', 'gameover', 'playstation',
        'nintendo', 'xbox', 'minecraft', 'fortnite', 'roblox', 'amongus', 'steam', 'counterstrike',
        'halflife', 'portal', 'doom', 'quake', 'diablo', 'warcraft', 'starcraft', 'overwatch',
        'apple', 'banana', 'orange1', 'lemon', 'grape', 'mango', 'kiwi', 'watermelon', 'strawberry',
        'chocolate', 'candy', 'icecream', 'coffee', 'pizza', 'burger', 'taco', 'sushi', 'ramen',
        'donut', 'cookie1', 'cake123', 'cupcake', 'beer', 'vodka', 'whiskey', 'tequila', 'rum',
        'marijuana', 'cocaine', 'heroin', 'weed420', 'stoner', 'drugs', 'smoking', 'alcohol', 'party',
        'dance', 'music123', 'rapstar', 'eminem', 'drdre', 'snoopdogg', '2pac', 'biggie', 'metallica',
        'nirvana', 'queen', 'beatles', 'elvis', 'madonna', 'beyonce', 'rihanna', 'shakira', 'ladygaga',
        'taylor', 'selena', 'miley', 'justin1', 'bieber', 'harry', 'zayn', 'niall', 'liam', 'louis',
        'onepiece', 'naruto123', 'sasuke', 'goku', 'vegeta', 'dragonball', 'pokemon123', 'pikachu',
        'charmander', 'bulbasaur', 'squirtle', 'ashketchum', 'yugioh', 'digimon', 'marvel', 'dccomics',
        'spiderman', 'ironman', 'batman123', 'superman1', 'hulk', 'thor', 'loki', 'captain', 'flash',
        'joker', 'harleyquinn', 'deadpool', 'wolverine', 'avengers', 'justiceleague', 'starwars1',
        'jedi', 'sith', 'vader', 'yoda', 'skywalker', 'obiwan', 'han', 'leia', 'chewbacca', 'r2d2',
        'c3po', 'darth', 'empire', 'galaxy', 'universe', 'planet', 'earth', 'moon', 'sun', 'mars',
        'venus', 'mercury', 'saturn', 'uranus', 'neptune', 'pluto', 'cosmos', 'space', 'stars',
        'blackhole', 'milkyway', 'bigbang', 'eclipse', 'meteor', 'asteroid', 'comet', 'alien',
        'ufo', 'area51', 'extraterrestrial', 'robot', 'android', 'cyborg', 'ai', 'machine', 'hacker',
        'root', 'toor', 'nmap', 'kali', 'linux', 'ubuntu', 'windows1', 'macos', 'iphone', 'samsung1',
        'galaxy1', 'pixel', 'nokia', 'sony', 'xiaomi', 'huawei', 'oppo', 'vivo', 'realme', 'lenovo',
        'dell', 'hp', 'asus', 'acer', 'msi', 'thinkpad', 'alienware', 'razor', 'logitech', 'corsair'
    ])
    for s in samples:
        f = extract_features(s)
        rows.append((f, score_from_features(f)))
    # build model
    df = pd.DataFrame([{**r[0], 'score': r[1]} for r in rows])
    X = df.drop(columns=['score']).to_dict(orient='records')
    y = df['score'].values
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction import DictVectorizer
    dv = DictVectorizer(sparse=False)
    Xv = dv.fit_transform(X)
    clf = RandomForestRegressor(n_estimators=200, random_state=42)
    clf.fit(Xv, y)
    return {'dv': dv, 'clf': clf}

# MainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PasswordStrengthML - Real-time Predictor')
        self.setWindowIcon(QIcon("AI5.ico"))
        self.resize(900,500)
        self.model = build_model()
        self._predictor = None

        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(); central.setLayout(layout)

        title = QLabel('Password Strength - ML Predictor')
        title.setStyleSheet('font-weight:700; font-size:18px; color:#dbeafe;')
        layout.addWidget(title)

        instr = QLabel('Type a password below. The app predicts a strength score (0-100) in real-time. Avoid testing real passwords.')
        layout.addWidget(instr)

        self.input_pw = QLineEdit(); self.input_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_pw.setPlaceholderText('Type password here...')
        layout.addWidget(self.input_pw)

        self.show_pw_cb = QCheckBox('Show password'); self.show_pw_cb.stateChanged.connect(self.on_toggle_show)
        layout.addWidget(self.show_pw_cb)

        # results
        self.score_bar = QProgressBar(); self.score_bar.setRange(0,100); self.score_bar.setValue(0)
        layout.addWidget(self.score_bar)
        self.score_label = QLabel('Strength: N/A')
        layout.addWidget(self.score_label)

        # tokenized features display
        self.features_box = QTextEdit(); self.features_box.setReadOnly(True); self.features_box.setFixedHeight(140)
        layout.addWidget(QLabel('Extracted features:'))
        layout.addWidget(self.features_box)

        # buttons
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton('Save Model'); self.btn_save.clicked.connect(self.on_save_model)
        self.btn_load = QPushButton('Load Model'); self.btn_load.clicked.connect(self.on_load_model)
        btn_layout.addWidget(self.btn_save); btn_layout.addWidget(self.btn_load)
        layout.addLayout(btn_layout)

        self.setStyleSheet(QSS)

        # debounce timer
        self._debounce = QTimer(); self._debounce.setSingleShot(True); self._debounce.timeout.connect(self.on_predict_debounced)
        self.input_pw.textChanged.connect(self._on_pw_change)

        # initial help text
        self.features_box.setPlainText('Hints: longer passwords, mixed character classes and high entropy increase strength. Avoid common passwords like "password" or "123456".')

    def on_toggle_show(self):
        if self.show_pw_cb.isChecked():
            self.input_pw.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.input_pw.setEchoMode(QLineEdit.EchoMode.Password)

    def _on_pw_change(self):
        self._debounce.start(250)

    def on_predict_debounced(self):
        pw = self.input_pw.text()
        features = extract_features(pw)
        # update features box
        lines = [f"{k}: {v}" for k,v in features.items()]
        self.features_box.setPlainText('\n'.join(lines))
        # threaded prediction
        self._predictor = PredictorThread(self.model, features)
        self._predictor.finished.connect(self._on_predicted)
        self._predictor.error.connect(self._on_error)
        self._predictor.start()

    def _on_predicted(self, score, features):
        try:
            val = int(round(score))
            self.score_bar.setValue(val)
            self.score_label.setText(f'Strength: {val}/100')
        except Exception as e:
            print('Prediction handling error', e)

    def _on_error(self, tb):
        QMessageBox.critical(self, 'Prediction error', 'A background prediction failed - see console')
        print(tb)

    def on_save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save model', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        joblib.dump(self.model, path)
        QMessageBox.information(self, 'Saved', f'Model saved to {path}')

    def on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load model', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        try:
            obj = joblib.load(path)
            self.model = obj
            QMessageBox.information(self, 'Loaded', f'Model loaded from {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))

# main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
