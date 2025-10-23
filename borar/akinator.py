"""
PestAkinator - Identificador tipo Akinator para plagas de cultivo
Archivo único: pest_akinator.py

Descripción:
Un prototipo funcional (CLI + pequeño servidor web opcional) que implementa
un sistema de preguntas para identificar plagas agrícolas usando: 
- Representación de plagas como vectores de atributos (probabilidades).
- Actualización Bayesiana de probabilidades de candidatos según respuestas.
- Selección de la siguiente pregunta basada en ganancia de información (entropía).
- Capacidad de aprender una plaga nueva cuando no encuentra la correcta.

Uso:
1) CLI (por defecto):
   python pest_akinator.py

2) Modo servidor web (opcional, requiere flask):
   FLASK_MODE=1 python pest_akinator.py
   Visitar: http://127.0.0.1:5000/ (interfaz mínima)

Notas:
- Este código es un prototipo educativo. Para producción, mejorar dataset,
  manejar seguridad, concurrence, y proveer imágenes/ML para reconocimiento visual.

"""

import json
import math
import os
import random
import threading
import sys
from copy import deepcopy

# Optional web UI
FLASK_MODE = os.getenv('FLASK_MODE', '0') == '1'
if FLASK_MODE:
    try:
        from flask import Flask, request, render_template_string, redirect, url_for
    except Exception as e:
        print('FLASK no está instalado o falló la importación. Ejecuta pip install flask para usar el modo web.')
        FLASK_MODE = False

DB_FILE = 'pest_db.json'

# ---------- Dataset inicial (ejemplos) ----------
INITIAL_DB = {
    "pests": [
        {
            "id": "spodoptera_frugiperda",
            "name": "Gusano cogollero (Spodoptera frugiperda)",
            "common_names": ["gusano cogollero","fall armyworm"],
            "notes": "Principal plaga del maíz; oruga que come hojas y cogollos.",
            "attributes": {
                "host_maize": 0.95,
                "chews_leaves": 0.95,
                "frass_present": 0.9,
                "nocturnal": 0.8,
                "larva_visible_on_whorl": 0.9,
                "causes_dead_hearts": 0.7
            }
        },
        {
            "id": "ostrinia_nubilalis",
            "name": "Barrenador del maíz (Ostrinia nubilalis)",
            "common_names": ["barrenador","corn borer"],
            "notes": "Larva perfora tallos y mazorcas; síntomas internos y marchitez.",
            "attributes": {
                "host_maize": 0.9,
                "chews_leaves": 0.4,
                "frass_present": 0.6,
                "larva_inside_stem": 0.9,
                "holes_in_stem": 0.8,
                "causes_dead_hearts": 0.6
            }
        },
        {
            "id": "tuta_absoluta",
            "name": "Minador de la hoja de tomate (Tuta absoluta)",
            "common_names": ["tuta absoluta","minador del tomate"],
            "notes": "Ataca tomate; galerías en hojas y frutos y defoliación.",
            "attributes": {
                "host_tomato": 0.95,
                "leaf_mines": 0.95,
                "small_holes_in_leaves": 0.9,
                "high_reproduction": 0.9,
                "larva_small_and_green": 0.7
            }
        },
        {
            "id": "bemisia_tabaci",
            "name": "Mosca blanca (Bemisia tabaci)",
            "common_names": ["mosca blanca","whitefly"],
            "notes": "Pequeños insectos en envés de hojas; succión y melaza.",
            "attributes": {
                "sucking_insect": 0.95,
                "white_tiny_insects_on_underside": 0.95,
                "honeydew_exudate": 0.9,
                "transmits_viruses": 0.7
            }
        },
        {
            "id": "helicoverpa_armigera",
            "name": "Helicoverpa (gusano del algodón/tomate)",
            "common_names": ["gusano del algodón","Helicoverpa"],
            "notes": "Oruga polífaga que come flores, frutos y hojas.",
            "attributes": {
                "chews_flowers_and_fruits": 0.9,
                "polyphagous": 0.85,
                "visible_caterpillar": 0.9,
                "frass_present": 0.7
            }
        }
    ],
    "questions": {
        "host_maize": "¿El cultivo afectado es maíz?",
        "chews_leaves": "¿Ves signos de masticación (bordes irregulares, agujeros grandes)?",
        "frass_present": "¿Hay excremento (frass) visible cerca de las hojas o cogollos?",
        "nocturnal": "¿El daño parece más activo durante la noche? (observado en la mañana con daño fresco)",
        "larva_visible_on_whorl": "¿Ves orugas dentro del cogollo/espiral de la planta?",
        "causes_dead_hearts": '¿Las plantas presentan "dead heart" (tallo central seco/roto)?',
        "larva_inside_stem": "¿Hay larvas dentro del tallo?",
        "holes_in_stem": "¿Notas agujeros en los tallos?",
        "host_tomato": "¿El cultivo afectado es tomate?",
        "leaf_mines": "¿Hay galerías o minas en las hojas?",
        "small_holes_in_leaves": "¿Ves hoyitos pequeños en las hojas?",
        "high_reproduction": "¿La plaga parece multiplicarse muy rápido (muchos individuos)?",
        "larva_small_and_green": "¿Las larvas son pequeñas y de color verdoso?",
        "sucking_insect": "¿El insecto parece succionar (hojas pegajosas, amarillamiento)?",
        "white_tiny_insects_on_underside": "¿Hay insectos blancos muy pequeños en el envés de las hojas?",
        "honeydew_exudate": "¿Notas melaza o residuo pegajoso en las hojas?",
        "transmits_viruses": "¿Se observan síntomas de virus en plantas (moteado, deformaciones)?",
        "chews_flowers_and_fruits": "¿Se comen flores y frutos?",
        "polyphagous": "¿Ataca muchos tipos de cultivos?",
        "visible_caterpillar": "¿Ves orugas grandes y visibles?"
    }
}

# ---------- Utilidades matemáticas ----------

def entropy(probs):
    e = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        e -= p * math.log2(p)
    return e

def normalize(d):
    total = sum(d.values())
    if total == 0:
        n = {k: 1.0 / len(d) for k in d}
        return n
    return {k: v / total for k, v in d.items()}

# ---------- Cargar / guardar DB ----------

def load_db(file=DB_FILE):
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        save_db(INITIAL_DB, file)
        return deepcopy(INITIAL_DB)

def save_db(db, file=DB_FILE):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# ---------- Motor de inferencia ----------
class PestAkinator:
    def __init__(self, db):
        self.db = db
        self.questions = db['questions']
        self.prior = {p['id']: 1.0 for p in db['pests']}
        self.probs = normalize(self.prior)
        self.asked = set()

    def get_attribute_prob(self, pest_id, attribute):
        pest = next((p for p in self.db['pests'] if p['id'] == pest_id), None)
        if not pest:
            return 0.5
        return pest['attributes'].get(attribute, 0.5)

    def expected_entropy_after_question(self, attribute):
        P_yes = sum(self.probs[p] * self.get_attribute_prob(p, attribute) for p in self.probs)
        P_no = sum(self.probs[p] * (1 - self.get_attribute_prob(p, attribute)) for p in self.probs)
        post_yes = {}
        post_no = {}
        for p in self.probs:
            a = self.get_attribute_prob(p, attribute)
            post_yes[p] = self.probs[p] * a
            post_no[p] = self.probs[p] * (1 - a)
        post_yes = normalize(post_yes)
        post_no = normalize(post_no)
        e_yes = entropy(post_yes.values())
        e_no = entropy(post_no.values())
        expected = P_yes * e_yes + P_no * e_no
        return expected

    def choose_best_question(self):
        best_q = None
        best_score = float('inf')
        for attr in self.questions:
            if attr in self.asked:
                continue
            exp_e = self.expected_entropy_after_question(attr)
            if exp_e < best_score:
                best_score = exp_e
                best_q = attr
        return best_q

    def update_with_answer(self, attribute, answer):
        for p in self.probs:
            a = self.get_attribute_prob(p, attribute)
            if answer == 'yes':
                likelihood = a
            elif answer == 'no':
                likelihood = 1 - a
            else:
                likelihood = 1.0
            self.probs[p] = self.probs[p] * likelihood
        self.probs = normalize(self.probs)
        self.asked.add(attribute)

    def top_candidates(self, n=3):
        items = sorted(self.probs.items(), key=lambda x: x[1], reverse=True)
        return items[:n]

    def run_cli(self):
        print('\n--- PestAkinator: Identificador de plagas (modo CLI) ---\n')
        round_count = 0
        while True:
            round_count += 1
            top = self.top_candidates(1)[0]
            if top[1] > 0.85 or len(self.asked) >= 12:
                pest = next((p for p in self.db['pests'] if p['id'] == top[0]), None)
                print(f"Creo que la plaga es: {pest['name']} (probabilidad {top[1]:.2f})")
                print('Notas:', pest.get('notes',''))
                confirmed = input('¿Es correcto? (s/n): ').strip().lower()
                if confirmed in ('s','si','y','yes'):
                    print('¡Genial! Se confirma la identificación.')
                    break
                else:
                    print('Lo siento. ¿Quieres enseñarme la plaga correcta para que aprenda? (s/n)')
                    teach = input().strip().lower()
                    if teach in ('s','si'):
                        self.teach_new_pest()
                        break
                    else:
                        print('Puedo intentar seguir preguntando.')
                        self.asked.clear()
                        continue
            q = self.choose_best_question()
            if q is None:
                print('No quedan preguntas útiles. Mostrar candidatos:')
                for pid, pr in self.top_candidates(5):
                    pest = next((p for p in self.db['pests'] if p['id']==pid), None)
                    print(f" - {pest['name']} ({pr:.2f})")
                teach = input('¿Aprendo la respuesta correcta? (s/n): ').strip().lower()
                if teach in ('s','si'):
                    self.teach_new_pest()
                break
            prompt = self.questions[q]
            print('\nPregunta:', prompt)
            ans = input('Responde: (s = sí, n = no, u = no sé) ').strip().lower()
            if ans in ('s','si','y'):
                self.update_with_answer(q, 'yes')
            elif ans in ('n','no'):
                self.update_with_answer(q, 'no')
            else:
                self.update_with_answer(q, 'unknown')
            print('\nCandidatos principales:')
            for pid, pr in self.top_candidates(3):
                pest = next((p for p in self.db['pests'] if p['id']==pid), None)
                print(f"  {pest['name']}: {pr:.2f}")

    def teach_new_pest(self):
        print('\n--- Enseñar nueva plaga al sistema ---')
        name = input('Nombre común o científico de la plaga: ').strip()
        notes = input('Notas breves (síntomas, cultivos afectados): ').strip()
        new_id = name.lower().replace(' ', '_') + '_' + str(random.randint(1000,9999))
        attrs = {}
        print('Para cada pregunta, responde s/n/u. Si no sabes, deja en blanco o escribe u.')
        for attr, qtext in self.questions.items():
            ans = input(qtext + ' ').strip().lower()
            if ans in ('s','si','y'):
                attrs[attr] = 0.95
            elif ans in ('n','no'):
                attrs[attr] = 0.05
            else:
                attrs[attr] = 0.5
        new_p = {
            'id': new_id,
            'name': name,
            'common_names': [name],
            'notes': notes,
            'attributes': attrs
        }
        self.db['pests'].append(new_p)
        save_db(self.db)
        print('Plaga agregada al sistema. Gracias.')

# ---------- Interfaz web mínima (opcional) ----------
WEB_TEMPLATE = '''
<!doctype html>
<title>PestAkinator</title>
<h1>PestAkinator - Identificador de plagas (web)</h1>
{% if question %}
  <form method="post">
    <p><b>Pregunta:</b> {{question}}</p>
    <p>
      <button name="answer" value="yes">Sí</button>
      <button name="answer" value="no">No</button>
      <button name="answer" value="unknown">No sé</button>
    </p>
  </form>
{% else %}
  <p>Inicia identificación:</p>
  <form method="post">
    <button name="start" value="1">Comenzar</button>
  </form>
{% endif %}

{% if candidates %}
  <h3>Candidatos</h3>
  <ul>
  {% for c in candidates %}
    <li>{{c[0]}}: {{'%.2f'|format(c[1])}}</li>
  {% endfor %}
  </ul>
{% endif %}

'''

if FLASK_MODE:
    app = Flask(__name__)
    global_akinator = None

    @app.route('/', methods=['GET','POST'])
    def index():
        global global_akinator
        if request.method == 'POST':
            if 'start' in request.form:
                db = load_db()
                global_akinator = PestAkinator(db)
            elif 'answer' in request.form and global_akinator is not None:
                attr = request.cookies.get('current_question')
                ans = request.form['answer']
                global_akinator.update_with_answer(attr, ans)
        if global_akinator is None:
            return render_template_string(WEB_TEMPLATE, question=None, candidates=None)
        q = global_akinator.choose_best_question()
        candidates = global_akinator.top_candidates(5)
        resp = render_template_string(WEB_TEMPLATE, question=(global_akinator.questions[q] if q else None), candidates=candidates)
        response = app.make_response(resp)
        if q:
            response.set_cookie('current_question', q)
        return response

# ---------- Ejecutable ----------
def main():
    db = load_db()
    akin = PestAkinator(db)
    if FLASK_MODE:
        print('Iniciando servidor Flask en http://127.0.0.1:5000/')
        app.run(debug=False)
    else:
        akin.run_cli()

if __name__ == '__main__':
    main()
