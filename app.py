# app.py

import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Initialiser l'application Flask
app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Remplacez par une clé secrète sécurisée

# Configurer le dossier de téléchargement et les extensions autorisées
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de taille : 16MB

# Créer le dossier de téléchargement s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle pré-entraîné
try:
    model = load_model('beauty_score_model.h5')
    print("Modèle chargé avec succès.")
except Exception as e:
    print("Erreur lors du chargement du modèle :", e)

# Fonction pour vérifier les types de fichiers autorisés
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonction pour prétraiter l'image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normaliser à [0,1]
    return img_array

# Route pour la page d'accueil (upload)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Vérifier si le fichier est présent dans la requête
        if 'file' not in request.files:
            flash('Aucun fichier trouvé.')
            return redirect(request.url)
        file = request.files['file']
        # Vérifier si un fichier a été sélectionné
        if file.filename == '':
            flash('Aucune image sélectionnée pour le téléchargement.')
            return redirect(request.url)
        # Vérifier si le fichier est autorisé
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                # Prétraiter et prédire
                img_array = preprocess_image(filepath)
                prediction = model.predict(img_array)
                score = prediction[0][0]
                return render_template('result.html', score=score, user_image=filename)
            except Exception as e:
                flash('Erreur lors du traitement de l\'image.')
                print("Erreur de prédiction :", e)
                return redirect(request.url)
        else:
            flash('Types d\'images autorisés : png, jpg, jpeg, gif.')
            return redirect(request.url)
    return render_template('index.html')

# Route pour servir les images téléchargées (si nécessaire)
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Exécuter l'application
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Utiliser le port fourni par Heroku ou 5000 par défaut
    app.run(host='0.0.0.0', port=port)
