# app.py
from flask import Flask, render_template, request
import openai
import pinecone
import argparse

#utilisation des arguments pour securiser les clés api
parse = argparse.ArgumentParser(description="Entrez des clés api: openai, pinecone et env de pinecone. \n python app.py openai_key='' pinecone_key='' pinecone_env=''")
parse.add_argument("openai_key", type=str, help="Entrer la clé de openai")
parse.add_argument("pinecone_key", type=str, help="Entrer la clé de pinecone")
parse.add_argument("pinecone_env", type=str, help="Entrer le nom de l'environnement de pinecone.")
args = parse.parse_args()

# connection sur openai, pinecone
pinecone.init(api_key=args.pinecone_key, environment=args.pinecone_env)
openai.api_key = args.openai_key

# Ouvrir et lire le fichier texte
path = "../transformed_data.txt" # chemin

with open(path, 'r',  encoding="utf8") as file:
    text = file.read()

# Diviser le texte en lignes
lines = text.split("\n")

# Diviser chaque ligne en paires de questions et réponses en fonction du délimiteur '#'
qa_pairs_list = [line.split("#") for line in lines]

qa_pairs = []
for _ in qa_pairs_list:
  if( len(_) == 2):
    qa_pairs.append(_)

qa_pairs

# creation d'une instance client
client = pinecone.Index("lndeducation")

#model
MODEL = "text-embedding-ada-002"

def model_verify(question):
  id_answer = client.query(openai.Embedding.create(input=question, engine=MODEL)['data'][0]['embedding'], 
             top_k = 1, include_metadata=True)["matches"][0]["id"]
  if(id_answer != ""):
    id = (int)(id_answer.split("id-")[1])
    return (qa_pairs[id][0],qa_pairs[id][1])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prompt = ""
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        # Ici, vous pourriez envoyer `prompt` à votre modèle IA et obtenir une réponse
    best_prompt, reponse = model_verify(prompt)
    if( prompt in best_prompt):
        return render_template('index.html', prompt=prompt, response = reponse)
    else:
        return render_template('index.html', response = reponse, prompt = prompt, bon_prompt = best_prompt)

if __name__ == '__main__':
    app.run(debug=True)
